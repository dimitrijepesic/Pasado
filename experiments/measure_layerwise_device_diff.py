"""Measures layer-by-layer CPU vs. device divergence across PreciseSigmoidDualZonotope layers.

For each network, image, epsilon index, and cubic solver ('complex' vs. 'real'),
this script runs the precise forward pass on CPU and on the target device with
identical random seeds and records after every sigmoid layer (plus the final
affine output layer):
  - Tensor differences (max_abs_diff, max_rel_diff, finite_pattern_mismatches)
    for real_centers, real_generators, dual_centers, dual_generators, dual_lb, dual_ub
  - Dual interval width (ub - lb) maximum on CPU and device, and its relative difference
  - Temporary layerwise Lipschitz bound max(|dual_lb|, |dual_ub|) on CPU and device,
    and its relative difference
  - Cubic root solver stages inside check_nonlinear_boundary_tensor:
      * total cubic root candidates (3 * in-range neurons across ly and uy)
      * candidates with a non-zero imaginary part (killed by torch.isreal)
      * candidates surviving torch.isreal
      * candidates surviving inv_sigmoid(s) = log(s / (1 - s)) in (0, 1)
      * roots kept inside the [lx, ux] interval box

Run:
  python experiments/measure_layerwise_device_diff.py --device cuda --out layerwise_diff.csv
  python experiments/measure_layerwise_device_diff.py --device mps --networks 3layer 4layer 5layer big
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))

L_BOUND = -1.0 / (6.0 * np.sqrt(3.0))
U_BOUND = -L_BOUND

CSV_COLUMNS = [
    "network",
    "image_id",
    "epsilon_index",
    "epsilon",
    "dtype",
    "solver",
    "layer",
    "tensor",
    "max_abs_diff",
    "max_rel_diff",
    "finite_pattern_mismatches",
    "cpu_width_max",
    "cuda_width_max",
    "width_rel_diff",
    "cpu_bound",
    "cuda_bound",
    "bound_rel_diff",
    "candidates_cpu",
    "candidates_cuda",
    "nonreal_roots_cpu",
    "nonreal_roots_cuda",
    "after_isreal_cpu",
    "after_isreal_cuda",
    "after_invsig_cpu",
    "after_invsig_cuda",
    "roots_kept_cpu",
    "roots_kept_cuda",
    "final_lc_cpu",
    "final_lc_cuda",
    "final_lc_rel_diff",
]


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def numpy_seed(value):
    parsed = int(value)
    if not 0 <= parsed < 2 ** 32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32 - 1")
    return parsed


def parse_args():
    ap = argparse.ArgumentParser(
        description="Measure layer-by-layer CPU vs. device divergence in the precise forward pass"
    )
    ap.add_argument(
        "--networks",
        "--network",
        dest="networks",
        nargs="+",
        choices=["3layer", "4layer", "5layer", "big"],
        default=["3layer", "4layer", "5layer", "big"],
        help="networks to evaluate; default: 3layer 4layer 5layer big",
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("PASADO_DEVICE", "cuda"),
        help="target torch device (cuda, mps, cpu); default: PASADO_DEVICE or cuda",
    )
    ap.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="floating-point dtype; default: float32 on mps, float64 otherwise",
    )
    ap.add_argument(
        "--num-images",
        type=positive_int,
        default=1,
        help="number of correctly classified test images to evaluate; default: 1",
    )
    ap.add_argument(
        "--eps-indices",
        type=int,
        nargs="+",
        default=[0],
        help="indices into the 16-epsilon schedule; default: 0 (largest epsilon)",
    )
    ap.add_argument(
        "--solvers",
        nargs="+",
        choices=["complex", "real"],
        default=["complex", "real"],
        help="cubic root solvers to evaluate; default: complex real",
    )
    ap.add_argument("--seed", type=numpy_seed, default=7)
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="suppress per-pass tables and print only the aggregate layerwise summary",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="optional path to write detailed layerwise CSV rows",
    )
    args = ap.parse_args()

    if len(set(args.eps_indices)) != len(args.eps_indices):
        ap.error("--eps-indices must not contain duplicates")
    invalid_eps = [i for i in args.eps_indices if not 0 <= i < 16]
    if invalid_eps:
        ap.error(f"--eps-indices entries must be between 0 and 15: {invalid_eps}")

    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        ap.error(f"invalid --device {args.device!r}: {error}")
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda was requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        sys.exit("--device mps was requested but MPS is not available")

    default_dtype = "float32" if device.type == "mps" else "float64"
    dtype_name = args.dtype or default_dtype
    if device.type == "mps" and dtype_name == "float64":
        ap.error("MPS does not support float64; use --dtype float32")

    return args, device, dtype_name, getattr(torch, dtype_name)


def load_network_and_images(network_name, device, dtype, num_images):
    import torchvision
    import torchvision.transforms as transforms
    from model import FCN, FCNBig

    testset = torchvision.datasets.MNIST(
        root=os.path.join(ROOT, "Section_5_4", "MNIST_Data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    if network_name == "3layer":
        net = FCN(3)
        weights_file = "model_3layer.pth"
    elif network_name == "4layer":
        net = FCN(4)
        weights_file = "model_4layer.pth"
    elif network_name == "5layer":
        net = FCN(5)
        weights_file = "model_5layer.pth"
    elif network_name == "big":
        net = FCNBig()
        weights_file = "model_big.pth"
    else:
        raise ValueError(f"unknown network: {network_name}")

    weights_path = os.path.join(ROOT, "Section_5_4", "trained", weights_file)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net = net.eval().to(dtype=dtype, device=device)

    indices_path = os.path.join(ROOT, "Section_5_4", "trained", f"indices_{network_name}.pth")
    correct_indices = {
        int(index)
        for index in torch.load(indices_path, map_location="cpu")
    }
    if num_images > len(correct_indices):
        raise RuntimeError(
            f"--num-images {num_images} exceeds the {len(correct_indices)} "
            f"available correctly classified images for {network_name}"
        )

    selected = []
    for idx in range(len(testset)):
        if idx in correct_indices:
            image_tensor, _ = testset[idx]
            selected.append((idx, image_tensor.flatten().to(dtype=dtype).cpu()))
            if len(selected) >= num_images:
                break

    if len(selected) < num_images:
        raise RuntimeError(f"found only {len(selected)} correctly classified images")
    return net, selected


def get_layer_pairs(net, network_name):
    """Return list of (weight_T, bias, is_sigmoid) layer specifications."""
    if network_name == "big":
        return [
            (net[1].weight.T, net[1].bias, True),
            (net[3].weight.T, net[3].bias, True),
            (net[5].weight.T, net[5].bias, True),
            (net[7].weight.T, net[7].bias, True),
            (net[9].weight.T, net[9].bias.data, False),
        ]
    layers = [
        (net.fc1.weight.T, net.fc1.bias.data, True),
        (net.fc2.weight.T, net.fc2.bias.data, True),
    ]
    if network_name in ("4layer", "5layer"):
        layers.append((net.fc3.weight.T, net.fc3.bias.data, True))
    if network_name == "5layer":
        layers.append((net.fc4.weight.T, net.fc4.bias.data, True))
    layers.append((net.fc_final.weight.T, net.fc_final.bias.data, False))
    return layers


def count_boundary_roots(pt_mod, lx, ux, ly, uy, abcs_tensor, real_cubic):
    """Count candidate roots and survivors at each stage of check_nonlinear_boundary_tensor."""
    a_coeff = abcs_tensor[:, 0]
    in_range_neurons = 0
    nonreal_roots = 0
    after_isreal = 0
    after_invsig = 0
    roots_kept = 0

    for y_bound in (ly, uy):
        r = pt_mod.filter_range(L_BOUND, U_BOUND, pt_mod.inf_to_nan((a_coeff / y_bound).clone()))
        ok = ~torch.isnan(r)
        in_range_neurons += int(ok.sum().item())
        roots_poly = (
            pt_mod.inverse_poly_real_tensor(r.clone())
            if real_cubic
            else pt_mod.inverse_poly_tensor(r.clone())
        )
        for t in roots_poly:
            is_nonreal = (~torch.isreal(t)) & ok
            nonreal_roots += int(is_nonreal.sum().item())
            t_real = t if real_cubic else (t.real * pt_mod.bool_to_nan(t.clone()))
            after_isreal += int((~torch.isnan(t_real)).sum().item())
            t_inv = pt_mod.inv_sigmoid(t_real)
            after_invsig += int((~torch.isnan(t_inv)).sum().item())
            t_kept = pt_mod.filter_range(lx, ux, t_inv.clone())
            roots_kept += int((~torch.isnan(t_kept)).sum().item())

    return {
        "candidates": 3 * in_range_neurons,
        "nonreal_roots": nonreal_roots,
        "after_isreal": after_isreal,
        "after_invsig": after_invsig,
        "roots_kept": roots_kept,
    }


def snapshot_dual_zonotope(dz):
    """Capture CPU float64 copies of DualZonotope tensors and interval/bound summaries."""
    real_c = dz.real.centers.detach().cpu().double().clone()
    real_g = dz.real.generators.detach().cpu().double().clone()
    dual_c = dz.dual.centers.detach().cpu().double().clone()
    dual_g = dz.dual.generators.detach().cpu().double().clone()
    dual_lb = dz.dual.get_lb().detach().cpu().double().clone()
    dual_ub = dz.dual.get_ub().detach().cpu().double().clone()
    width_max = float((dual_ub - dual_lb).max().item())
    bound = float(torch.max(torch.maximum(dual_lb.abs(), dual_ub.abs())).item())
    return {
        "tensors": {
            "real_centers": real_c,
            "real_generators": real_g,
            "dual_centers": dual_c,
            "dual_generators": dual_g,
            "dual_lb": dual_lb,
            "dual_ub": dual_ub,
        },
        "width_max": width_max,
        "bound": bound,
    }


def tensor_diff_stats(cpu_t, dev_t):
    """Return (max_abs_diff, max_rel_diff, finite_pattern_mismatches) between two tensors."""
    a = cpu_t.double()
    b = dev_t.double()
    pattern_mismatches = int(
        (torch.isfinite(a) != torch.isfinite(b)).sum().item()
        + (torch.isnan(a) != torch.isnan(b)).sum().item()
    )
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    if not both_finite.any():
        return 0.0, 0.0, pattern_mismatches
    abs_diff = (a[both_finite] - b[both_finite]).abs()
    rel_diff = abs_diff / a[both_finite].abs().clamp_min(1e-300)
    return float(abs_diff.max().item()), float(rel_diff.max().item()), pattern_mismatches


def scalar_rel_diff(cpu_val, dev_val):
    return abs(cpu_val - dev_val) / max(abs(cpu_val), 1e-300)


def percentile_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 95)),
        float(np.max(arr)),
    )


def print_aggregate_summary(all_rows, networks, solvers, device):
    """Print median / p95 / max layerwise CPU vs. device summary across all (image, epsilon) pairs."""
    print(f"\n==================== AGGREGATE LAYERWISE SUMMARY (CPU vs. {device}) ====================")
    for network_name in networks:
        for solver_name in solvers:
            sub = [
                r
                for r in all_rows
                if r["network"] == network_name
                and r["solver"] == solver_name
                and r["tensor"] == "dual_generators"
            ]
            if not sub:
                continue
            layers_order = []
            for r in sub:
                if r["layer"] not in layers_order:
                    layers_order.append(r["layer"])
            n_pairs = len(sub) // len(layers_order)
            print(
                f"\nNetwork={network_name} | Solver={solver_name} | "
                f"(image, epsilon) pairs={n_pairs}"
            )
            print(
                f"{'layer':>12s} | {'nonreal_dev(med/max)':>20s} | "
                f"{'kept_diff(med/max)':>18s} | "
                f"{'dual_g_rel (med / p95 / max)':>32s} | "
                f"{'bound_rel (med / p95 / max)':>32s}"
            )
            print("-" * 126)
            for layer_label in layers_order:
                l_rows = [r for r in sub if r["layer"] == layer_label]
                nonreal_dev = [r["nonreal_roots_cuda"] for r in l_rows]
                kept_diff = [abs(r["roots_kept_cpu"] - r["roots_kept_cuda"]) for r in l_rows]
                dg_med, dg_p95, dg_max = percentile_stats([r["max_rel_diff"] for r in l_rows])
                b_med, b_p95, b_max = percentile_stats([r["bound_rel_diff"] for r in l_rows])
                nr_med = int(round(float(np.median(nonreal_dev))))
                nr_max = int(max(nonreal_dev))
                kd_med = int(round(float(np.median(kept_diff))))
                kd_max = int(max(kept_diff))
                print(
                    f"{layer_label:>12s} | "
                    f"{nr_med:9d} / {nr_max:<8d} | "
                    f"{kd_med:8d} / {kd_max:<7d} | "
                    f"{dg_med:9.2e} / {dg_p95:9.2e} / {dg_max:9.2e} | "
                    f"{b_med:9.2e} / {b_p95:9.2e} / {b_max:9.2e}"
                )


def run_single_pass(pt_mod, net, network_name, image_cpu, eps_value, run_device, dtype, seed, real_cubic):
    """Run one precise forward pass on run_device and record per-layer snapshots + root counts."""
    from Duals import DualIntervalTensor
    from SimpleZono import HyperDualIntervalToDualZonotope

    pt_mod.USE_REAL_CUBIC = real_cubic
    orig_boundary = pt_mod.check_nonlinear_boundary_tensor
    boundary_stats = []

    def wrapped_boundary(lx, ux, ly, uy, abcs_tensor):
        boundary_stats.append(
            count_boundary_roots(pt_mod, lx, ux, ly, uy, abcs_tensor, real_cubic)
        )
        return orig_boundary(lx, ux, ly, uy, abcs_tensor)

    pt_mod.check_nonlinear_boundary_tensor = wrapped_boundary
    layer_snapshots = []
    try:
        net_dev = net.to(device=run_device, dtype=dtype)
        image = image_cpu.to(device=run_device, dtype=dtype)
        one = torch.tensor([1.0], dtype=dtype, device=run_device)
        interval_eps = DualIntervalTensor(
            real_l=torch.tensor([0.0], dtype=dtype, device=run_device),
            real_u=torch.tensor([eps_value], dtype=dtype, device=run_device),
        )
        interval_eps.e1_l[0] = 1
        interval_eps.e1_u[0] = 1
        zono_eps = HyperDualIntervalToDualZonotope(interval_eps)
        x = (one - zono_eps) * image + zono_eps

        np.random.seed(seed)
        torch.manual_seed(seed)

        layer_specs = get_layer_pairs(net_dev, network_name)
        sigmoid_idx = 0
        with torch.no_grad():
            for weight_t, bias, is_sigmoid in layer_specs:
                x = pt_mod.AffineDualZonotope(x, weight_t) + bias
                if is_sigmoid:
                    x = pt_mod.PreciseSigmoidDualZonotope(x)
                    snap = snapshot_dual_zonotope(x)
                    snap["layer"] = str(sigmoid_idx)
                    snap["roots"] = boundary_stats[-1]
                    layer_snapshots.append(snap)
                    sigmoid_idx += 1
                else:
                    snap = snapshot_dual_zonotope(x)
                    snap["layer"] = "final_affine"
                    snap["roots"] = {
                        "candidates": 0,
                        "nonreal_roots": 0,
                        "after_isreal": 0,
                        "after_invsig": 0,
                        "roots_kept": 0,
                    }
                    layer_snapshots.append(snap)
    finally:
        pt_mod.check_nonlinear_boundary_tensor = orig_boundary

    final_lc = layer_snapshots[-1]["bound"]
    return layer_snapshots, final_lc


def main():
    args, device, dtype_name, dtype = parse_args()

    import precise_transformer as pt

    torch.set_default_dtype(dtype)
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True

    eps_schedule = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    all_rows = []

    print(
        f"Layerwise CPU vs. {device} comparison | dtype={dtype_name} | "
        f"seed={args.seed} | num_images={args.num_images} | eps_indices={args.eps_indices}"
    )

    for network_name in args.networks:
        net_cpu, selected_images = load_network_and_images(
            network_name, torch.device("cpu"), dtype, args.num_images
        )
        for image_id, image_cpu in selected_images:
            for eps_idx in args.eps_indices:
                eps_val = eps_schedule[eps_idx]
                cpu_snaps_by_solver = {}
                for solver_name in args.solvers:
                    real_cubic = solver_name == "real"
                    cpu_snaps, cpu_lc = run_single_pass(
                        pt,
                        net_cpu,
                        network_name,
                        image_cpu,
                        eps_val,
                        torch.device("cpu"),
                        dtype,
                        args.seed,
                        real_cubic,
                    )
                    cpu_snaps_by_solver[solver_name] = (cpu_snaps, cpu_lc)
                    dev_snaps, dev_lc = run_single_pass(
                        pt,
                        net_cpu,
                        network_name,
                        image_cpu,
                        eps_val,
                        device,
                        dtype,
                        args.seed,
                        real_cubic,
                    )
                    final_lc_rel = scalar_rel_diff(cpu_lc, dev_lc)

                    if not args.summary_only:
                        print(
                            f"\n=== network={network_name} | image={image_id} | "
                            f"eps_idx={eps_idx} ({eps_val:.6f}) | solver={solver_name} "
                            f"(CPU vs. {device}) ==="
                        )
                        print(
                            f"Final lc_precise: CPU={cpu_lc:.8f}, {device}={dev_lc:.8f}, "
                            f"rel_diff={final_lc_rel:.3e}"
                        )
                        print(
                            f"{'layer':>12s} | {'cand':>5s} | {'nonreal(c|d)':>14s} | "
                            f"{'invsig(c|d)':>13s} | {'kept(c|d)':>11s} | "
                            f"{'dual_g_rel':>10s} | {'width_rel':>10s} | "
                            f"{'cpu_bound':>11s} | {'dev_bound':>11s} | {'bound_rel':>10s}"
                        )
                        print("-" * 132)

                    for c_snap, d_snap in zip(cpu_snaps, dev_snaps):
                        layer_label = c_snap["layer"]
                        c_roots = c_snap["roots"]
                        d_roots = d_snap["roots"]
                        w_rel = scalar_rel_diff(c_snap["width_max"], d_snap["width_max"])
                        b_rel = scalar_rel_diff(c_snap["bound"], d_snap["bound"])

                        tensor_stats_map = {}
                        for tensor_name in (
                            "real_centers",
                            "real_generators",
                            "dual_centers",
                            "dual_generators",
                            "dual_lb",
                            "dual_ub",
                        ):
                            max_abs, max_rel, pat_mis = tensor_diff_stats(
                                c_snap["tensors"][tensor_name],
                                d_snap["tensors"][tensor_name],
                            )
                            tensor_stats_map[tensor_name] = (max_abs, max_rel, pat_mis)
                            all_rows.append(
                                {
                                    "network": network_name,
                                    "image_id": image_id,
                                    "epsilon_index": eps_idx,
                                    "epsilon": eps_val,
                                    "dtype": dtype_name,
                                    "solver": solver_name,
                                    "layer": layer_label,
                                    "tensor": tensor_name,
                                    "max_abs_diff": max_abs,
                                    "max_rel_diff": max_rel,
                                    "finite_pattern_mismatches": pat_mis,
                                    "cpu_width_max": c_snap["width_max"],
                                    "cuda_width_max": d_snap["width_max"],
                                    "width_rel_diff": w_rel,
                                    "cpu_bound": c_snap["bound"],
                                    "cuda_bound": d_snap["bound"],
                                    "bound_rel_diff": b_rel,
                                    "candidates_cpu": c_roots["candidates"],
                                    "candidates_cuda": d_roots["candidates"],
                                    "nonreal_roots_cpu": c_roots["nonreal_roots"],
                                    "nonreal_roots_cuda": d_roots["nonreal_roots"],
                                    "after_isreal_cpu": c_roots["after_isreal"],
                                    "after_isreal_cuda": d_roots["after_isreal"],
                                    "after_invsig_cpu": c_roots["after_invsig"],
                                    "after_invsig_cuda": d_roots["after_invsig"],
                                    "roots_kept_cpu": c_roots["roots_kept"],
                                    "roots_kept_cuda": d_roots["roots_kept"],
                                    "final_lc_cpu": cpu_lc,
                                    "final_lc_cuda": dev_lc,
                                    "final_lc_rel_diff": final_lc_rel,
                                }
                            )

                        dual_g_rel = tensor_stats_map["dual_generators"][1]
                        if not args.summary_only:
                            print(
                                f"{layer_label:>12s} | "
                                f"{c_roots['candidates']:5d} | "
                                f"{c_roots['nonreal_roots']:6d}|{d_roots['nonreal_roots']:<6d} | "
                                f"{c_roots['after_invsig']:5d}|{d_roots['after_invsig']:<6d} | "
                                f"{c_roots['roots_kept']:4d}|{d_roots['roots_kept']:<5d} | "
                                f"{dual_g_rel:10.3e} | "
                                f"{w_rel:10.3e} | "
                                f"{c_snap['bound']:11.5f} | "
                                f"{d_snap['bound']:11.5f} | "
                                f"{b_rel:10.3e}"
                            )

                if (
                    not args.summary_only
                    and "complex" in cpu_snaps_by_solver
                    and "real" in cpu_snaps_by_solver
                ):
                    comp_snaps, comp_lc = cpu_snaps_by_solver["complex"]
                    real_snaps, real_lc = cpu_snaps_by_solver["real"]
                    solver_lc_rel = scalar_rel_diff(real_lc, comp_lc)
                    print(
                        f"\n--- network={network_name} | image={image_id} | "
                        f"eps_idx={eps_idx} ({eps_val:.6f}) | "
                        f"CPU complex (theirs) vs. CPU real (Viete) ---"
                    )
                    print(
                        f"Final lc_precise: complex={comp_lc:.8f}, real={real_lc:.8f}, "
                        f"rel_diff={solver_lc_rel:.3e}"
                    )
                    print(
                        f"{'layer':>12s} | {'cand':>5s} | {'invsig(c|r)':>13s} | "
                        f"{'kept(c|r)':>11s} | {'dual_g_rel':>10s} | "
                        f"{'width_rel':>10s} | {'comp_bound':>11s} | "
                        f"{'real_bound':>11s} | {'bound_rel':>10s}"
                    )
                    print("-" * 116)
                    for c_snap, r_snap in zip(comp_snaps, real_snaps):
                        c_roots = c_snap["roots"]
                        r_roots = r_snap["roots"]
                        _, dg_rel, _ = tensor_diff_stats(
                            r_snap["tensors"]["dual_generators"],
                            c_snap["tensors"]["dual_generators"],
                        )
                        w_rel = scalar_rel_diff(r_snap["width_max"], c_snap["width_max"])
                        b_rel = scalar_rel_diff(r_snap["bound"], c_snap["bound"])
                        print(
                            f"{c_snap['layer']:>12s} | "
                            f"{c_roots['candidates']:5d} | "
                            f"{c_roots['after_invsig']:5d}|{r_roots['after_invsig']:<6d} | "
                            f"{c_roots['roots_kept']:4d}|{r_roots['roots_kept']:<5d} | "
                            f"{dg_rel:10.3e} | "
                            f"{w_rel:10.3e} | "
                            f"{c_snap['bound']:11.5f} | "
                            f"{r_snap['bound']:11.5f} | "
                            f"{b_rel:10.3e}"
                        )

    if args.summary_only or (args.num_images * len(args.eps_indices) > 1):
        print_aggregate_summary(all_rows, args.networks, args.solvers, device)

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
