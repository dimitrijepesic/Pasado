"""Measures the CPU lstsq(gelsd) fallback inside the hybrid precise forward pass.

For every call to lin_reg_tensor_batched, this script records:
  - d2h_x_ms:          GPU -> CPU transfer of x_batch ([n, m, 2])
  - d2h_z_ms:          GPU -> CPU transfer of zs_batch ([n, m])
  - design_ms:         CPU construction of the [n, m, 3] design matrix
  - gelsd_ms:          CPU torch.linalg.lstsq(..., driver='gelsd') solve
  - h2d_ms:            CPU -> GPU transfer of the [n, 3] solution
  - fallback_total_ms: sum of the five steps above

It also records the wall-clock time of each PreciseSigmoidDualZonotope call
(layer_total_ms) and of the entire precise forward pass (precise_total_ms), with
device synchronization at every boundary.

Run:
  python experiments/measure_hybrid_fallback.py --device cuda --network big --out fallback_big.csv
  python experiments/measure_hybrid_fallback.py --device mps --network big
"""
import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def numpy_seed(value):
    parsed = int(value)
    if not 0 <= parsed < 2 ** 32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32 - 1")
    return parsed


def parse_args():
    ap = argparse.ArgumentParser(
        description="Measure CPU fallback time inside the hybrid precise forward pass"
    )
    ap.add_argument("--network", choices=["3layer", "4layer", "5layer", "big"], default="big")
    ap.add_argument("--device", default=os.environ.get("PASADO_DEVICE", "cuda"),
                    help="torch device (cpu, mps, cuda); default: PASADO_DEVICE or cuda")
    ap.add_argument("--dtype", choices=["float32", "float64"], default=None,
                    help="default: float32 on mps, float64 otherwise")
    ap.add_argument("--num-images", type=positive_int, default=3,
                    help="number of correctly classified test images; default: 3")
    ap.add_argument("--eps-indices", type=int, nargs="+", default=[0, 8, 15],
                    help="indices into the 16-epsilon range; default: 0 8 15")
    ap.add_argument("--n-splits", type=positive_int, default=1)
    ap.add_argument("--seed", type=numpy_seed, default=7)
    ap.add_argument("--warmup", type=non_negative_int, default=5,
                    help="number of unmeasured warm-up passes before timing; default: 5")
    ap.add_argument("--repeats", type=positive_int, default=10,
                    help="number of measured repeats per (image, epsilon); default: 10")
    ap.add_argument("--real-cubic", dest="real_cubic", action="store_true", default=True,
                    help="enable the real Viete cubic root solver (default)")
    ap.add_argument("--no-real-cubic", dest="real_cubic", action="store_false",
                    help="use the legacy complex Cardano cubic root solver")
    ap.add_argument("--out", default=None,
                    help="optional path to write per-layer CSV measurements")
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
        ap.error("--dtype float64 is not supported on MPS")

    return args, device, dtype_name


CSV_COLUMNS = [
    "network",
    "image_id",
    "epsilon_index",
    "repeat",
    "layer",
    "n",
    "m",
    "dtype",
    "d2h_x_ms",
    "d2h_z_ms",
    "design_ms",
    "gelsd_ms",
    "h2d_ms",
    "fallback_total_ms",
    "layer_total_ms",
    "precise_total_ms",
    "fallback_share_layer",
    "fallback_share_precise",
]


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def stamp(device):
    sync_device(device)
    return time.perf_counter()


def stats_triplet(values):
    return statistics.median(values), min(values), max(values)


def main():
    args, device, dtype_name = parse_args()

    import torchvision
    import torchvision.transforms as transforms
    from Duals import DualIntervalTensor
    from SimpleZono import HyperDualIntervalToDualZonotope
    import precise_transformer as pt
    from model import FCN, FCNBig

    analysis_dtype = getattr(torch, dtype_name)
    torch.set_default_dtype(analysis_dtype)

    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True
    pt.USE_REAL_CUBIC = args.real_cubic

    section_dir = os.path.join(ROOT, "Section_5_4")
    if args.network == "3layer":
        net = FCN(3)
        layers = 3
    elif args.network == "4layer":
        net = FCN(4)
        layers = 4
    elif args.network == "5layer":
        net = FCN(5)
        layers = 5
    else:
        net = FCNBig()
        layers = None

    model_path = os.path.join(section_dir, "trained", f"model_{args.network}.pth")
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()
    net.to(dtype=analysis_dtype, device=device)

    indices_path = os.path.join(section_dir, "trained", f"indices_{args.network}.pth")
    correct_indices = {
        int(index)
        for index in torch.load(indices_path, map_location="cpu")
    }
    if args.num_images > len(correct_indices):
        raise SystemExit(
            f"--num-images {args.num_images} exceeds the {len(correct_indices)} "
            f"available correctly classified images for {args.network}"
        )

    testset = torchvision.datasets.MNIST(
        root=os.path.join(section_dir, "MNIST_Data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    selected_images = []
    for img_index, (image, _) in enumerate(testloader):
        if img_index in correct_indices:
            selected_images.append(
                (img_index, image.flatten().to(dtype=analysis_dtype, device=device))
            )
            if len(selected_images) == args.num_images:
                break

    if args.network == "big":
        def forward_zono_precise(x):
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
            x = pt.AffineDualZonotope(x, net[9].weight.T) + net[9].bias
            return x
    else:
        def forward_zono_precise(x):
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
            if layers >= 4:
                x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias.data)
            if layers == 5:
                x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias.data)
            x = pt.AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
            return x

    all_epsilons = [10 ** (-k / 4) * 2 for k in range(2, 18)]

    def make_intervals(epsilon):
        delta = epsilon / args.n_splits
        intervals = []
        for i in range(args.n_splits):
            l_ = i * delta
            r_ = l_ + delta
            ie = DualIntervalTensor(
                real_l=torch.tensor([l_], dtype=analysis_dtype, device=device),
                real_u=torch.tensor([r_], dtype=analysis_dtype, device=device),
            )
            ie.e1_l[0] = 1
            ie.e1_u[0] = 1
            intervals.append(ie)
        return intervals

    one = torch.tensor([1], dtype=analysis_dtype, device=device)

    def run_single_precise(img_f, intervals):
        lc_parts = []
        for ie in intervals:
            zono_eps = HyperDualIntervalToDualZonotope(ie)
            hazed_zono = (one - zono_eps) * img_f + zono_eps
            if args.seed is not None:
                np.random.seed(args.seed)
            output = forward_zono_precise(hazed_zono)
            lc_val = torch.max(
                torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))
            ).item()
            lc_parts.append(lc_val)
        return max(lc_parts)

    orig_lstsq = pt.lin_reg_tensor_batched
    orig_precise_sigmoid = pt.PreciseSigmoidDualZonotope
    current_layers = []

    def timed_lin_reg_tensor_batched(x_batch, zs_batch):
        dev = x_batch.device
        n = int(x_batch.shape[0])
        m = int(x_batch.shape[1])

        t0 = stamp(dev)
        if dev.type != "cpu":
            x_cpu = x_batch.cpu()
            t1 = time.perf_counter()
            z_cpu = zs_batch.cpu()
            t2 = time.perf_counter()
        else:
            x_cpu = x_batch
            z_cpu = zs_batch
            t1 = t2 = t0

        if z_cpu.dim() == 2:
            z_cpu = z_cpu.unsqueeze(-1)
        ones = torch.ones(x_cpu.shape[0], x_cpu.shape[1], 1,
                          dtype=x_cpu.dtype, device=x_cpu.device)
        xplusone = torch.cat((ones, x_cpu), dim=2)
        t3 = time.perf_counter()

        R = torch.linalg.lstsq(xplusone, z_cpu, driver="gelsd").solution
        t4 = time.perf_counter()

        if dev.type != "cpu":
            out = R.squeeze(-1).to(dev)
            t5 = stamp(dev)
        else:
            out = R.squeeze(-1)
            t5 = t4

        d2h_x_ms = (t1 - t0) * 1e3
        d2h_z_ms = (t2 - t1) * 1e3
        design_ms = (t3 - t2) * 1e3
        gelsd_ms = (t4 - t3) * 1e3
        h2d_ms = (t5 - t4) * 1e3
        fallback_total_ms = (t5 - t0) * 1e3

        current_layers.append({
            "layer": len(current_layers),
            "n": n,
            "m": m,
            "d2h_x_ms": d2h_x_ms,
            "d2h_z_ms": d2h_z_ms,
            "design_ms": design_ms,
            "gelsd_ms": gelsd_ms,
            "h2d_ms": h2d_ms,
            "fallback_total_ms": fallback_total_ms,
            "layer_total_ms": 0.0,
        })
        return out

    def timed_precise_sigmoid(dz):
        t0 = stamp(device)
        out = orig_precise_sigmoid(dz)
        t1 = stamp(device)
        current_layers[-1]["layer_total_ms"] = (t1 - t0) * 1e3
        return out

    # Sanity check: instrumented pass must produce the exact same lc_precise as uninstrumented.
    first_img_id, first_img = selected_images[0]
    first_intervals = make_intervals(all_epsilons[args.eps_indices[0]])
    with torch.no_grad():
        ref_lc = run_single_precise(first_img, first_intervals)

    rows = []
    pass_summaries = []

    pt.lin_reg_tensor_batched = timed_lin_reg_tensor_batched
    pt.PreciseSigmoidDualZonotope = timed_precise_sigmoid
    try:
        with torch.no_grad():
            current_layers.clear()
            inst_lc = run_single_precise(first_img, first_intervals)
            if inst_lc != ref_lc:
                raise RuntimeError(
                    f"Instrumentation changed lc_precise on image {first_img_id}: "
                    f"ref={ref_lc!r} vs instrumented={inst_lc!r}"
                )

            # Warm-up passes (unrecorded)
            for w in range(args.warmup):
                img_id, img_f = selected_images[w % len(selected_images)]
                eps_idx = args.eps_indices[w % len(args.eps_indices)]
                intervals = make_intervals(all_epsilons[eps_idx])
                current_layers.clear()
                _ = run_single_precise(img_f, intervals)

            # Measured passes: outer loop over repeats so systemic drift is spread evenly
            for rep in range(args.repeats):
                for eps_idx in args.eps_indices:
                    intervals = make_intervals(all_epsilons[eps_idx])
                    for img_id, img_f in selected_images:
                        current_layers.clear()
                        t_pass_0 = stamp(device)
                        _ = run_single_precise(img_f, intervals)
                        t_pass_1 = stamp(device)
                        precise_total_ms = (t_pass_1 - t_pass_0) * 1e3

                        pass_fallback_ms = sum(item["fallback_total_ms"] for item in current_layers)
                        pass_sigmoid_ms = sum(item["layer_total_ms"] for item in current_layers)
                        pass_d2h_x_ms = sum(item["d2h_x_ms"] for item in current_layers)
                        pass_d2h_z_ms = sum(item["d2h_z_ms"] for item in current_layers)
                        pass_design_ms = sum(item["design_ms"] for item in current_layers)
                        pass_gelsd_ms = sum(item["gelsd_ms"] for item in current_layers)
                        pass_h2d_ms = sum(item["h2d_ms"] for item in current_layers)
                        pass_share_precise = (pass_fallback_ms / precise_total_ms * 100.0) if precise_total_ms > 0 else 0.0

                        pass_summaries.append({
                            "network": args.network,
                            "image_id": img_id,
                            "epsilon_index": eps_idx,
                            "repeat": rep,
                            "d2h_x_ms": pass_d2h_x_ms,
                            "d2h_z_ms": pass_d2h_z_ms,
                            "design_ms": pass_design_ms,
                            "gelsd_ms": pass_gelsd_ms,
                            "h2d_ms": pass_h2d_ms,
                            "fallback_total_ms": pass_fallback_ms,
                            "sigmoid_total_ms": pass_sigmoid_ms,
                            "other_precise_ms": precise_total_ms - pass_sigmoid_ms,
                            "precise_total_ms": precise_total_ms,
                            "fallback_share_precise": pass_share_precise,
                        })

                        for item in current_layers:
                            layer_ms = item["layer_total_ms"]
                            fb_ms = item["fallback_total_ms"]
                            share_layer = (fb_ms / layer_ms * 100.0) if layer_ms > 0 else 0.0
                            share_precise = (fb_ms / precise_total_ms * 100.0) if precise_total_ms > 0 else 0.0
                            rows.append({
                                "network": args.network,
                                "image_id": img_id,
                                "epsilon_index": eps_idx,
                                "repeat": rep,
                                "layer": item["layer"],
                                "n": item["n"],
                                "m": item["m"],
                                "dtype": dtype_name,
                                "d2h_x_ms": item["d2h_x_ms"],
                                "d2h_z_ms": item["d2h_z_ms"],
                                "design_ms": item["design_ms"],
                                "gelsd_ms": item["gelsd_ms"],
                                "h2d_ms": item["h2d_ms"],
                                "fallback_total_ms": fb_ms,
                                "layer_total_ms": layer_ms,
                                "precise_total_ms": precise_total_ms,
                                "fallback_share_layer": share_layer,
                                "fallback_share_precise": share_precise,
                            })
    finally:
        pt.lin_reg_tensor_batched = orig_lstsq
        pt.PreciseSigmoidDualZonotope = orig_precise_sigmoid

    n_passes = len(pass_summaries)
    n_layers = len(rows) // n_passes if n_passes else 0

    print("===== Hybrid CPU Fallback Timing =====")
    print("config", {
        "network": args.network,
        "device": str(device),
        "dtype": dtype_name,
        "num_images": len(selected_images),
        "image_ids": [img_id for img_id, _ in selected_images],
        "eps_indices": args.eps_indices,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "measured_passes": n_passes,
        "real_cubic": args.real_cubic,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
    })
    print(f"sanity_check: lc_precise(ref == instrumented) = {ref_lc:.12g} (OK)")

    print("\n--- Per-Pass Summary Across All Measured Forward Passes (median [min, max]) ---")
    pass_metrics = [
        ("precise_total_ms", "Total precise forward (ms)"),
        ("sigmoid_total_ms", "  Sigmoid layers total (ms)"),
        ("other_precise_ms", "  Affine + other precise (ms)"),
        ("fallback_total_ms", "  CPU fallback total (ms)"),
        ("d2h_x_ms", "    D2H x_batch (ms)"),
        ("d2h_z_ms", "    D2H zs_batch (ms)"),
        ("design_ms", "    CPU design matrix (ms)"),
        ("gelsd_ms", "    CPU lstsq gelsd (ms)"),
        ("h2d_ms", "    H2D solution (ms)"),
        ("fallback_share_precise", "Fallback share of precise (%)"),
    ]
    for key, label in pass_metrics:
        med, mn, mx = stats_triplet([p[key] for p in pass_summaries])
        print(f"  {label:32s}: {med:8.3f}  [{mn:8.3f}, {mx:8.3f}]")

    print("\n--- Per-Layer Median Breakdown (ms and % share) ---")
    header = (
        f"{'layer':>5s} {'n':>5s} {'m':>3s} | "
        f"{'d2h_x':>7s} {'d2h_z':>7s} {'design':>7s} {'gelsd':>8s} {'h2d':>7s} | "
        f"{'fb_tot':>8s} {'lyr_tot':>8s} | {'fb/lyr%':>8s} {'fb/prec%':>8s}"
    )
    print(header)
    print("-" * len(header))
    for layer_idx in range(n_layers):
        lrows = [r for r in rows if r["layer"] == layer_idx]
        n_val = lrows[0]["n"]
        m_val = lrows[0]["m"]
        med_d2h_x = statistics.median(r["d2h_x_ms"] for r in lrows)
        med_d2h_z = statistics.median(r["d2h_z_ms"] for r in lrows)
        med_design = statistics.median(r["design_ms"] for r in lrows)
        med_gelsd = statistics.median(r["gelsd_ms"] for r in lrows)
        med_h2d = statistics.median(r["h2d_ms"] for r in lrows)
        med_fb = statistics.median(r["fallback_total_ms"] for r in lrows)
        med_lyr = statistics.median(r["layer_total_ms"] for r in lrows)
        med_sh_lyr = statistics.median(r["fallback_share_layer"] for r in lrows)
        med_sh_prec = statistics.median(r["fallback_share_precise"] for r in lrows)
        print(
            f"{layer_idx:5d} {n_val:5d} {m_val:3d} | "
            f"{med_d2h_x:7.3f} {med_d2h_z:7.3f} {med_design:7.3f} {med_gelsd:8.3f} {med_h2d:7.3f} | "
            f"{med_fb:8.3f} {med_lyr:8.3f} | {med_sh_lyr:7.2f}% {med_sh_prec:7.2f}%"
        )

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nSaved {len(rows)} layer rows to {args.out}")


if __name__ == "__main__":
    main()
