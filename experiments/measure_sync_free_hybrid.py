"""Compares the baseline hybrid precise pass against a sync-free test version.

This script does NOT modify production code. Instead, it defines a test version of
the four functions in memory and compares it side-by-side (interleaved) with the
production code on the exact same images, epsilons, and seed.

The four changes in the sync-free test version are:
  1. inf_to_nan / filter_range: replace boolean mask indexing (b[guard] = nan,
     which triggers a GPU->CPU nonzero() sync 10 times per layer) with torch.where.
  2. perturb: draw np.random.normal on CPU and add it to A on CPU before copying
     the [n, 3] regression result to the GPU (eliminates a second H2D transfer).
  3. assert bool(((As > 0) | (As < 0)).all()): run this check on CPU before the
     H2D transfer instead of syncing a boolean back from the GPU.
  4. SigmoidZonotope: skip the `if torch.all(lb == ub):` GPU->CPU sync when
     running on GPU (where the input zonotope has non-empty generators).

Run:
  python experiments/measure_sync_free_hybrid.py --device mps --network big
  python experiments/measure_sync_free_hybrid.py --device cuda --network big --out sync_free_big.csv
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
        description="Compare baseline vs sync-free hybrid precise forward pass"
    )
    ap.add_argument("--network", choices=["3layer", "4layer", "5layer", "big"], default="big")
    ap.add_argument("--device", default=os.environ.get("PASADO_DEVICE", "mps"),
                    help="torch device (cpu, mps, cuda); default: PASADO_DEVICE or mps")
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
                    help="optional path to write per-layer CSV measurements for both variants")
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
    "variant",
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
    from SimpleZono import HyperDualIntervalToDualZonotope, Zonotope, traceify
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

    def run_single_precise(img_f, intervals, return_output=False):
        lc_parts = []
        last_output = None
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
            last_output = output
        if return_output:
            return max(lc_parts), last_output
        return max(lc_parts)

    # Save original production functions
    orig_inf_to_nan = pt.inf_to_nan
    orig_filter_range = pt.filter_range
    orig_sigmoid_zono = pt.SigmoidZonotope
    orig_sig_prime_prod = pt.sigmoid_prime_product_tensor
    orig_lstsq = pt.lin_reg_tensor_batched
    orig_precise_sigmoid = pt.PreciseSigmoidDualZonotope

    current_layers = []

    def make_timed_lstsq(fuse_perturb_and_assert=False):
        def timed_lstsq(x_batch, zs_batch):
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

            R = torch.linalg.lstsq(xplusone, z_cpu, driver="gelsd").solution.squeeze(-1)
            if fuse_perturb_and_assert:
                # Fix 2 & 3: draw vectorized perturb on CPU (bit-identical RNG stream
                # to the 1024-call Python comprehension), add to A on CPU, and run
                # the non-zero A assertion on CPU before transferring [n, 3] to GPU.
                perturb_cpu = torch.from_numpy(
                    np.random.normal(0, .0001, size=n)
                ).to(dtype=R.dtype)
                As_cpu = R[:, 1] + perturb_cpu
                assert bool(((As_cpu > 0) | (As_cpu < 0)).all())
                R = torch.stack((As_cpu, R[:, 2], R[:, 0]), dim=1)
            t4 = time.perf_counter()

            if dev.type != "cpu":
                out = R.to(dev)
                t5 = stamp(dev)
            else:
                out = R
                t5 = t4

            current_layers.append({
                "layer": len(current_layers),
                "n": n,
                "m": m,
                "d2h_x_ms": (t1 - t0) * 1e3,
                "d2h_z_ms": (t2 - t1) * 1e3,
                "design_ms": (t3 - t2) * 1e3,
                "gelsd_ms": (t4 - t3) * 1e3,
                "h2d_ms": (t5 - t4) * 1e3,
                "fallback_total_ms": (t5 - t0) * 1e3,
                "layer_total_ms": 0.0,
            })
            return out
        return timed_lstsq

    # Fix 1: torch.where instead of boolean mask indexing
    def sync_free_inf_to_nan(x):
        return torch.where(torch.isinf(x), float("nan"), x)

    def sync_free_filter_range(l, u, x):
        return torch.where((x <= u) & (x >= l), x, float("nan"))

    # Fix 4: SigmoidZonotope without GPU->CPU sync on `torch.all(lb == ub)`
    def sync_free_sigmoid_zonotope(Zono):
        lb = Zono.get_lb()
        ub = Zono.get_ub()
        centers = Zono.centers
        generators = Zono.generators
        if lb.device.type == "cpu" and torch.all(lb == ub):
            centers = torch.sigmoid(lb)
            generators = torch.zeros_like(centers).unsqueeze(0)
            return Zonotope(centers, generators)
        sig_lb = torch.sigmoid(lb)
        sig_ub = torch.sigmoid(ub)
        lambda_opt = torch.min(sig_lb * (1 - sig_lb), sig_ub * (1 - sig_ub))
        mu1 = 0.5 * (sig_ub + sig_lb - lambda_opt * (ub + lb))
        mu2 = 0.5 * (sig_ub - sig_lb - lambda_opt * (ub - lb))
        new_center = (lambda_opt * centers) + mu1
        new_generators_before = lambda_opt * generators
        traceified_mu2 = traceify(mu2)
        new_generators = torch.cat((new_generators_before, traceified_mu2), 0)
        res = Zonotope(new_center, new_generators)
        old_noise_symbols = Zono.get_num_noise_symbs()
        Zono.expand(res.get_num_noise_symbs() - old_noise_symbols)
        return res

    # Fix 2 & 3 consumer: sigmoid_prime_product_tensor using fused CPU perturb+assert
    def sync_free_sigmoid_prime_product_tensor(x, y):
        lx = x.get_lb()
        ux = x.get_ub()
        ly = y.get_lb()
        uy = y.get_ub()
        x_batch = pt.get_linspace_batched(lx, ux, ly, uy)
        xs_grid = x_batch[..., 0]
        ys_grid = x_batch[..., 1]
        zs_batch = (1.0 - torch.sigmoid(xs_grid)) * (torch.sigmoid(xs_grid) * ys_grid)

        # Returns [n, 3] in (As, Bs, Cs) order with perturb and assert already done on CPU
        ABCs_tensor = pt.lin_reg_tensor_batched(x_batch, zs_batch)
        As = ABCs_tensor[:, 0]
        Bs = ABCs_tensor[:, 1]
        Cs = ABCs_tensor[:, 2]

        errors = pt.compute_max_error(lx, ux, ly, uy, None, ABCs_tensor)
        errors = pt.lst_as_tensor(errors)
        errors = traceify(errors)
        new_centers = As * x.centers + Bs * y.centers + Cs
        new_generators = As * x.generators + Bs * y.generators
        new_generators = torch.cat((new_generators, errors), 0)
        return Zonotope(new_centers, new_generators)

    def timed_precise_sigmoid(dz):
        t0 = stamp(device)
        out = orig_precise_sigmoid(dz)
        t1 = stamp(device)
        current_layers[-1]["layer_total_ms"] = (t1 - t0) * 1e3
        return out

    def configure_variant(variant):
        if variant == "baseline":
            pt.inf_to_nan = orig_inf_to_nan
            pt.filter_range = orig_filter_range
            pt.SigmoidZonotope = orig_sigmoid_zono
            pt.sigmoid_prime_product_tensor = orig_sig_prime_prod
            pt.lin_reg_tensor_batched = make_timed_lstsq(fuse_perturb_and_assert=False)
            pt.PreciseSigmoidDualZonotope = timed_precise_sigmoid
        elif variant == "sync_free":
            pt.inf_to_nan = sync_free_inf_to_nan
            pt.filter_range = sync_free_filter_range
            pt.SigmoidZonotope = sync_free_sigmoid_zonotope
            pt.sigmoid_prime_product_tensor = sync_free_sigmoid_prime_product_tensor
            pt.lin_reg_tensor_batched = make_timed_lstsq(fuse_perturb_and_assert=True)
            pt.PreciseSigmoidDualZonotope = timed_precise_sigmoid
        else:
            raise ValueError(f"Unknown variant {variant!r}")

    def restore_originals():
        pt.inf_to_nan = orig_inf_to_nan
        pt.filter_range = orig_filter_range
        pt.SigmoidZonotope = orig_sigmoid_zono
        pt.sigmoid_prime_product_tensor = orig_sig_prime_prod
        pt.lin_reg_tensor_batched = orig_lstsq
        pt.PreciseSigmoidDualZonotope = orig_precise_sigmoid

    rows = []
    summaries = {"baseline": [], "sync_free": []}

    try:
        # 1. Bit-exactness verification across all selected images and epsilons
        with torch.no_grad():
            for eps_idx in args.eps_indices:
                intervals = make_intervals(all_epsilons[eps_idx])
                for img_id, img_f in selected_images:
                    restore_originals()
                    ref_lc, ref_out = run_single_precise(img_f, intervals, return_output=True)

                    for variant in ("baseline", "sync_free"):
                        configure_variant(variant)
                        current_layers.clear()
                        var_lc, var_out = run_single_precise(img_f, intervals, return_output=True)
                        if var_lc != ref_lc:
                            raise RuntimeError(
                                f"{variant} lc_precise mismatch on image {img_id}, eps_idx {eps_idx}: "
                                f"ref={ref_lc!r} vs {variant}={var_lc!r}"
                            )
                        for name, a, b in (
                            ("real.centers", ref_out.real.centers, var_out.real.centers),
                            ("real.generators", ref_out.real.generators, var_out.real.generators),
                            ("dual.centers", ref_out.dual.centers, var_out.dual.centers),
                            ("dual.generators", ref_out.dual.generators, var_out.dual.generators),
                        ):
                            if not torch.equal(a, b):
                                raise RuntimeError(
                                    f"{variant} tensor {name} is not bit-identical on image {img_id}, eps_idx {eps_idx}"
                                )

            # 2. Warm-up passes (both variants)
            for w in range(args.warmup):
                img_id, img_f = selected_images[w % len(selected_images)]
                eps_idx = args.eps_indices[w % len(args.eps_indices)]
                intervals = make_intervals(all_epsilons[eps_idx])
                for variant in ("baseline", "sync_free"):
                    configure_variant(variant)
                    current_layers.clear()
                    _ = run_single_precise(img_f, intervals)

            # 3. Interleaved measured passes: baseline and sync_free alternate on every
            # (repeat, eps_idx, image_id) so thermal/clock drift affects both equally.
            for rep in range(args.repeats):
                for eps_idx in args.eps_indices:
                    intervals = make_intervals(all_epsilons[eps_idx])
                    for img_id, img_f in selected_images:
                        order = ("baseline", "sync_free") if (rep % 2 == 0) else ("sync_free", "baseline")
                        for variant in order:
                            configure_variant(variant)
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
                            pass_share_precise = (
                                pass_fallback_ms / precise_total_ms * 100.0
                                if precise_total_ms > 0 else 0.0
                            )

                            summaries[variant].append({
                                "variant": variant,
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
                                "gpu_sigmoid_non_fb_ms": pass_sigmoid_ms - pass_fallback_ms,
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
                                    "variant": variant,
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
        restore_originals()

    n_passes = len(summaries["baseline"])
    n_layers = (len(rows) // (2 * n_passes)) if n_passes else 0

    print("===== Baseline vs Sync-Free Hybrid Timing =====")
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
        "measured_passes_per_variant": n_passes,
        "real_cubic": args.real_cubic,
        "torch": torch.__version__,
    })
    print("bit_exact_check: ALL (image, epsilon) outputs and lc_precise match bit-for-bit (OK)")

    print("\n--- Per-Pass Median Comparison (ms) ---")
    metrics = [
        ("precise_total_ms", "Total precise forward (ms)"),
        ("other_precise_ms", "  Affine + other non-sigmoid (ms)"),
        ("sigmoid_total_ms", "  Sigmoid layers total (ms)"),
        ("gpu_sigmoid_non_fb_ms", "    Sigmoid GPU part (non-fb, ms)"),
        ("fallback_total_ms", "    CPU fallback total (ms)"),
        ("d2h_x_ms", "      D2H x_batch (ms)"),
        ("d2h_z_ms", "      D2H zs_batch (ms)"),
        ("design_ms", "      CPU design matrix (ms)"),
        ("gelsd_ms", "      CPU lstsq gelsd (+perturb) (ms)"),
        ("h2d_ms", "      H2D solution (ms)"),
        ("fallback_share_precise", "Fallback share of precise (%)"),
    ]
    print(f"{'Metric':36s} | {'Baseline':>10s} | {'Sync-Free':>10s} | {'Diff':>9s} | {'Speedup':>8s}")
    print("-" * 86)
    for key, label in metrics:
        b_med, _, _ = stats_triplet([p[key] for p in summaries["baseline"]])
        s_med, _, _ = stats_triplet([p[key] for p in summaries["sync_free"]])
        diff = s_med - b_med
        if key == "fallback_share_precise":
            print(f"{label:36s} | {b_med:9.2f}% | {s_med:9.2f}% | {diff:+8.2f}% |      n/a")
        else:
            ratio = (b_med / s_med) if s_med > 0 else 1.0
            print(f"{label:36s} | {b_med:10.3f} | {s_med:10.3f} | {diff:+9.3f} | {ratio:7.2f}x")

    for variant in ("baseline", "sync_free"):
        print(f"\n--- Per-Layer Median Breakdown [{variant}] ---")
        header = (
            f"{'layer':>5s} {'n':>5s} {'m':>3s} | "
            f"{'d2h_x':>7s} {'d2h_z':>7s} {'design':>7s} {'gelsd':>8s} {'h2d':>7s} | "
            f"{'fb_tot':>8s} {'lyr_tot':>8s} | {'fb/lyr%':>8s} {'fb/prec%':>8s}"
        )
        print(header)
        print("-" * len(header))
        for layer_idx in range(n_layers):
            lrows = [r for r in rows if r["variant"] == variant and r["layer"] == layer_idx]
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
