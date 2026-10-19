"""Measures the precision vs. runtime trade-off of interval splitting (--n-splits).

For each network, image, epsilon index, and n_splits value (default: 1, 2, 4, 8, 16),
this script divides [0, epsilon] into n_splits equal sub-intervals:
  [i * delta, (i + 1) * delta],  delta = epsilon / n_splits,
evaluates the precise forward pass on each sub-interval, and takes the maximum
local Lipschitz bound across the sub-intervals.

It records:
  - lc_precise:           bound for (network, image, epsilon, n_splits)
  - lc_split1:            baseline bound at n_splits = 1
  - bound_reduction_pct:  (lc_split1 - lc_precise) / lc_split1 * 100
  - tightening_factor:    lc_split1 / lc_precise
  - time_median_ms:       median wall-clock time (ms) across measured repeats
  - time_min_ms, time_max_ms
  - time_multiplier:      time_median_ms / time_split1_median_ms

Run:
  python experiments/measure_split_sweep.py --device cuda --networks 3layer big --out split_sweep.csv
  python experiments/measure_split_sweep.py --device mps --networks 3layer
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

CSV_COLUMNS = [
    "network",
    "device",
    "dtype",
    "image_id",
    "epsilon_index",
    "epsilon",
    "n_splits",
    "lc_precise",
    "lc_split1",
    "bound_reduction_pct",
    "tightening_factor",
    "time_median_ms",
    "time_min_ms",
    "time_max_ms",
    "time_split1_median_ms",
    "time_multiplier",
    "per_split_ms",
    "peak_vram_mb",
]


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
        description="Measure precision vs. runtime trade-off across --n-splits values"
    )
    ap.add_argument(
        "--networks",
        "--network",
        dest="networks",
        nargs="+",
        choices=["3layer", "4layer", "5layer", "big"],
        default=["3layer", "big"],
        help="networks to evaluate; default: 3layer big",
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("PASADO_DEVICE", "cuda"),
        help="torch device (cuda, mps, cpu); default: PASADO_DEVICE or cuda",
    )
    ap.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default=None,
        help="floating-point dtype; default: float32 on mps, float64 otherwise",
    )
    ap.add_argument(
        "--splits",
        type=positive_int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="list of n_splits values to evaluate; default: 1 2 4 8 16",
    )
    ap.add_argument(
        "--num-images",
        type=positive_int,
        default=3,
        help="number of correctly classified test images; default: 3",
    )
    ap.add_argument(
        "--eps-indices",
        type=int,
        nargs="+",
        default=[0, 8, 15],
        help="indices into the 16-epsilon schedule; default: 0 8 15",
    )
    ap.add_argument("--seed", type=numpy_seed, default=7)
    ap.add_argument(
        "--warmup",
        type=non_negative_int,
        default=2,
        help="number of unmeasured warm-up runs per configuration; default: 2",
    )
    ap.add_argument(
        "--repeats",
        type=positive_int,
        default=3,
        help="number of timed repeats per (image, epsilon, n_splits); default: 3",
    )
    ap.add_argument(
        "--real-cubic",
        dest="real_cubic",
        action="store_true",
        default=True,
        help="use the real Viete cubic root solver (default)",
    )
    ap.add_argument(
        "--no-real-cubic",
        dest="real_cubic",
        action="store_false",
        help="use the legacy complex Cardano cubic root solver",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="optional path to write CSV measurements",
    )
    args = ap.parse_args()

    if 1 not in args.splits:
        args.splits = [1] + list(args.splits)
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


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def stamp(device):
    sync_device(device)
    return time.perf_counter()


def load_network_and_images(network_name, device, dtype, num_images):
    import torchvision
    import torchvision.transforms as transforms
    from model import FCN, FCNBig

    section_dir = os.path.join(ROOT, "Section_5_4")
    if network_name == "3layer":
        net = FCN(3)
    elif network_name == "4layer":
        net = FCN(4)
    elif network_name == "5layer":
        net = FCN(5)
    elif network_name == "big":
        net = FCNBig()
    else:
        raise ValueError(f"unknown network: {network_name}")

    weights_path = os.path.join(section_dir, "trained", f"model_{network_name}.pth")
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net = net.eval().to(dtype=dtype, device=device)

    indices_path = os.path.join(section_dir, "trained", f"indices_{network_name}.pth")
    correct_indices = {
        int(index)
        for index in torch.load(indices_path, map_location="cpu")
    }
    if num_images > len(correct_indices):
        raise RuntimeError(
            f"--num-images {num_images} exceeds the {len(correct_indices)} "
            f"available correctly classified images for {network_name}"
        )

    testset = torchvision.datasets.MNIST(
        root=os.path.join(section_dir, "MNIST_Data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    selected = []
    for idx in range(len(testset)):
        if idx in correct_indices:
            image_tensor, _ = testset[idx]
            selected.append((idx, image_tensor.flatten().to(dtype=dtype, device=device)))
            if len(selected) >= num_images:
                break
    return net, selected


def build_forward_precise(pt_mod, net, network_name):
    if network_name == "big":
        def forward_zono_precise(x):
            x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net[1].weight.T) + net[1].bias)
            x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net[3].weight.T) + net[3].bias)
            x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net[5].weight.T) + net[5].bias)
            x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net[7].weight.T) + net[7].bias)
            x = pt_mod.AffineDualZonotope(x, net[9].weight.T) + net[9].bias
            return x
        return forward_zono_precise

    layers = int(network_name[0])

    def forward_zono_precise(x):
        x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
        x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
        if layers >= 4:
            x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net.fc3.weight.T) + net.fc3.bias.data)
        if layers == 5:
            x = pt_mod.PreciseSigmoidDualZonotope(pt_mod.AffineDualZonotope(x, net.fc4.weight.T) + net.fc4.bias.data)
        x = pt_mod.AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
        return x

    return forward_zono_precise


def eval_splits_once(forward_fn, image, eps_value, n_splits, seed, device, dtype):
    from Duals import DualIntervalTensor
    from SimpleZono import HyperDualIntervalToDualZonotope

    delta = eps_value / n_splits
    one = torch.tensor([1.0], dtype=dtype, device=device)
    sub_bounds = []
    with torch.no_grad():
        for split_idx in range(n_splits):
            l_val = split_idx * delta
            r_val = l_val + delta
            interval_eps = DualIntervalTensor(
                real_l=torch.tensor([l_val], dtype=dtype, device=device),
                real_u=torch.tensor([r_val], dtype=dtype, device=device),
            )
            interval_eps.e1_l[0] = 1
            interval_eps.e1_u[0] = 1
            zono_eps = HyperDualIntervalToDualZonotope(interval_eps)
            hazed_zono = (one - zono_eps) * image + zono_eps
            np.random.seed(seed)
            output = forward_fn(hazed_zono)
            bound_val = torch.max(
                torch.maximum(torch.abs(output.dual.get_lb()), torch.abs(output.dual.get_ub()))
            ).item()
            sub_bounds.append(bound_val)
    return max(sub_bounds)


def main():
    args, device, dtype_name, dtype = parse_args()

    import precise_transformer as pt

    torch.set_default_dtype(dtype)
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True
    pt.USE_REAL_CUBIC = args.real_cubic

    eps_schedule = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    all_rows = []

    print(
        f"Split sweep | device={device} | dtype={dtype_name} | real_cubic={args.real_cubic} | "
        f"splits={args.splits} | num_images={args.num_images} | eps_indices={args.eps_indices} | "
        f"warmup={args.warmup} | repeats={args.repeats}"
    )

    for network_name in args.networks:
        net, selected_images = load_network_and_images(
            network_name, device, dtype, args.num_images
        )
        forward_fn = build_forward_precise(pt, net, network_name)

        # Warm-up passes on the first image and first epsilon
        _, first_img = selected_images[0]
        first_eps = eps_schedule[args.eps_indices[0]]
        for _ in range(args.warmup):
            eval_splits_once(forward_fn, first_img, first_eps, 1, args.seed, device, dtype)

        print(f"\n==================== Network: {network_name} ({device}, {dtype_name}) ====================")

        for eps_idx in args.eps_indices:
            eps_val = eps_schedule[eps_idx]
            print(f"\n--- eps_idx={eps_idx} (epsilon={eps_val:.6f}), averaged over {len(selected_images)} images ---")
            print(
                f"{'n_splits':>8s} | {'mean_lc_precise':>16s} | {'reduction_pct':>13s} | "
                f"{'tightening':>11s} | {'time_med_ms':>12s} | {'time_mult':>10s} | "
                f"{'ms_per_split':>12s} | {'peak_vram_mb':>12s}"
            )
            print("-" * 113)

            per_image_split1 = {}
            for image_id, image in selected_images:
                for n_splits in args.splits:
                    times_ms = []
                    peak_mbs = []
                    lc_val = None
                    for _ in range(args.repeats):
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        t0 = stamp(device)
                        lc_val = eval_splits_once(
                            forward_fn, image, eps_val, n_splits, args.seed, device, dtype
                        )
                        t1 = stamp(device)
                        times_ms.append((t1 - t0) * 1000.0)
                        if device.type == "cuda":
                            peak_mbs.append(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
                        else:
                            peak_mbs.append(0.0)

                    t_med = statistics.median(times_ms)
                    t_min = min(times_ms)
                    t_max = max(times_ms)
                    peak_mb = max(peak_mbs)

                    if n_splits == 1:
                        per_image_split1[image_id] = (lc_val, t_med)

                    lc_1, t_1 = per_image_split1[image_id]
                    red_pct = (lc_1 - lc_val) / max(abs(lc_1), 1e-300) * 100.0
                    tight_factor = lc_1 / max(abs(lc_val), 1e-300)
                    t_mult = t_med / max(t_1, 1e-300)

                    all_rows.append(
                        {
                            "network": network_name,
                            "device": str(device),
                            "dtype": dtype_name,
                            "image_id": image_id,
                            "epsilon_index": eps_idx,
                            "epsilon": eps_val,
                            "n_splits": n_splits,
                            "lc_precise": lc_val,
                            "lc_split1": lc_1,
                            "bound_reduction_pct": red_pct,
                            "tightening_factor": tight_factor,
                            "time_median_ms": t_med,
                            "time_min_ms": t_min,
                            "time_max_ms": t_max,
                            "time_split1_median_ms": t_1,
                            "time_multiplier": t_mult,
                            "per_split_ms": t_med / n_splits,
                            "peak_vram_mb": peak_mb,
                        }
                    )

            for n_splits in args.splits:
                sub = [
                    r
                    for r in all_rows
                    if r["network"] == network_name
                    and r["epsilon_index"] == eps_idx
                    and r["n_splits"] == n_splits
                ]
                mean_lc = statistics.mean(r["lc_precise"] for r in sub)
                mean_lc1 = statistics.mean(r["lc_split1"] for r in sub)
                red_pct = (mean_lc1 - mean_lc) / max(abs(mean_lc1), 1e-300) * 100.0
                tight = mean_lc1 / max(abs(mean_lc), 1e-300)
                med_time = statistics.mean(r["time_median_ms"] for r in sub)
                med_time1 = statistics.mean(r["time_split1_median_ms"] for r in sub)
                t_mult = med_time / max(med_time1, 1e-300)
                per_split = med_time / n_splits
                peak_vram = max(r["peak_vram_mb"] for r in sub)
                print(
                    f"{n_splits:8d} | {mean_lc:16.6f} | {red_pct:12.2f}% | "
                    f"{tight:10.2f}x | {med_time:12.2f} | {t_mult:9.2f}x | "
                    f"{per_split:12.2f} | {peak_vram:10.2f}MB"
                )

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
