"""How far n_splits can go while one precise pass stays below a time limit (section F of
results/postfix_rerun_2026-11/PROTOCOL.md), and the end-to-end time of a full standard run.

Ramp: per network and mode, n = --start, 2*start, ... doubling while the median pass time
(one image, one epsilon) stays below --limit-s, up to --max-splits. Batched passes are
chunked so that the peak memory stays below half of the free device memory (the peak per
split is measured at the first n). Only measured passes are reported.

  python experiments/measure_split_ramp.py --device cuda --out ramp.csv
  python experiments/measure_split_ramp.py --device cuda --full-run big=64 3layer=512 --out full.csv
"""
import argparse
import csv
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_split_sweep as mss  # noqa: E402

import batched_splits  # noqa: E402
import precise_transformer as pt  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
MODES = ("sequential", "batched_G", "batched_auto")


def free_bytes(device):
    return torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None


def make_pass(mode, forward, layers, device, dtype, seed):
    def one_pass(img, eps, n, chunk):
        batched_splits.USE_AUTO_PLANE_FIT = mode == "batched_auto"
        if mode == "sequential":
            return mss.eval_splits_once(forward, img, eps, n, seed, device, dtype)
        return batched_splits.lipschitz_bound(layers, img, eps, n, seed, chunk)
    return one_pass


def timed(one_pass, img, eps, n, chunk, device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = mss.stamp(device)
    bound = one_pass(img, eps, n, chunk)
    seconds = mss.stamp(device) - t0
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0
    return seconds, bound, peak


def write(rows, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def ramp(args, device, dtype):
    rows = []
    for network in args.networks:
        net, images = mss.load_network_and_images(network, device, dtype, 1)
        forward = mss.build_forward_precise(pt, net, network)
        layers = batched_splits.extract_affine_layers(net, network)
        image_id, img = images[0]
        eps = EPS[args.eps_index]
        for mode in args.modes:
            one_pass = make_pass(mode, forward, layers, device, dtype, args.seed)
            n, chunk, per_split_mb, last = args.start, None, None, 0.0
            while n <= args.max_splits:
                if mode != "sequential":
                    if per_split_mb is None:
                        chunk = min(n, 64)
                    else:
                        free = free_bytes(device)
                        cap = int(0.5 * free / 2 ** 20 / per_split_mb) if free else 64
                        chunk = max(1, min(n, cap))
                warm = 0 if last > 10 else 1
                reps = 1 if last > 10 else 2
                for _ in range(warm):
                    one_pass(img, eps, n, chunk)
                times, peaks, bound = [], [], None
                for _ in range(reps):
                    s, bound, peak = timed(one_pass, img, eps, n, chunk, device)
                    times.append(s); peaks.append(peak)
                last = statistics.median(times)
                if mode != "sequential" and per_split_mb is None and device.type == "cuda":
                    per_split_mb = max(peaks) / chunk
                rows.append(dict(network=network, mode=mode, n_splits=n, chunk=chunk or "",
                                 image_id=image_id, eps_index=args.eps_index, repeats=reps,
                                 median_s=last, min_s=min(times), max_s=max(times),
                                 peak_vram_mb=max(peaks), bound=bound,
                                 below_limit=last < args.limit_s))
                print(f"{network:6} {mode:12} n={n:6d} chunk={chunk}  {last:8.3f} s  "
                      f"peak {max(peaks):9.1f} MB  bound {bound:.6g}", flush=True)
                write(rows, args.out)
                if last >= args.limit_s:
                    break
                n *= 2
            below = [r["n_splits"] for r in rows if r["network"] == network and r["mode"] == mode
                     and r["below_limit"]]
            print(f"--> {network} {mode}: largest measured n below {args.limit_s:g} s: "
                  f"{max(below) if below else 'none'}", flush=True)
    batched_splits.USE_AUTO_PLANE_FIT = False


def full_run(args, device, dtype):
    rows = []
    for item in args.full_run:
        network, n = item.split("=")
        n = int(n)
        net, images = mss.load_network_and_images(network, device, dtype, 30)
        layers = batched_splits.extract_affine_layers(net, network)
        batched_splits.USE_AUTO_PLANE_FIT = args.full_run_mode == "batched_auto"
        chunk = args.big_chunk if network == "big" else None
        t0 = mss.stamp(device)
        bounds = []
        for eps in EPS:
            for _, img in images:
                bounds.append(batched_splits.lipschitz_bound(layers, img, eps, n, args.seed, chunk))
        seconds = mss.stamp(device) - t0
        rows.append(dict(network=network, n_splits=n, mode=args.full_run_mode, chunk=chunk or n,
                         images=30, epsilons=16, passes=len(bounds), total_s=seconds,
                         mean_bound_eps0=statistics.mean(bounds[:30])))
        print(f"full run {network} n={n} {args.full_run_mode}: {seconds:.1f} s for {len(bounds)} passes",
              flush=True)
        write(rows, args.out)
    batched_splits.USE_AUTO_PLANE_FIT = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--networks", nargs="+", default=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--start", type=int, default=64)
    ap.add_argument("--max-splits", type=int, default=65536)
    ap.add_argument("--limit-s", type=float, default=60.0)
    ap.add_argument("--eps-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--full-run", nargs="+", default=None, metavar="NETWORK=N",
                    help="instead of the ramp: one full 30-image x 16-eps run per NETWORK=N")
    ap.add_argument("--full-run-mode", choices=MODES[1:], default="batched_auto")
    ap.add_argument("--big-chunk", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
    (full_run if args.full_run else ramp)(args, device, dtype)


if __name__ == "__main__":
    main()
