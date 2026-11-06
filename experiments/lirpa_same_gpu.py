"""auto_LiRPA (best-first BaB) and Pasado (uniform splits) on the same device, per image,
with the time of every solve recorded (results/lirpa_postfix_2026-11/PROTOCOL.md).

One JSON line per (tool, network, eps index, image) is appended to --out-dir/lirpa.jsonl and
--out-dir/pasado.jsonl; records that are already there are skipped, so the run resumes after
a disconnect. Analysis: experiments/analyze_lirpa_postfix.py.

  python experiments/lirpa_same_gpu.py --networks 3layer 4layer 5layer --eps-indices 0 2 4 \
      --images 0-9 --budget 9 --out-dir DIR
  python experiments/lirpa_same_gpu.py --networks big --eps-indices 0 2 4 --images 0-9 \
      --budget 1 --out-dir DIR
"""
import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import colab_bab_sweep as cbs  # noqa: E402  (sets the default dtype to float64)
import measure_split_sweep as mss  # noqa: E402  (puts the source dirs on sys.path)
from check_sweep_soundness import sampled_max_derivative  # noqa: E402

import batched_splits  # noqa: E402
import precise_transformer as pt  # noqa: E402

EPS = cbs.EPSILONS
PASADO_KS = (1, 3, 5, 7, 9, 16, 32, 64, 128, 256)


def parse_images(text):
    """'0-9' or '0,3,5' -> positions in the list of correctly classified images."""
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def bab_curve_timed(network, img, eps, budget, dev):
    """{K: (bound, cumulative seconds of the first K solves)} for best-first bisection.

    Same algorithm as colab_bab_sweep.bab_curve; additionally times every solve."""
    solve_times = []

    def solve(lo, hi):
        sync(dev); t0 = time.perf_counter()
        value = cbs.solve_interval(network, img, lo, hi, dev)
        sync(dev); solve_times.append(time.perf_counter() - t0)
        return value

    b0 = solve(0.0, eps)
    domains = [(b0, 0.0, eps)]
    curve, solves = {1: (b0, sum(solve_times))}, 1
    while solves + 2 <= budget:
        i = max(range(len(domains)), key=lambda k: domains[k][0])
        _, lo, hi = domains.pop(i)
        mid = (lo + hi) / 2.0
        domains.append((solve(lo, mid), lo, mid))
        domains.append((solve(mid, hi), mid, hi))
        solves += 2
        curve[solves] = (max(d[0] for d in domains), sum(solve_times))
    return curve, solve_times


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                r = json.loads(line)
                keys.add((r["network"], r["eps_index"], r["image_id"]))
    return keys


def env_info(dev):
    info = dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                device=dev, cpu=platform.processor() or platform.machine())
    if dev == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import auto_LiRPA
        info["auto_LiRPA"] = getattr(auto_LiRPA, "__version__", "?")
    except ImportError:
        pass
    info["pasado_commit"] = os.environ.get("PASADO_COMMIT", "")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--networks", nargs="+", default=["3layer", "4layer", "5layer"])
    ap.add_argument("--eps-indices", nargs="+", type=int, default=[0, 2, 4])
    ap.add_argument("--images", default="0-9", help="positions among the correct images, e.g. 0-9")
    ap.add_argument("--budget", type=int, default=9)
    ap.add_argument("--pasado-ks", nargs="+", type=int, default=list(PASADO_KS))
    ap.add_argument("--tools", nargs="+", choices=["lirpa", "pasado"], default=["pasado", "lirpa"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--big-chunk", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    dev = args.device
    os.makedirs(args.out_dir, exist_ok=True)
    positions = parse_images(args.images)
    info = env_info(dev)
    print(json.dumps(info), flush=True)
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
    batched_splits.USE_AUTO_PLANE_FIT = True
    ds = cbs.load_test_images()
    big_chunk = args.big_chunk
    if big_chunk is None:
        free = torch.cuda.mem_get_info()[0] if dev == "cuda" else 0
        big_chunk = 64 if free >= 70 * 2 ** 30 else 32

    for network in args.networks:
        correct = set(int(i) for i in torch.load(
            os.path.join(cbs.SEC, "trained", f"indices_{network}.pth"), map_location="cpu"))
        selected = [i for i in range(len(ds)) if i in correct]
        image_ids = [selected[p] for p in positions]
        net_cpu = cbs.build_network(network)  # for the sampled autograd maximum
        if "pasado" in args.tools:
            pnet, _ = mss.load_network_and_images(network, torch.device(dev), torch.float64, 1)
            layers = batched_splits.extract_affine_layers(pnet, network)
            chunk = big_chunk if network == "big" else None
            warmed = set()
        for tool in args.tools:
            path = os.path.join(args.out_dir, f"{tool}.jsonl")
            done = done_keys(path)
            for ei in args.eps_indices:
                eps = EPS[ei]
                for pos, idx in zip(positions, image_ids):
                    if (network, ei, idx) in done:
                        continue
                    img_cpu = ds[idx][0].flatten().to(torch.float64)
                    sampled = sampled_max_derivative(net_cpu, img_cpu, eps, 4001)
                    rec = dict(tool=tool, network=network, eps_index=ei, epsilon=eps,
                               image_id=idx, image_position=pos, sampled_max=sampled, env=info)
                    if tool == "lirpa":
                        img = img_cpu.unsqueeze(0).to(dev)
                        curve, solve_times = bab_curve_timed(network, img, eps, args.budget, dev)
                        rec.update(budget=args.budget, solve_seconds=solve_times,
                                   curve={str(k): dict(bound=b, seconds=s) for k, (b, s) in curve.items()})
                        bounds = [b for b, _ in curve.values()]
                    else:
                        img = img_cpu.to(dev)
                        runs = {}
                        for k in args.pasado_ks:
                            if k not in warmed:
                                batched_splits.lipschitz_bound(layers, img, eps, k, args.seed, chunk)
                                warmed.add(k)
                            times = []
                            for _ in range(3):
                                sync(dev); t0 = time.perf_counter()
                                b = batched_splits.lipschitz_bound(layers, img, eps, k, args.seed, chunk)
                                sync(dev); times.append(time.perf_counter() - t0)
                            runs[str(k)] = dict(bound=b, seconds=statistics.median(times),
                                                seconds_all=times)
                        rec.update(splits=runs, chunk=chunk or "all",
                                   plane_fit="auto (gelsd < 512 boxes, A2 >= 512)")
                        bounds = [r["bound"] for r in runs.values()]
                    rec["sound_vs_sampled"] = all(b >= sampled for b in bounds)
                    with open(path, "a") as handle:
                        handle.write(json.dumps(rec) + "\n")
                    print(f"{tool:6} {network:6} eps[{ei}] img {idx:5d}: "
                          f"{min(bounds):.6g}..{max(bounds):.6g} (sampled max {sampled:.6g})", flush=True)
                    gc.collect()


if __name__ == "__main__":
    main()
