"""Where the eager closed-form plane fit overtakes gelsd: the threshold for an automatic
choice between the two, measured on the A100.

The rule below was fixed and committed before the measurement.

Solvers: G = precise_transformer.lin_reg_tensor_batched (gelsd on the CPU, including the
transfers, as in production) and A2 = planar_fit.fit_planes (eager, no TorchInductor).
Both get the same grid and targets.

Protocol: float64 and float32; n = 64, 100, 128, 200, 256, 400, 512, 640, 800, 1024,
2048 boxes per call (well-conditioned boxes, lstsq_contract.g0_bounds); per (n, dtype)
5 warm-up calls and 20 timed calls of each solver, alternating G and A2 size by size,
torch.cuda.synchronize() around every call; median, min and max recorded. Three
rounds over all sizes and dtypes.

Checks, per (n, dtype): coefficients of A2 within 1e-10 (float64) or 1e-3 (float32) of
G relative to the largest coefficient of each box, and the max error D of both planes
within 1e-9 (float64) or 1e-3 (float32) of each other relative to D. A failed check
stops the measurement.

Threshold, per dtype and round: the smallest tested n at which the median time of A2 is
at least 1.2 times faster than G (t_G / t_A2 >= 1.2) and A2 is not slower than G
(t_G / t_A2 >= 1.0) at every larger tested n. A dtype without such an n in any round has
no threshold. The threshold is stable when the three rounds give the same n or n one
step apart in the list; the threshold used is then the largest of the three. Otherwise
it is unstable, and no automatic dispatch is implemented for that dtype.

Run (Colab, A100):
  python experiments/measure_plane_fit_crossover.py --out DIR
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, HERE)

import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402
import compare_lstsq_solvers as h  # noqa: E402
import lstsq_contract as ct  # noqa: E402

SIZES = [64, 100, 128, 200, 256, 400, 512, 640, 800, 1024, 2048]
DTYPES = ["float64", "float32"]
WARMUP, REPEATS, ROUNDS = 5, 20, 3
COEF_RTOL = {"float64": 1e-10, "float32": 1e-3}
D_RTOL = {"float64": 1e-9, "float32": 1e-3}
FAST, NOT_SLOWER = 1.2, 1.0


def timed(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), min(times), max(times)


def check_agreement(grid, z, dtype_name):
    g = pt.lin_reg_tensor_batched(grid, z)
    a = planar_fit.fit_planes(grid, z)
    scale = g.abs().max(dim=1).values.clamp_min(torch.finfo(g.dtype).tiny)
    coef = ((a - g).abs().max(dim=1).values / scale).max().item()
    lx, ux, ly, uy = h.bounds_from_grid(grid)
    d_g = h.max_error(g[:, 1], g[:, 2], g[:, 0], lx, ux, ly, uy)
    d_a = h.max_error(a[:, 1], a[:, 2], a[:, 0], lx, ux, ly, uy)
    d = ((d_a - d_g).abs() / d_g.abs().clamp_min(torch.finfo(g.dtype).tiny)).max().item()
    return coef, d, coef <= COEF_RTOL[dtype_name] and d <= D_RTOL[dtype_name]


def threshold(rows):
    """The rule of the module docstring for one dtype and round; None if there is none."""
    rows = sorted(rows, key=lambda r: r["n"])
    speedups = [r["ms_G"] / r["ms_A2"] for r in rows]
    for i, r in enumerate(rows):
        if speedups[i] >= FAST and all(s >= NOT_SLOWER for s in speedups[i:]):
            return r["n"]
    return None


def environment():
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    return dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                cuda=torch.version.cuda, python=platform.python_version(),
                platform=platform.platform(), commit=commit,
                time=time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-other-gpu", action="store_true",
                    help="run on a GPU other than an A100 (the result is then not the threshold)")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    gpu = torch.cuda.get_device_name(0)
    assert "A100" in gpu or args.allow_other_gpu, f"the threshold is defined on an A100, not {gpu}"
    os.makedirs(args.out, exist_ok=True)
    h.set_production_flags()
    device = torch.device("cuda")

    inputs = {}
    for dtype_name in DTYPES:
        dtype = getattr(torch, dtype_name)
        for k, n in enumerate(SIZES):
            b = ct.g0_bounds("well_conditioned", n, 7000 + k)
            grid, z = h.grid_and_targets(*(t.to(device=device, dtype=dtype) for t in b))
            coef, d, ok = check_agreement(grid, z, dtype_name)
            print(f"check {dtype_name} n={n}: coefficients {coef:.1e}, D {d:.1e}", flush=True)
            assert ok, f"A2 and G disagree at {dtype_name} n={n}: coefficients {coef}, D {d}"
            inputs[(dtype_name, n)] = (grid, z)

    rows = []
    with torch.no_grad():
        for rnd in range(ROUNDS):
            for dtype_name in DTYPES:
                for n in SIZES:
                    grid, z = inputs[(dtype_name, n)]
                    row = dict(round=rnd, dtype=dtype_name, n=n)
                    for name, fn in (("G", pt.lin_reg_tensor_batched), ("A2", planar_fit.fit_planes)):
                        row[f"ms_{name}"], row[f"min_ms_{name}"], row[f"max_ms_{name}"] = timed(fn, grid, z)
                    row["speedup_A2_vs_G"] = row["ms_G"] / row["ms_A2"]
                    rows.append(row)
                    print(f"round {rnd} {dtype_name} n={n:5d}: G {row['ms_G']:7.3f} ms, "
                          f"A2 {row['ms_A2']:7.3f} ms, {row['speedup_A2_vs_G']:5.2f}x", flush=True)

    result = dict(rule="smallest n with t_G/t_A2 >= 1.2 and >= 1.0 at every larger n; stable if "
                       "the 3 rounds agree within one step of the list, then the largest",
                  sizes=SIZES, thresholds={})
    for dtype_name in DTYPES:
        per_round = [threshold([r for r in rows if r["round"] == rnd and r["dtype"] == dtype_name])
                     for rnd in range(ROUNDS)]
        stable = None not in per_round and \
            max(SIZES.index(t) for t in per_round) - min(SIZES.index(t) for t in per_round) <= 1
        result["thresholds"][dtype_name] = dict(per_round=per_round, stable=stable,
                                                threshold=max(per_round) if stable else None)
        print(f"{dtype_name}: per round {per_round}, stable {stable}, "
              f"threshold {result['thresholds'][dtype_name]['threshold']}")

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "crossover.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    with open(os.path.join(args.out, "thresholds.json"), "w") as f:
        json.dump(dict(result, environment=environment(), warmup=WARMUP, repeats=REPEATS,
                       rounds=ROUNDS), f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
