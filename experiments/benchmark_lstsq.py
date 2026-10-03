"""Compares the per-neuron lin_reg_tensor loop with lin_reg_tensor_batched.
Results go to logs/microbenchmark_results.csv. Run on an idle machine.

Run: python experiments/benchmark_lstsq.py"""
import csv
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import torch

torch.set_default_dtype(torch.float64)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import precise_transformer as pt
from batched_lin_reg import lin_reg_tensor_batched
from test_batched_lin_reg import realistic_grids, old_loop

CSV_PATH = os.path.join(REPO, "logs", "microbenchmark_results.csv")
SEED = 12345
WARMUP = 3
REPS = 15


def bench(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def main():
    rows = []
    print(f"torch {torch.__version__}, float64, threads={torch.get_num_threads()}, "
          f"warmup={WARMUP}, reps={REPS}")
    for n in (1, 10, 100, 1024):
        x, zs = realistic_grids(n, SEED + n, torch.float64)

        med_l, lo_l, hi_l = bench(lambda: old_loop(x, zs))
        med_b, lo_b, hi_b = bench(lambda: lin_reg_tensor_batched(x, zs))

        speedup = med_l / med_b
        print(f"n={n:5d}  loop:    median={med_l * 1e3:9.3f} ms  "
              f"[{lo_l * 1e3:.3f}, {hi_l * 1e3:.3f}]")
        print(f"n={n:5d}  batched: median={med_b * 1e3:9.3f} ms  "
              f"[{lo_b * 1e3:.3f}, {hi_b * 1e3:.3f}]   speedup={speedup:.1f}x")

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for variant, med, lo, hi in (("loop", med_l, lo_l, hi_l),
                                     ("batched", med_b, lo_b, hi_b)):
            rows.append(dict(datetime_utc=now, experiment="lin_reg_tensor",
                             variant=variant, n=n, m=25, dtype="float64",
                             warmup=WARMUP, reps=REPS,
                             median_ms=f"{med * 1e3:.4f}",
                             min_ms=f"{lo * 1e3:.4f}", max_ms=f"{hi * 1e3:.4f}"))

    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"appended {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
