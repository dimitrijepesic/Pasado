"""Tries torch.compile on the pure tensor kernels (corner check, nonlinear boundary
check, the max-objective helper and the batched regression): first-call time,
steady-state time, match with eager mode and graph breaks. Appends rows to
logs/microbenchmark_results.csv."""
import csv
import os
import statistics
import sys
import time

import torch

torch.set_default_dtype(torch.float64)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import precise_transformer as pt

CSV_PATH = os.path.join(REPO, "logs", "microbenchmark_results.csv")
SEED = 12345
WARMUP, REPS = 5, 30


def make_inputs(n):
    g = torch.Generator().manual_seed(SEED + n)
    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 2 + 1e-3
    ly = torch.rand(n, generator=g) * 4 - 2
    uy = ly + torch.rand(n, generator=g) * 2 + 1e-3
    ABCs_t = torch.randn(n, 3, generator=g)
    ABCs_t[:, 0] = ABCs_t[:, 0].abs() + 1e-3
    grids = torch.rand(n, 25, 2, generator=g)
    zs = torch.rand(n, 25, generator=g)
    roots = torch.rand(n, 3, generator=g) * 4 - 2
    roots[torch.rand(n, 3, generator=g) < 0.3] = float("nan")
    return dict(lx=lx, ux=ux, ly=ly, uy=uy, ABCs_t=ABCs_t,
                grids=grids, zs=zs, roots=roots)


def median_time(fn, reps=REPS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def explain_breaks(fn, *args):
    try:
        ex = torch._dynamo.explain(fn)(*args)
        reasons = "; ".join(str(r.reason)[:80] for r in ex.break_reasons) or "-"
        return ex.graph_break_count, reasons
    except Exception as e:
        return -1, f"explain failed: {type(e).__name__}: {e}"


def run_candidate(name, fn, args, n, rows):
    torch._dynamo.reset()
    eager_out = fn(*args)
    t_eager = median_time(lambda: fn(*args))

    breaks, reasons = explain_breaks(fn, *args)
    print(f"[{name} n={n}] eager={t_eager * 1e6:9.1f} us  "
          f"graph_breaks={breaks} ({reasons})")

    for mode_label, kwargs in (("default", {}),
                               ("reduce-overhead", {"mode": "reduce-overhead"})):
        torch._dynamo.reset()
        try:
            cfn = torch.compile(fn, **kwargs)
            t0 = time.perf_counter()
            out = cfn(*args)
            first_call = time.perf_counter() - t0
            t_steady = median_time(lambda: cfn(*args))
            same = torch.allclose(
                torch.nan_to_num(out if isinstance(out, torch.Tensor) else torch.cat([o for o in out]),
                                 nan=-1.0),
                torch.nan_to_num(eager_out if isinstance(eager_out, torch.Tensor) else torch.cat([o for o in eager_out]),
                                 nan=-1.0),
                rtol=1e-12, atol=1e-14)
            speedup = t_eager / t_steady
            print(f"    compile[{mode_label:15s}] first={first_call:7.2f} s  "
                  f"steady={t_steady * 1e6:9.1f} us  speedup={speedup:5.2f}x  "
                  f"equal={'YES' if same else 'NO'}")
            rows.append(dict(experiment=f"compile:{name}", variant=mode_label, n=n,
                             m="", dtype="float64", warmup=WARMUP, reps=REPS,
                             median_ms=f"{t_steady * 1e3:.4f}",
                             min_ms=f"{first_call * 1e3:.1f}",   # first-call in min_ms slot, labeled below
                             max_ms=f"{'OK' if same else 'DIFF'}"))
        except Exception as e:
            print(f"    compile[{mode_label:15s}] FAILED: {type(e).__name__}: {str(e)[:120]}")


def main():
    print(f"torch {torch.__version__}, float64, CPU. NOTE: in CSV rows for "
          f"compile experiments, min_ms column holds FIRST-CALL ms and max_ms "
          f"holds numeric-equality status.")
    rows = []
    from batched_lin_reg import lin_reg_tensor_batched
    for n in (100, 1024):
        inp = make_inputs(n)
        run_candidate(
            "check_corners_tensor",
            pt.check_corners_tensor,
            (inp["ABCs_t"], inp["lx"], inp["ux"], inp["ly"], inp["uy"]), n, rows)
        run_candidate(
            "check_nonlinear_boundary_tensor",
            pt.check_nonlinear_boundary_tensor,
            (inp["lx"].clone(), inp["ux"].clone(), inp["ly"].clone(),
             inp["uy"].clone(), inp["ABCs_t"]), n, rows)
        run_candidate(
            "_max_objective_over_x_candidates",
            pt._max_objective_over_x_candidates,
            (inp["ABCs_t"][:, 0], inp["ABCs_t"][:, 1], inp["ABCs_t"][:, 2],
             inp["roots"], inp["ly"].unsqueeze(1)), n, rows)
        run_candidate(
            "lin_reg_tensor_batched",
            lin_reg_tensor_batched,
            (inp["grids"], inp["zs"]), n, rows)

    if rows:
        exists = os.path.exists(CSV_PATH)
        fields = ["datetime_utc", "experiment", "variant", "n", "m", "dtype",
                  "warmup", "reps", "median_ms", "min_ms", "max_ms"]
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            for r in rows:
                r["datetime_utc"] = now
                w.writerow(r)
        print(f"appended {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
