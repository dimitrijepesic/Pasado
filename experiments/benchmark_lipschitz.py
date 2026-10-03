"""Times get_lipschitz.py end to end for different PASADO_* variants and appends
the results to logs/benchmark_results.csv.

Example: python experiments/benchmark_lipschitz.py --network 3layer --variant original vec_both --runs 3"""
import argparse
import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = os.path.join(REPO_ROOT, "Section_5_4")
CSV_PATH = os.path.join(REPO_ROOT, "logs", "benchmark_results.csv")

VARIANT_ENV = {
    "original":         {"PASADO_VECTORIZED_PRECISE": "0"},
    "vec_corners":      {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_VEC_BOUNDARY": "0"},
    "vec_both":         {"PASADO_VECTORIZED_PRECISE": "1"},
    "vec_both_batched": {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_BATCHED_LSTSQ": "1"},
    # Also batches the sampling grid: no per-neuron loop or .item() left.
    "vec_batched_grid": {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_BATCHED_LSTSQ": "1",
                         "PASADO_BATCHED_GRID": "1"},
    # Real cubic solver. It changes the bounds slightly (up to 5.7e-05 relative),
    # so it is timed as its own variant.
    "vec_grid_realcubic": {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_BATCHED_LSTSQ": "1",
                           "PASADO_BATCHED_GRID": "1", "PASADO_REAL_CUBIC": "1"},
}

CSV_FIELDS = [
    "datetime_utc", "git_commit", "diff_hash", "python", "torch", "numpy",
    "os", "cpu", "cuda", "threads", "network", "variant", "run", "seconds",
    "health_gflops",
]


def machine_health():
    """Fixed matmul probe run immediately before each timed benchmark.

    Several runs in this sprint were silently corrupted by concurrent system
    load (a benchmark taking 1.5-20x longer than physically explicable). This
    records how fast the machine actually was at that moment, so a contaminated
    row is identifiable afterwards instead of being mistaken for a code effect.
    """
    try:
        import torch
        torch.set_default_dtype(torch.float64)
        a = torch.randn(1024, 1024)
        for _ in range(2):          # warm-up
            a @ a
        t0 = time.perf_counter()
        for _ in range(5):
            a @ a
        dt = time.perf_counter() - t0
        return 2 * 5 * 1024 ** 3 / dt / 1e9
    except Exception:
        return float("nan")


def _run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def git_commit():
    r = _run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT)
    return r.stdout.strip() if r.returncode == 0 else "nogit"


def diff_hash():
    """Short stable hash of the current diff of the two sprint source files, so a
    result row is traceable to the exact working-tree state that produced it."""
    import hashlib
    r = _run(["git", "diff", "--",
              "Section_5_4/get_lipschitz.py",
              "forward_mode_tensorized_src/precise_transformer.py"], cwd=REPO_ROOT)
    return hashlib.sha1(r.stdout.encode("utf-8", "replace")).hexdigest()[:10]


def env_meta():
    try:
        import torch
        torch_v = torch.__version__
        cuda = str(torch.cuda.is_available())
        threads = str(torch.get_num_threads())
    except Exception:
        torch_v, cuda, threads = "?", "?", "?"
    try:
        import numpy
        numpy_v = numpy.__version__
    except Exception:
        numpy_v = "?"
    return {
        "git_commit": git_commit(),
        "diff_hash": diff_hash(),
        "python": platform.python_version(),
        "torch": torch_v,
        "numpy": numpy_v,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cuda": cuda,
        "threads": threads,
    }


def append_csv(rows):
    exists = os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def pin_to_pcores():
    """Restrict this process (and therefore its children) to the P-cores.

    This machine is an i5-12450H: 4 performance cores (logical 0-7, hyper-
    threaded) plus 4 efficiency cores (logical 8-11). The Windows scheduler
    freely migrates the benchmark between them, and the E-cores are 2-3x slower
    for this Python-heavy workload - which produced 2-3x run-to-run swings that
    the multi-threaded matmul health probe could not detect (it saturates every
    core and reports high throughput regardless of where the benchmark landed).
    Pinning removes that source of variance. torch's default 8 threads matches
    the 8 P-core logical processors exactly.
    """
    try:
        import psutil
        p = psutil.Process()
        available = p.cpu_affinity()
        pcores = [c for c in available if c < 8]
        if len(pcores) == 8:
            p.cpu_affinity(pcores)
            return pcores
    except Exception:
        pass
    return None


def run_once(network, variant, no_save):
    env = dict(os.environ)
    env.update(VARIANT_ENV[variant])
    cmd = [sys.executable, "get_lipschitz.py", "--network", network]
    if no_save:
        cmd.append("--no-save")
    t0 = time.perf_counter()
    # utf-8/replace: tqdm's progress bytes crash the cp1250 reader threads on
    # Windows (noisy tracebacks in logs; timing itself was never affected).
    proc = subprocess.run(cmd, cwd=SEC, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        raise RuntimeError(f"get_lipschitz.py failed (variant={variant}, net={network})")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", nargs="+", default=["3layer"],
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--variant", nargs="+", default=["original", "vec_both"],
                    choices=list(VARIANT_ENV))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--interleave", action="store_true",
                    help="Alternate variants (A,B,A,B,...) instead of running all "
                         "repetitions of A then all of B. On a machine with drifting "
                         "background load this is the only fair way to compare "
                         "variants: both see the same load pattern.")
    ap.add_argument("--no-save", action="store_true", default=True)
    ap.add_argument("--save", dest="no_save", action="store_false")
    ap.add_argument("--pin-pcores", action="store_true",
                    help="Pin the benchmark to the CPU's performance cores. On a "
                         "hybrid P/E processor the scheduler otherwise migrates "
                         "runs onto the slower E-cores, causing 2-3x swings that "
                         "the health probe cannot see.")
    args = ap.parse_args()

    if args.pin_pcores:
        pinned = pin_to_pcores()
        print(f"pinned to P-cores: {pinned}" if pinned else
              "WARNING: could not pin to P-cores (psutil missing or unexpected topology)")

    meta = env_meta()
    print(f"env: py{meta['python']} torch{meta['torch']} cuda={meta['cuda']} "
          f"threads={meta['threads']} commit={meta['git_commit']} diff={meta['diff_hash']}")

    all_rows = []
    for network in args.network:
        times = {v: [] for v in args.variant}
        # order of (variant, run) pairs: interleaved A,B,A,B,... or grouped A,A,B,B
        if args.interleave:
            schedule = [(v, r) for r in range(1, args.runs + 1) for v in args.variant]
        else:
            schedule = [(v, r) for v in args.variant for r in range(1, args.runs + 1)]

        for variant, run in schedule:
            health = machine_health()
            dt = run_once(network, variant, args.no_save)
            times[variant].append(dt)
            row = dict(meta)
            row.update({
                "datetime_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "network": network, "variant": variant, "run": run,
                "seconds": f"{dt:.3f}", "health_gflops": f"{health:.1f}",
            })
            all_rows.append(row)
            # written after every run, so a killed matrix keeps the runs that finished
            append_csv([row])
            print(f"[{network:7s} {variant:16s} run {run}/{args.runs}] {dt:8.2f}s "
                  f"(machine {health:.0f} GFLOP/s)")

        for variant in args.variant:
            t = times[variant]
            med, lo, hi = statistics.median(t), min(t), max(t)
            note = "single run" if len(t) == 1 else f"n={len(t)}"
            print(f"  -> {network} {variant}: median={med:.2f}s  min={lo:.2f}s  "
                  f"max={hi:.2f}s  ({note})")
        if args.interleave and len(args.variant) == 2:
            a, b = args.variant
            pairs = [ta / tb for ta, tb in zip(times[a], times[b])]
            print(f"  -> {network} paired {a}/{b}: per-pair ratios "
                  f"{', '.join(f'{p:.2f}x' for p in pairs)}  "
                  f"median={statistics.median(pairs):.3f}x")
    print(f"appended {len(all_rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
