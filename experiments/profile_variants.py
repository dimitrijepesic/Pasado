"""Profile the precise path under several selector variants and diff them.

Purpose. After batching the sampling grid, the previously dominant
`get_linspace` (~44 % of sigmoid_prime_product_tensor on 3layer, plus 4 .item()
syncs per neuron) is gone, and nothing tells us what the new bottleneck is. The
question this answers is binary and decides whether CPU work continues:

  * still Python overhead on top  -> more vectorization is worth doing;
  * pure BLAS on top (AffineZonotope matmul, abs/sum reductions)
    -> CPU side is finished and everything further belongs on the GPU.

Method. Runs get_lipschitz.py under cProfile once per variant (variants are
selected purely through the PASADO_* environment variables, so no source is
edited), then prints one side-by-side table of cumulative and self time for the
functions on the precise path. A side-by-side diff is the point: an absolute
profile alone cannot show what *moved*.

Reading the numbers -- important caveat. cProfile charges a fixed cost per
Python call, so it systematically overstates code that makes many cheap calls
and understates a few large tensor operations. That bias is exactly backwards
for judging our optimizations: it flatters the un-optimized variant's
bottlenecks and can make the optimized one look BLAS-bound sooner than it is.
Use these numbers for *attribution* (which function, and did it move), never as
wall-clock -- the interleaved benchmark harness is the authority on time.

Usage (from the Pasado repo root):
    .venv\\Scripts\\python.exe experiments\\profile_variants.py
    .venv\\Scripts\\python.exe experiments\\profile_variants.py \\
        --network 3layer --variants vec_both_batched vec_batched_grid
"""
import argparse
import os
import pstats
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = os.path.join(REPO, "Section_5_4")
PROFILES = os.path.join(REPO, "profiles")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")

# Same mapping the benchmark harness uses, so a profile and a timing run of the
# same variant name are guaranteed to exercise the same code.
VARIANT_ENV = {
    "original":         {"PASADO_VECTORIZED_PRECISE": "0"},
    "vec_corners":      {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_VEC_BOUNDARY": "0"},
    "vec_both":         {"PASADO_VECTORIZED_PRECISE": "1"},
    "vec_both_batched": {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_BATCHED_LSTSQ": "1"},
    "vec_batched_grid": {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_BATCHED_LSTSQ": "1",
                         "PASADO_BATCHED_GRID": "1"},
}

# Rows of the report. Matched as substrings against "file:line(function)", so
# both our functions and the torch/aten entry points they call are covered.
ROWS = [
    "forward_zono_precise",
    "PreciseSigmoidDualZonotope",
    "sigmoid_prime_product_tensor",
    "get_linspace_batched",
    "get_linspace",
    "_linspace_rows",
    "compute_max_error",
    "check_corners_tensor",
    "check_nonlinear_boundary_tensor",
    "check_corners",
    "check_nonlinear_boundary",
    "lin_reg_tensor_batched",
    "lin_reg_tensor",
    "linalg_lstsq",
    "inverse_poly_tensor",
    "sigmoid_prime_times_y",
    "objective_fn",
    "AffineZonotope",
    "AffineDualZonotope",
    "get_coeff_abs",
    "traceify",
    "cartesian_prod",
    "linspace",
    "'item'",
    "'sigmoid'",
    "'abs'",
    "'sum'",
    "'matmul'",
    "'stack'",
    "'cat'",
]


def run_profile(network, variant, force):
    """cProfile one variant; returns the .prof path. Skips an existing profile
    unless --force, so re-running the report is cheap."""
    out = os.path.join(PROFILES, f"prof_{network}_{variant}.prof")
    if os.path.exists(out) and not force:
        print(f"  [{variant}] reusing {os.path.relpath(out, REPO)}")
        return out
    os.makedirs(PROFILES, exist_ok=True)
    env = dict(os.environ)
    env.update(VARIANT_ENV[variant])
    # --no-save keeps torch.save out of the profile; an early attempt in this
    # project was dominated by ~93s of serialization and was useless for
    # compute attribution.
    cmd = [PY, "-m", "cProfile", "-o", out, "get_lipschitz.py",
           "--network", network, "--no-save"]
    print(f"  [{variant}] profiling ...", end="", flush=True)
    begin = time.perf_counter()
    proc = subprocess.run(cmd, cwd=SEC, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(" FAILED")
        sys.stderr.write((proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:])
        raise RuntimeError(f"profiling failed for {variant}")
    print(f" {time.perf_counter() - begin:.0f}s (profiler time, not wall-clock)")
    return out


def collect(prof_path):
    """{row_label: (cumulative, self, ncalls)} for the ROWS substrings.

    Several profile entries can match one label (e.g. torch and aten layers of
    the same op); they are summed, and totals are therefore upper bounds on the
    concept the label names rather than a single function's cost.
    """
    st = pstats.Stats(prof_path)
    st.strip_dirs()
    agg = {r: [0.0, 0.0, 0] for r in ROWS}
    total = 0.0
    for func, (cc, nc, tt, ct, _callers) in st.stats.items():
        key = f"{func[0]}:{func[1]}({func[2]})"
        total = max(total, ct)
        for r in ROWS:
            if r in key:
                agg[r][0] += ct
                agg[r][1] += tt
                agg[r][2] += nc
    return agg, st.total_tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="3layer",
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--variants", nargs="+",
                    default=["vec_both_batched", "vec_batched_grid"],
                    choices=list(VARIANT_ENV))
    ap.add_argument("--force", action="store_true",
                    help="re-profile even if a .prof already exists")
    args = ap.parse_args()

    print(f"network={args.network}  variants={args.variants}")
    profs = {v: run_profile(args.network, v, args.force) for v in args.variants}
    data = {v: collect(p) for v, p in profs.items()}

    print(f"\n{'':38s}" + "".join(f"{v:>26s}" for v in args.variants))
    print(f"{'':38s}" + "".join(f"{'cum':>9s}{'self':>9s}{'calls':>8s}"
                                for _ in args.variants))
    print("-" * (38 + 26 * len(args.variants)))
    for r in ROWS:
        cells = ""
        present = False
        for v in args.variants:
            cum, slf, nc = data[v][0][r]
            if nc:
                present = True
                cells += f"{cum:9.2f}{slf:9.2f}{nc:8d}"
            else:
                cells += f"{'-':>9s}{'-':>9s}{'-':>8s}"
        if present:
            print(f"{r:38s}{cells}")

    print("-" * (38 + 26 * len(args.variants)))
    print(f"{'TOTAL (profiler time)':38s}"
          + "".join(f"{data[v][1]:9.2f}{'':9s}{'':8s}" for v in args.variants))

    print("\nTop 12 by SELF time, fully optimized variant "
          "(what is actually left):")
    st = pstats.Stats(profs[args.variants[-1]])
    st.strip_dirs().sort_stats("time").print_stats(12)


if __name__ == "__main__":
    main()
