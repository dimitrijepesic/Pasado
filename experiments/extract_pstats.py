"""Writes a pstats report from a cProfile file: the top 150 functions by cumulative
and by self time, plus callers and callees of selected functions.

Usage: python extract_pstats.py <in.prof> <out.txt>"""
import sys
import pstats

prof = sys.argv[1]
out = sys.argv[2]

# substrings identifying the functions of interest for caller analysis
FUNCS_OF_INTEREST = [
    "sigmoid_prime_product_tensor",
    "lin_reg_tensor",
    "lstsq",
    "compute_max_error",
    "check_corners_tensor",
    "check_corners",
    "check_nonlinear_boundary_tensor",
    "check_nonlinear_boundary",
    "PreciseSigmoidDualZonotope",
    "get_linspace",
    "sigmoid_prime_times_y",
    "matmul",
    "cartesian_prod",
    "linspace",
    "AffineDualZonotope",
    "SigmoidZonotope",
    "expand",
    "traceify",
]

with open(out, "w") as f:
    stats = pstats.Stats(prof, stream=f)
    f.write("=" * 70 + "\n")
    f.write("TOP 150 BY CUMULATIVE TIME\n")
    f.write("=" * 70 + "\n")
    stats.strip_dirs().sort_stats("cumulative").print_stats(150)

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("TOP 150 BY SELF (INTERNAL) TIME\n")
    f.write("=" * 70 + "\n")
    stats.strip_dirs().sort_stats("time").print_stats(150)

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("CALLERS OF FUNCTIONS OF INTEREST\n")
    f.write("=" * 70 + "\n")
    for name in FUNCS_OF_INTEREST:
        f.write("\n\n---- callers of *%s* ----\n" % name)
        stats.strip_dirs().sort_stats("cumulative").print_callers(name)

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("CALLEES OF FUNCTIONS OF INTEREST\n")
    f.write("=" * 70 + "\n")
    for name in FUNCS_OF_INTEREST:
        f.write("\n\n---- callees of *%s* ----\n" % name)
        stats.strip_dirs().sort_stats("cumulative").print_callees(name)

print("wrote", out)
