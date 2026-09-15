"""SUPERSEDED by experiments/extract_pstats.py.

This was the first, hardcoded-path pstats dumper (top 100 by cumulative + by
self time for one specific profile). The replacement takes the input/output
paths as arguments and additionally prints callers/callees for the functions of
interest:

    .venv/Scripts/python.exe experiments/extract_pstats.py <in.prof> <out.txt>

Kept only because logs/lipschitz_3layer_nosave_pstats.txt (the original
baseline profile report) was produced with it.
"""
import pstats

prof = "../profiles/lipschitz_3layer_nosave.prof"
out = "../logs/lipschitz_3layer_nosave_pstats.txt"

with open(out, "w") as f:
    stats = pstats.Stats(prof, stream=f)
    stats.strip_dirs().sort_stats("cumulative").print_stats(100)
    f.write("\n\n--- SORTED BY SELF TIME ---\n\n")
    stats.strip_dirs().sort_stats("time").print_stats(100)

print("wrote", out)
