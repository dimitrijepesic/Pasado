"""Superseded by extract_pstats.py. Kept because logs/lipschitz_3layer_nosave_pstats.txt
was produced with it."""
import pstats

prof = "../profiles/lipschitz_3layer_nosave.prof"
out = "../logs/lipschitz_3layer_nosave_pstats.txt"

with open(out, "w") as f:
    stats = pstats.Stats(prof, stream=f)
    stats.strip_dirs().sort_stats("cumulative").print_stats(100)
    f.write("\n\n--- SORTED BY SELF TIME ---\n\n")
    stats.strip_dirs().sort_stats("time").print_stats(100)

print("wrote", out)
