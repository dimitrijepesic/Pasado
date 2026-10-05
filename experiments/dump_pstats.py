"""Superseded by extract_pstats.py. Kept because logs/lipschitz_3layer_nosave_pstats.txt
was produced with it."""
import argparse
import pstats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile", help="input cProfile .prof file")
    ap.add_argument("out", help="output text report")
    args = ap.parse_args()

    with open(args.out, "w") as f:
        stats = pstats.Stats(args.profile, stream=f)
        stats.strip_dirs().sort_stats("cumulative").print_stats(100)
        f.write("\n\n--- SORTED BY SELF TIME ---\n\n")
        stats.strip_dirs().sort_stats("time").print_stats(100)

    print("wrote", args.out)


if __name__ == "__main__":
    main()
