"""Measures how much lc_precise changes between runs that differ only in the random seed.

The precise transformer adds np.random.normal(0, 1e-4) to coefficient A, and
get_lipschitz.py never seeds it, so two identical runs give slightly different bounds.

Part 1 runs get_lipschitz.py (3layer, all five flags set, --no-save) several times.
Part 2 runs single forward passes in this process with a fixed seed, to show that
the same seed repeats exactly and a different seed does not.

Run: python experiments/measure_seed_noise.py [--runs 3]
"""
import argparse
import ast
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import profile_precise_only as pp          # noqa: E402

ROOT = os.path.dirname(HERE)
BEST = {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_VEC_BOUNDARY": "1",
        "PASADO_BATCHED_LSTSQ": "1", "PASADO_BATCHED_GRID": "1", "PASADO_REAL_CUBIC": "0"}


def run_get_lipschitz():
    """One 3layer run; returns the list of lc_precise values (one per epsilon)."""
    env = {**os.environ, **BEST}
    p = subprocess.run([sys.executable, "get_lipschitz.py", "--network", "3layer", "--no-save"],
                       cwd=os.path.join(ROOT, "Section_5_4"), env=env,
                       capture_output=True, text=True)
    p.check_returncode()
    for line in p.stdout.splitlines():
        if line.startswith("lc_precise "):
            return ast.literal_eval(line[len("lc_precise "):])
    raise RuntimeError("lc_precise line not found")


def rel(a, b):
    return [abs(x - y) / abs(x) for x, y in zip(a, b)]


def lc_of(out):
    lb, ub = out.dual.get_lb(), out.dual.get_ub()
    return torch.max(torch.maximum(lb.abs(), ub.abs())).item()


def forward_lc(net, img, eps, seed):
    np.random.seed(seed)
    with torch.no_grad():
        return lc_of(pp.ce.forward_zono_precise(net, pp.ce.make_hazed(img, eps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    print(f"Part 1: {args.runs} unseeded get_lipschitz.py runs, flags {BEST}")
    runs = [run_get_lipschitz() for _ in range(args.runs)]
    diffs = []
    for i in range(1, len(runs)):
        d = rel(runs[0], runs[i])
        diffs.append(d)
        print(f"  run 0 vs run {i}: max relative difference {max(d):.2e}, "
              f"median {sorted(d)[len(d) // 2]:.2e}")
    allmax = max(max(d) for d in diffs)
    print(f"  worst case over all epsilons and run pairs: {allmax:.2e}")
    print(f"  first epsilon, lc_precise per run: {[round(r[0], 6) for r in runs]}")

    print("\nPart 2: single forward passes, seeded")
    pp.set_flags("best")
    net = pp.ce.build_net()
    imgs = pp.load_images(2)
    test_range = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    worst_other_seed = 0.0
    same_seed_exact = True
    for i, img in enumerate(imgs):
        for e in (0, 8, 15):
            eps = test_range[e]
            a = forward_lc(net, img, eps, 1)
            b = forward_lc(net, img, eps, 1)
            c = forward_lc(net, img, eps, 2)
            same_seed_exact &= (a == b)
            worst_other_seed = max(worst_other_seed, abs(a - c) / abs(a))
    print(f"  same seed twice gives exactly the same lc: {same_seed_exact}")
    print(f"  different seed, worst relative difference over 6 cases: {worst_other_seed:.2e}")


if __name__ == "__main__":
    main()
