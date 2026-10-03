"""Profile ONLY the precise zonotope forward pass (forward_zono_precise).

Why. Earlier profiles ran all of Section_5_4/get_lipschitz.py, which per image
also runs the plain zonotope and the interval analysis. Percentages from those
runs (for example "gelsd 1.7 %, matmul 45 %") therefore have the whole run as
their denominator, not the precise path alone. The precise path is the part
that is being optimized and ported to GPU, so this script times and profiles
that forward pass in isolation.

What it does (3layer, float64 like the benchmark):
  1. Sets all five PASADO_* selectors explicitly, so the result never depends
     on whatever is in the environment.
  2. Re-seeds np.random before every forward. The transformer perturbs
     coefficient A with a random draw, so without a fixed seed two runs would
     not do identical work.
  3. Runs one warm-up forward, then a wall-clock pass without the profiler.
  4. Runs a second, identical pass under cProfile.
  5. Prints the top-N functions by self time (tottime) and by cumulative time.

How to read it. cProfile adds a fixed cost per Python call, so it overstates
code that makes many cheap calls. Use its numbers for relative attribution
(which function takes what share), and the wall-clock line for actual time.

Usage (from the Pasado repo root, venv active):
    python experiments/profile_precise_only.py
    python experiments/profile_precise_only.py --n-images 5 --eps-indices 0 4 8 12 --top 20
    python experiments/profile_precise_only.py --flags original
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                      # makes correctness_e2e importable
# Importing correctness_e2e also puts the source folders on sys.path and sets
# the default dtype to float64, which the precise_transformer import below needs.
import correctness_e2e as ce                  # noqa: E402

import precise_transformer as pt              # noqa: E402

# Named selector sets. Every set lists all five flags, so a run is fully
# determined by its name. "best" is the fastest bit-identical CPU configuration
# (real_cubic stays off because it changes the numerics); "original" is the
# unoptimized code path.
FLAGS = {
    "best":     dict(vec=True,  bnd=True,  lstsq=True,  grid=True,  cubic=False),
    "original": dict(vec=False, bnd=False, lstsq=False, grid=False, cubic=False),
}

SEED = 12345


def set_flags(name):
    """Apply the named selector set to precise_transformer and return it.

    The selectors are module-level variables read at call time, so setting them
    here takes effect immediately, without restarting the process.
    """
    f = FLAGS[name]
    pt.USE_VECTORIZED_PRECISE = f["vec"]
    pt.USE_VECTORIZED_BOUNDARY = f["bnd"]
    pt.USE_BATCHED_LSTSQ = f["lstsq"]
    pt.USE_BATCHED_GRID = f["grid"]
    pt.USE_REAL_CUBIC = f["cubic"]
    return f


def load_images(n_images):
    """Return the first n_images correctly classified 3layer test images.

    Images are flattened to 784 values. "Correctly classified" matches the
    selection used by Section_5_4/get_lipschitz.py (the saved indices file).
    """
    import torchvision
    import torchvision.transforms as transforms
    testset = torchvision.datasets.MNIST(
        root=os.path.join(ce.REPO, "Section_5_4", "MNIST_Data"), train=False,
        download=True, transform=transforms.ToTensor())
    correct = torch.load(
        os.path.join(ce.REPO, "Section_5_4", "trained", "indices_3layer.pth"),
        map_location="cpu")
    imgs, idx = [], 0
    for image, _ in torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False):
        if idx in correct:
            imgs.append(image.flatten())
            if len(imgs) == n_images:
                break
        idx += 1
    return imgs


def one_forward(net, img_f, eps):
    """Run one seeded precise forward pass and return the output zonotope."""
    # Only np.random matters on the precise path (the perturbation of A). Do
    # not call torch.manual_seed here: under Dynamo it walks the Python stack
    # (traceback.format_stack) and showed up as about 20 % of the profile. That
    # is cost of this harness, not of the analysis being measured.
    np.random.seed(SEED)
    with torch.no_grad():
        return ce.forward_zono_precise(net, ce.make_hazed(img_f, eps))


def run_all(net, imgs, epsilons, reps=1):
    """Run one precise forward for every (image, epsilon) pair, reps times."""
    for _ in range(reps):
        for img_f in imgs:
            for eps in epsilons:
                one_forward(net, img_f, eps)


def main():
    """Parse arguments, time the precise forward, then profile it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--eps-indices", type=int, nargs="+", default=[0, 4, 8, 12],
                    help="indices into the default 16-epsilon range")
    ap.add_argument("--reps", type=int, default=20,
                    help="repeat the whole (images x eps) set this many times; "
                         "one pass is only tens of ms, too short for stable shares")
    ap.add_argument("--flags", choices=list(FLAGS), default="best")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None, help="optional path to save the report")
    args = ap.parse_args()

    flags = set_flags(args.flags)
    test_range = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    epsilons = [test_range[i] for i in args.eps_indices]

    net = ce.build_net()
    imgs = load_images(args.n_images)
    n_fwd = len(imgs) * len(epsilons) * args.reps

    header = (f"precise-only profile | net=3layer | flags={args.flags} {flags} | "
              f"images={len(imgs)} eps_idx={args.eps_indices} -> {n_fwd} forwards | "
              f"torch={torch.__version__} threads={torch.get_num_threads()} "
              f"dtype={torch.get_default_dtype()}")
    print(header)

    # warm-up: first call pays imports/allocator/BLAS thread-pool start-up
    one_forward(net, imgs[0], epsilons[0])

    t0 = time.perf_counter()
    run_all(net, imgs, epsilons, args.reps)
    wall = time.perf_counter() - t0
    print(f"\nWALL-CLOCK (no profiler): {wall:.3f} s total, "
          f"{wall / n_fwd * 1e3:.1f} ms per forward ({n_fwd} forwards)")

    prof = cProfile.Profile()
    prof.enable()
    run_all(net, imgs, epsilons, args.reps)
    prof.disable()

    buf = io.StringIO()
    st = pstats.Stats(prof, stream=buf).strip_dirs()
    total = st.total_tt
    buf.write(f"\ncProfile total (inflated by per-call overhead): {total:.3f} s\n")
    buf.write(f"\n=== TOP {args.top} BY SELF TIME (tottime) ===\n")
    st.sort_stats("tottime").print_stats(args.top)
    buf.write(f"\n=== TOP {args.top} BY CUMULATIVE TIME ===\n")
    st.sort_stats("cumulative").print_stats(args.top)
    report = buf.getvalue()
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(header + f"\nWALL-CLOCK: {wall:.3f} s\n" + report)
        print("saved", args.out)


if __name__ == "__main__":
    main()
