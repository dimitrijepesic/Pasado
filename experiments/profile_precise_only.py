"""Times and profiles only the precise forward pass (3layer, float64), so the
percentages are about the precise path and not the whole get_lipschitz.py run.
cProfile overstates code with many cheap calls: use it for shares, and the
wall-clock line for time.

Run: python experiments/profile_precise_only.py"""
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

# Selector sets, always with all five flags. "best" is the fastest bit-identical
# CPU setup; real_cubic stays off because it changes the numbers.
FLAGS = {
    "best":     dict(vec=True,  bnd=True,  lstsq=True,  grid=True,  cubic=False),
    "original": dict(vec=False, bnd=False, lstsq=False, grid=False, cubic=False),
}

SEED = 12345


def set_flags(name):
    """Apply the named selector set and return it."""
    f = FLAGS[name]
    pt.USE_VECTORIZED_PRECISE = f["vec"]
    pt.USE_VECTORIZED_BOUNDARY = f["bnd"]
    pt.USE_BATCHED_LSTSQ = f["lstsq"]
    pt.USE_BATCHED_GRID = f["grid"]
    pt.USE_REAL_CUBIC = f["cubic"]
    return f


def load_images(n_images):
    """The first n correctly classified 3layer test images, flattened."""
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
    # Only np.random matters here (the perturbation of A). No torch.manual_seed:
    # under Dynamo it walks the stack and took about 20 % of the profile.
    np.random.seed(SEED)
    with torch.no_grad():
        return ce.forward_zono_precise(net, ce.make_hazed(img_f, eps))


def run_all(net, imgs, epsilons, reps=1):
    for _ in range(reps):
        for img_f in imgs:
            for eps in epsilons:
                one_forward(net, img_f, eps)


def main():
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
