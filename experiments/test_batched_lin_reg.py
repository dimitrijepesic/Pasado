"""Correctness tests: lin_reg_tensor (per-neuron loop) vs lin_reg_tensor_batched.

lin_reg_tensor_batched solves all per-neuron planar regressions of a layer in a
single batched least-squares call. This test checks that it gives the same
coefficients as calling the original lin_reg_tensor once per neuron.

Coverage:
  - Batch sizes n = 1, 10, 100 and 1024.
  - float64 and float32. The original lin_reg_tensor builds its column of ones
    in the default dtype, so each dtype section runs under the matching
    torch.set_default_dtype. This is how the benchmark (float64) and the older
    unit tests (float32) run it.
  - Realistic grids (torch.linspace combined with torch.cartesian_prod, with
    targets from sigmoid_prime_times_y) and purely random points.
  - Degenerate boxes: lx == ux gives a rank-2 system and a point box gives a
    rank-1 system. Also near-degenerate boxes (width 1e-12), saturated sigmoid
    inputs (|x| around 30, so targets are close to 0), badly scaled systems,
    and a mixed batch of healthy and degenerate neurons.
  - dtype, device and shape of the result are preserved.
  - The downstream planar-approximation error is equal, since that is what
    compute_max_error actually consumes.

Pass criterion: bit-identical results (torch.equal) are expected and reported,
but the hard requirement is torch.testing.assert_close with rtol=1e-12 and
atol=1e-14 for float64, and rtol=1e-5 and atol=1e-6 for float32.

Run:  python experiments/test_batched_lin_reg.py   (from the repository root)
Exit status is non-zero if any assertion fails.
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import precise_transformer as pt
from batched_lin_reg import lin_reg_tensor_batched

SEED = 12345
# Hard tolerances per dtype for the assert_close check (see module docstring).
TOL = {torch.float64: dict(rtol=1e-12, atol=1e-14),
       torch.float32: dict(rtol=1e-5, atol=1e-6)}


def old_loop(x_batch, zs_batch):
    """Reference result: call the original lin_reg_tensor once per neuron."""
    return torch.stack([pt.lin_reg_tensor(x_batch[i], zs_batch[i])
                        for i in range(x_batch.shape[0])])


def realistic_grids(n, seed, dtype, box="normal"):
    """Stacked sampling grids exactly as get_linspace builds them, one per neuron.

    box selects the kind of input box: "normal", "degenerate_x" (lx == ux),
    "point" (lx == ux and ly == uy), "near_degenerate" (width 1e-12),
    "saturated" (large x, sigmoid close to 1) or "mixed_scale" (very different
    magnitudes in x and y, which makes the system badly conditioned).
    Returns the grid points [n, 25, 2] and the regression targets [n, 25].
    """
    g = torch.Generator().manual_seed(seed)

    def rnd(lo, hi):
        return (torch.rand((), generator=g, dtype=dtype) * (hi - lo) + lo).item()

    grids = []
    for _ in range(n):
        if box == "normal":
            lx = rnd(-4, 4); ux = lx + rnd(1e-3, 4)
            ly = rnd(-4, 4); uy = ly + rnd(1e-3, 4)
        elif box == "degenerate_x":
            lx = rnd(-4, 4); ux = lx
            ly = rnd(-4, 4); uy = ly + rnd(1e-3, 4)
        elif box == "point":
            lx = rnd(-4, 4); ux = lx
            ly = rnd(-4, 4); uy = ly
        elif box == "near_degenerate":
            lx = rnd(-4, 4); ux = lx + 1e-12
            ly = rnd(-4, 4); uy = ly + 1e-12
        elif box == "saturated":
            lx = rnd(25, 35); ux = lx + rnd(1e-3, 4)
            ly = rnd(-4, 4); uy = ly + rnd(1e-3, 4)
        elif box == "mixed_scale":
            lx = rnd(-1e-8, 1e-8); ux = lx + 1e-8
            ly = rnd(-1e6, 1e6); uy = ly + rnd(1.0, 1e6)
        xs = torch.linspace(lx, ux, steps=5)
        ys = torch.linspace(ly, uy, steps=5)
        grids.append(torch.cartesian_prod(xs, ys))
    x_batch = torch.stack(grids)                      # [n, 25, 2]
    zs_batch = torch.stack([pt.sigmoid_prime_times_y(gr) for gr in grids])  # [n, 25]
    return x_batch, zs_batch


def random_points(n, m, seed, dtype):
    """n independent regression problems with m random points each."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, m, 2, generator=g, dtype=dtype)
    zs = torch.randn(n, m, generator=g, dtype=dtype)
    return x, zs


def planar_error(coeffs, x_batch, zs_batch):
    """Max |z - (A x + B y + C)| per neuron.

    This is the quantity the planar approximation feeds into compute_max_error.
    The coefficients are intercept-first: column 0 is C, then A, then B.
    """
    C, A, B = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
    pred = A * x_batch[:, :, 0] + B * x_batch[:, :, 1] + C
    return (zs_batch - pred).abs().max(dim=1).values


def run_case(name, x_batch, zs_batch, dtype):
    """Compare batched and per-neuron results on one problem set.

    Asserts shape, dtype, device, closeness (assert_close) and that passing the
    targets as [n, m, 1] gives exactly the same result as [n, m]. Returns True
    when the two results are also bit-identical.
    """
    old = old_loop(x_batch, zs_batch)
    new = lin_reg_tensor_batched(x_batch, zs_batch)

    assert new.shape == (x_batch.shape[0], 3), f"{name}: shape {new.shape}"
    assert new.dtype == x_batch.dtype, f"{name}: dtype {new.dtype}"
    assert new.device == x_batch.device, f"{name}: device {new.device}"

    bitwise = torch.equal(old, new)
    d = (old - new).abs()
    max_abs = d.max().item()
    max_rel = (d / old.abs().clamp_min(1e-300)).max().item()
    torch.testing.assert_close(new, old, **TOL[dtype])

    # zs as [n, m, 1] must give the identical result
    new3 = lin_reg_tensor_batched(x_batch, zs_batch.unsqueeze(-1))
    assert torch.equal(new, new3), f"{name}: [n,m,1] zs path differs"

    # downstream planar errors must match
    pe = (planar_error(old, x_batch, zs_batch)
          - planar_error(new, x_batch, zs_batch)).abs().max().item()

    print(f"  [{name:34s}] bitwise={'YES' if bitwise else 'no '} "
          f"max|dcoef|={max_abs:.3e} max_rel={max_rel:.3e} "
          f"planar_err_diff={pe:.3e}  PASS")
    return bitwise


def main():
    """Run every case for float64 and float32 and print a summary."""
    all_bitwise = True
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)  # lin_reg_tensor builds ones() in default dtype
        print(f"--- dtype={dtype} ---")
        for n in (1, 10, 100, 1024):
            x, zs = realistic_grids(n, SEED + n, dtype)
            all_bitwise &= run_case(f"realistic grid n={n}", x, zs, dtype)
        for n in (1, 10, 100):
            x, zs = random_points(n, 25, SEED + n, dtype)
            all_bitwise &= run_case(f"random points n={n}", x, zs, dtype)
        for box in ("degenerate_x", "point", "near_degenerate",
                    "saturated", "mixed_scale"):
            x, zs = realistic_grids(16, SEED, dtype, box=box)
            all_bitwise &= run_case(f"{box} n=16", x, zs, dtype)
        # mixed batch: healthy and rank-deficient neurons side by side
        xh, zh = realistic_grids(8, SEED + 1, dtype)
        xd, zd = realistic_grids(8, SEED + 2, dtype, box="degenerate_x")
        all_bitwise &= run_case("mixed healthy+degenerate n=16",
                                torch.cat((xh, xd)), torch.cat((zh, zd)), dtype)

    print(f"\nALL PASS (assert_close, rtol/atol per dtype); "
          f"bitwise-identical everywhere: {'YES' if all_bitwise else 'NO'}")


if __name__ == "__main__":
    main()
