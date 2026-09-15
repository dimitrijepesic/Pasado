"""Verifies the batched-grid and tensor-max-error optimizations.

Two checks, both against the original per-neuron path:

1. `get_linspace_batched` equals `torch.stack(get_linspace(...))` bit for bit,
   including the degenerate boxes the analysis actually produces.
2. `sigmoid_prime_product_tensor` produces the identical zonotope under every
   combination of the four selectors. This matters because the selectors are
   independent flags: the grid can be batched while the regression is not, the
   error checks can be half-vectorized ("corners only"), and each combination
   takes a different route through the code.

Run:  python test_batched_grid.py   (from forward_mode_tensorized_src/)
"""
import itertools

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import precise_transformer as pt
from SimpleZono import IntervalsToZonotope


def check_grid_equivalence():
    print("=== get_linspace_batched vs get_linspace ===")
    g = torch.Generator().manual_seed(4242)
    all_ok = True

    def one(name, lx, ux, ly, uy, steps=5):
        nonlocal all_ok
        ref = torch.stack(pt.get_linspace(lx, ux, ly, uy, steps))
        got = pt.get_linspace_batched(lx, ux, ly, uy, steps)
        ok = torch.equal(ref, got)
        all_ok &= ok
        detail = "" if ok else f"  max|diff|={(ref - got).abs().max().item():.3e}"
        print(f"  {name:42s} STEPS={steps}  bit-exact={ok}{detail}")

    for steps in (3, 5, 7, 9):
        n = 400
        lx = torch.rand(n, generator=g) * 10 - 5
        ux = lx + torch.rand(n, generator=g) * 5 + 1e-9
        ly = torch.rand(n, generator=g) * 10 - 5
        uy = ly + torch.rand(n, generator=g) * 5 + 1e-9
        one("random boxes", lx, ux, ly, uy, steps)

    n = 200
    z = torch.rand(n, generator=g) * 4 - 2
    # Degenerate boxes are not hypothetical here: they are what makes the
    # regression rank-deficient, which is why the CUDA gels driver is unusable.
    one("point box (lx==ux, ly==uy)", z, z.clone(), z, z.clone())
    one("lx==ux only", z, z.clone(), z, z + 1.0)
    one("width 1e-12", z, z + 1e-12, z, z + 1e-12)
    one("magnitude 1e6", torch.full((n,), 1e6), torch.full((n,), 1e6) + 10,
        torch.full((n,), -1e6), torch.full((n,), -1e6) + 10)
    print(f"  -> all bit-exact: {all_ok}\n")
    return all_ok


def run_variant(x_lb, x_ub, y_lb, y_ub, *, vec, boundary, lstsq, grid, seed=99):
    """One sigmoid_prime_product_tensor call under a fixed selector combination.

    The selectors are read at call time, and the np.random stream is reseeded
    per run because the transformer perturbs coefficient A with an unseeded
    draw -- without pinning it, two runs of the *same* variant would differ and
    nothing could be compared.
    """
    pt.USE_VECTORIZED_PRECISE = vec
    pt.USE_VECTORIZED_BOUNDARY = boundary
    pt.USE_BATCHED_LSTSQ = lstsq
    pt.USE_BATCHED_GRID = grid
    np.random.seed(seed)
    x = IntervalsToZonotope(x_lb.clone(), x_ub.clone())
    y = IntervalsToZonotope(y_lb.clone(), y_ub.clone())
    out = pt.sigmoid_prime_product_tensor(x, y)
    return out.centers.clone(), out.generators.clone()


def check_selector_equivalence():
    print("=== sigmoid_prime_product_tensor across selector combinations ===")
    g = torch.Generator().manual_seed(7)
    n = 60
    x_lb = torch.rand(n, generator=g) * 6 - 3
    x_ub = x_lb + torch.rand(n, generator=g) * 2 + 1e-3
    y_lb = torch.rand(n, generator=g) * 4 - 2
    y_ub = y_lb + torch.rand(n, generator=g) * 2 + 1e-3

    saved = (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY,
             pt.USE_BATCHED_LSTSQ, pt.USE_BATCHED_GRID)
    try:
        # Baseline: everything original.
        ref_c, ref_gen = run_variant(x_lb, x_ub, y_lb, y_ub,
                                     vec=False, boundary=False,
                                     lstsq=False, grid=False)
        print(f"  baseline (all original): centers {tuple(ref_c.shape)}, "
              f"generators {tuple(ref_gen.shape)}")

        worst = 0.0
        all_ok = True
        for vec, boundary, lstsq, grid in itertools.product([False, True], repeat=4):
            if not vec and boundary:
                continue  # boundary flag has no effect without ABCs_tensor
            c, gen = run_variant(x_lb, x_ub, y_lb, y_ub, vec=vec,
                                 boundary=boundary, lstsq=lstsq, grid=grid)
            if c.shape != ref_c.shape or gen.shape != ref_gen.shape:
                print(f"  vec={vec:d} bnd={boundary:d} lstsq={lstsq:d} grid={grid:d}"
                      f"  SHAPE MISMATCH {tuple(c.shape)} {tuple(gen.shape)}")
                all_ok = False
                continue
            dc = (c - ref_c).abs().max().item()
            dg = (gen - ref_gen).abs().max().item()
            worst = max(worst, dc, dg)
            # The vectorized error checks reorder floating-point reductions, so
            # they are allowed to differ in the last bits; the grid and the
            # regression must not differ at all.
            tol = 1e-12 if vec else 0.0
            ok = dc <= tol and dg <= tol
            all_ok &= ok
            print(f"  vec={vec:d} bnd={boundary:d} lstsq={lstsq:d} grid={grid:d}"
                  f"  max|dcenters|={dc:.3e}  max|dgen|={dg:.3e}  ok={ok}")
        print(f"  -> worst deviation across all combinations: {worst:.3e}")
        print(f"  -> all within tolerance: {all_ok}\n")
        return all_ok
    finally:
        (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY,
         pt.USE_BATCHED_LSTSQ, pt.USE_BATCHED_GRID) = saved


if __name__ == "__main__":
    ok = check_grid_equivalence()
    ok &= check_selector_equivalence()
    print("PASS" if ok else "FAIL")
