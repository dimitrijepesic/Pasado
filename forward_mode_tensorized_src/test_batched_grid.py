import itertools
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import precise_transformer as pt
from SimpleZono import IntervalsToZonotope


def check_grid_equivalence():
    """Batched grid vs the per-neuron loop. STEPS 3 and 5 and the zero-width boxes must match bit for bit."""
    print("=== get_linspace_batched vs get_linspace ===")
    g = torch.Generator().manual_seed(4242)
    all_ok = True

    def one(name, lx, ux, ly, uy, steps=5, strict=True, ulp_tol=4):
        """strict=True: grids must be identical. Otherwise a few ULP of difference is allowed."""
        nonlocal all_ok
        ref = torch.stack(pt.get_linspace(lx, ux, ly, uy, steps))
        got = pt.get_linspace_batched(lx, ux, ly, uy, steps)
        exact = torch.equal(ref, got)
        max_diff = (ref - got).abs().max().item()
        if strict:
            ok = exact
        else:
            bound = ulp_tol * torch.finfo(ref.dtype).eps * ref.abs().max().item()
            ok = exact or max_diff <= bound
        all_ok &= ok
        if exact:
            tag = "bit-exact"
        elif strict:
            tag = f"NOT bit-exact  max|diff|={max_diff:.3e}  FAIL"
        else:
            tag = (f"known non-bit-exact on arm64  max|diff|={max_diff:.3e}  "
                   f"({'within' if ok else 'EXCEEDS'} {ulp_tol} ULP)")
        print(f"  {name:42s} STEPS={steps}  {'strict' if strict else 'info  '}  {tag}")

    # STEPS=5 is what production uses, so it decides the exit code together with the
    # zero-width boxes below. STEPS=3 is also exact, so it stays strict.
    # STEPS=7/9 differ by 1 ULP on arm64, probably because torch.linspace fuses
    # start + step*i into one FMA while the batched version rounds twice (not checked
    # on x86). They are reported and bounded; _linspace_rows is left as it is.
    for steps in (3, 5, 7, 9):
        n = 400
        lx = torch.rand(n, generator=g) * 10 - 5
        ux = lx + torch.rand(n, generator=g) * 5 + 1e-9
        ly = torch.rand(n, generator=g) * 10 - 5
        uy = ly + torch.rand(n, generator=g) * 5 + 1e-9
        one("random boxes", lx, ux, ly, uy, steps, strict=steps in (3, 5))

    n = 200
    z = torch.rand(n, generator=g) * 4 - 2
    # Degenerate boxes are not hypothetical here: they are what makes the
    # regression rank-deficient, which is why the CUDA gels driver is unusable.
    one("point box (lx==ux, ly==uy)", z, z.clone(), z, z.clone())
    one("lx==ux only", z, z.clone(), z, z + 1.0)
    one("width 1e-12", z, z + 1e-12, z, z + 1e-12)
    one("magnitude 1e6", torch.full((n,), 1e6), torch.full((n,), 1e6) + 10,
        torch.full((n,), -1e6), torch.full((n,), -1e6) + 10)
    print(f"  -> strict cases bit-exact, info cases within bound: {all_ok}\n")
    return all_ok


def run_variant(x_lb, x_ub, y_lb, y_ub, *, vec, boundary, lstsq, grid, seed=99):
    """Run sigmoid_prime_product_tensor once with the given selectors.

    np.random is reseeded because the transformer adds a random perturbation to A.
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
    """Every selector combination must give the same zonotope as the original code.

    The vectorized checks may differ in the last bits (reordered sums). The grid and
    the regression must match exactly.
    """
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
            # The vectorized checks reorder sums, so they may differ in the last bits.
            # The grid and the regression must match exactly.
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
    sys.exit(0 if ok else 1)
