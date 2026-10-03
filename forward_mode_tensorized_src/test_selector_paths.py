"""
Routing and equivalence test for the PASADO_VECTORIZED_PRECISE and
PASADO_VEC_BOUNDARY selectors.

Runs the full sigmoid_prime_product_tensor entry point and confirms that:

  1. With the selector off, the original check_corners and
     check_nonlinear_boundary run (verified by call counters) and the
     vectorized versions are never called.
  2. With the selector on, check_corners_tensor and
     check_nonlinear_boundary_tensor run and the originals are never called.
  3. In the intermediate "corners only" setting (USE_VECTORIZED_BOUNDARY off),
     the corner check uses the tensor path and the boundary check uses the
     original path.
  4. With identical np.random seeds, all three settings produce numerically
     equivalent output zonotopes. The regressions are identical by
     construction, so only floating-point reordering inside the two checks
     may cause small differences.

The benchmark runs in float64 (get_lipschitz.py sets the default dtype), so
this test does too.

Run with:  python test_selector_paths.py   (from forward_mode_tensorized_src/)
Exit status is non-zero if any assertion fails.
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

import precise_transformer as pt
from SimpleZono import IntervalsToZonotope

# Tolerances for comparing the old and new zonotopes. They only need to absorb
# floating-point reordering in the error checks, which is far below 1e-9.
RTOL = 1e-9
ATOL = 1e-9
SEED = 12345


class CallCounter:
    """Wraps a function and counts how many times it is called."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


def make_inputs(n, seed, degenerate=False):
    """Build two random input zonotopes (x and y) with n neurons each.

    With degenerate=True the boxes have zero width (lx == ux and ly == uy),
    which is the case that makes the regression rank-deficient.
    """
    g = torch.Generator().manual_seed(seed)
    lx = torch.rand(n, generator=g, dtype=torch.float64) * 4 - 2
    ux = lx if degenerate else lx + torch.rand(n, generator=g, dtype=torch.float64) * 4 + 1e-3
    ly = torch.rand(n, generator=g, dtype=torch.float64) * 4 - 2
    uy = ly if degenerate else ly + torch.rand(n, generator=g, dtype=torch.float64) * 4 + 1e-3
    return IntervalsToZonotope(lx, ux), IntervalsToZonotope(ly, uy)


def run_with_flags(x, y, vectorized, vec_boundary=True):
    """Run sigmoid_prime_product_tensor under the given selector setting with
    call counters installed, restoring module state afterwards."""
    saved = (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY,
             pt.check_corners, pt.check_corners_tensor,
             pt.check_nonlinear_boundary, pt.check_nonlinear_boundary_tensor)
    counters = {name: CallCounter(getattr(pt, name)) for name in
                ("check_corners", "check_corners_tensor",
                 "check_nonlinear_boundary", "check_nonlinear_boundary_tensor")}
    try:
        pt.USE_VECTORIZED_PRECISE = vectorized
        pt.USE_VECTORIZED_BOUNDARY = vec_boundary
        for name, c in counters.items():
            setattr(pt, name, c)
        # Only np.random affects this code path (the perturbation of A). The
        # torch seed is set as well so the test stays deterministic if torch
        # randomness is ever introduced.
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        out = pt.sigmoid_prime_product_tensor(x.clone(), y.clone())
    finally:
        (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY,
         pt.check_corners, pt.check_corners_tensor,
         pt.check_nonlinear_boundary, pt.check_nonlinear_boundary_tensor) = saved
    return out, {name: c.calls for name, c in counters.items()}


def check_routing(counts, expect_active, expect_inactive, label):
    """Assert that the listed functions were called and the others were not."""
    for name in expect_active:
        assert counts[name] > 0, f"{label}: {name} was not called ({counts})"
    for name in expect_inactive:
        assert counts[name] == 0, f"{label}: {name} was unexpectedly called ({counts})"


def compare_outputs(z_old, z_new, label):
    """Assert that two zonotopes have the same shape, dtype and values."""
    dc = (z_old.centers - z_new.centers).abs().max().item()
    dg = (z_old.generators - z_new.generators).abs().max().item()
    assert z_old.centers.shape == z_new.centers.shape, label
    assert z_old.generators.shape == z_new.generators.shape, label
    assert z_old.centers.dtype == z_new.centers.dtype == torch.float64, label
    torch.testing.assert_close(z_old.centers, z_new.centers, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(z_old.generators, z_new.generators, rtol=RTOL, atol=ATOL)
    print(f"  {label}: max|dc|={dc:.3e}  max|dg|={dg:.3e}  OK")


def main():
    """Run all three selector settings for several sizes, with and without
    degenerate boxes, checking routing and output equivalence each time."""
    print(f"default USE_VECTORIZED_PRECISE={pt.USE_VECTORIZED_PRECISE} "
          f"(env PASADO_VECTORIZED_PRECISE, '1' if unset)")

    for n in (1, 5, 100):
        for degenerate in (False, True):
            label = f"n={n}{' degenerate' if degenerate else ''}"
            x, y = make_inputs(n, seed=SEED + n + (1000 if degenerate else 0),
                               degenerate=degenerate)

            z_old, c_old = run_with_flags(x, y, vectorized=False)
            check_routing(c_old,
                          ["check_corners", "check_nonlinear_boundary"],
                          ["check_corners_tensor", "check_nonlinear_boundary_tensor"],
                          label + " [old]")

            z_new, c_new = run_with_flags(x, y, vectorized=True)
            check_routing(c_new,
                          ["check_corners_tensor", "check_nonlinear_boundary_tensor"],
                          ["check_corners", "check_nonlinear_boundary"],
                          label + " [new]")

            z_mid, c_mid = run_with_flags(x, y, vectorized=True, vec_boundary=False)
            check_routing(c_mid,
                          ["check_corners_tensor", "check_nonlinear_boundary"],
                          ["check_corners", "check_nonlinear_boundary_tensor"],
                          label + " [corners-only]")

            compare_outputs(z_old, z_new, label + " old-vs-new")
            compare_outputs(z_old, z_mid, label + " old-vs-corners-only")

    print("ALL SELECTOR PATH TESTS PASSED")


if __name__ == "__main__":
    main()
