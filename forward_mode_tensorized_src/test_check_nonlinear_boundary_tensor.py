"""
Unit test for check_nonlinear_boundary_tensor (vectorized) against
check_nonlinear_boundary (the original per-neuron loop).

Both functions find the largest error of the planar approximation along the
nonlinear edges of each neuron's input box. That needs the roots of a cubic
equation, filtered to the box, so the tests include the awkward inputs: a zero
in ly or uy (division by zero), roots outside the box, zero-width boxes, and
the coefficient A passing through zero. The functions are called directly; the
full analysis is not run.

Run with:  python test_check_nonlinear_boundary_tensor.py   (from forward_mode_tensorized_src/)
Exit status is 0 when every case passes and 1 otherwise.
"""
import torch

from precise_transformer import check_nonlinear_boundary, check_nonlinear_boundary_tensor


def make_case(n, seed, lx, ux, ly, uy, A, B, C):
    """Package the inputs in the two coefficient formats the functions expect.

    Returns a list of (A, B, C) tuples for the original function and an [n, 3]
    tensor for the vectorized one, followed by the box bounds unchanged. The
    seed argument is unused and kept only so call sites read uniformly.
    """
    ABCs = [(A[i], B[i], C[i]) for i in range(n)]
    ABCs_tensor = torch.stack((A, B, C), dim=1)
    return ABCs, ABCs_tensor, lx, ux, ly, uy


def make_random_case(n, seed, ly_lo=0.5, ly_width=4.0):
    """Build random boxes with ly in [ly_lo, ly_lo + ly_width) and random A, B, C.

    ly is kept away from zero by default so that A / ly is well defined; the
    division-by-zero case is tested separately below.
    """
    g = torch.Generator().manual_seed(seed)

    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 4 + 1e-3

    ly = torch.rand(n, generator=g) * ly_width + ly_lo
    uy = ly + torch.rand(n, generator=g) * 4 + 1e-3

    # mixed positive/negative A, B, C
    A = torch.rand(n, generator=g) * 4 - 2
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1

    return make_case(n, seed, lx, ux, ly, uy, A, B, C)


def safe_abs_diff(a, b):
    """Absolute difference that treats matching infinities as zero difference.

    Both implementations return -inf for a neuron when every candidate root is
    outside the allowed domain. Plain subtraction would give NaN there (-inf
    minus -inf), even though the two results agree, so infinities with the same
    sign are mapped to a difference of 0. This only affects the printed
    diagnostic; the pass/fail decision uses allclose with equal_nan=True.
    """
    both_inf = torch.isinf(a) & torch.isinf(b) & (torch.sign(a) == torch.sign(b))
    diff = (a - b).abs()
    return torch.where(both_inf, torch.zeros_like(diff), diff)


def compare(name, ABCs, ABCs_tensor, lx, ux, ly, uy, atol=1e-4, rtol=1e-3):
    """Run both implementations on one case, print the result, return pass/fail.

    The tolerances are looser than for the corner check because the root
    computation goes through single-precision complex arithmetic.
    """
    old_maxs, old_maxs_uy = check_nonlinear_boundary(lx, ux, ly, uy, ABCs)
    new_maxs, new_maxs_uy = check_nonlinear_boundary_tensor(lx, ux, ly, uy, ABCs_tensor)

    old_maxs = torch.stack(old_maxs)
    old_maxs_uy = torch.stack(old_maxs_uy)

    ok_ly = torch.allclose(old_maxs, new_maxs, atol=atol, rtol=rtol, equal_nan=True)
    ok_uy = torch.allclose(old_maxs_uy, new_maxs_uy, atol=atol, rtol=rtol, equal_nan=True)
    ok = ok_ly and ok_uy

    diff_ly = safe_abs_diff(old_maxs, new_maxs).max().item()
    diff_uy = safe_abs_diff(old_maxs_uy, new_maxs_uy).max().item()

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:45s} max_diff_ly={diff_ly:.3e} max_diff_uy={diff_uy:.3e}")

    if not ok:
        print("  old_maxs   :", old_maxs)
        print("  new_maxs   :", new_maxs)
        print("  old_maxs_uy:", old_maxs_uy)
        print("  new_maxs_uy:", new_maxs_uy)

    return ok


def main():
    """Run all test cases and return the process exit status."""
    torch.manual_seed(0)
    all_ok = True

    # 1. plain random cases of various sizes
    for i, n in enumerate([1, 2, 5, 17, 100, 500]):
        ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=i)
        all_ok &= compare(f"random n={n}", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 2. ly / uy containing zero (and straddling zero) -> A/ly, A/uy hit inf/nan
    n = 20
    g = torch.Generator().manual_seed(100)
    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 4 + 1e-3
    ly = torch.linspace(-2.0, 2.0, n)  # includes exact 0
    uy = ly + torch.rand(n, generator=g) * 3 + 1e-3
    A = torch.rand(n, generator=g) * 4 - 2
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_case(n, 100, lx, ux, ly, uy, A, B, C)
    all_ok &= compare("ly/uy containing zero", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 3. roots forced outside [lx, ux]: tiny x-window far from where the
    #    sigmoid'' extremum for this A/ly would land
    n = 20
    g = torch.Generator().manual_seed(200)
    lx = torch.full((n,), 5.0) + torch.rand(n, generator=g) * 0.01
    ux = lx + 0.02  # very tight window, unlikely to contain any root
    ly = torch.rand(n, generator=g) * 2 + 0.5
    uy = ly + torch.rand(n, generator=g) * 2 + 1e-3
    A = torch.rand(n, generator=g) * 4 - 2
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_case(n, 200, lx, ux, ly, uy, A, B, C)
    all_ok &= compare("roots outside [lx,ux]", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 4. degenerate intervals: lx == ux (zero-width x-interval)
    n = 20
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=300)
    ux = lx.clone()
    all_ok &= compare("degenerate lx==ux", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 5. degenerate intervals: ly == uy too, combined with lx == ux
    n = 20
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=400)
    ux = lx.clone()
    uy = ly.clone()
    all_ok &= compare("degenerate lx==ux and ly==uy", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 6. random ABCs with both positive and negative A, including A close to 0
    n = 50
    g = torch.Generator().manual_seed(500)
    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 4 + 1e-3
    ly = torch.rand(n, generator=g) * 3 + 0.2
    uy = ly + torch.rand(n, generator=g) * 3 + 1e-3
    A = torch.linspace(-2.0, 2.0, n)  # sweeps through 0
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_case(n, 500, lx, ux, ly, uy, A, B, C)
    all_ok &= compare("A sweeping through 0, mixed sign", ABCs, ABCs_tensor, lx, ux, ly, uy)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
