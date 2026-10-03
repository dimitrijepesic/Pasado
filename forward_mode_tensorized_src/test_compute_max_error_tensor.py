"""
Unit test for compute_max_error: the vectorized path (ABCs_tensor given)
against the original path (ABCs list only).

When ABCs_tensor is passed, compute_max_error uses both vectorized checks
(check_corners_tensor and check_nonlinear_boundary_tensor), so this test
compares the fully combined result of the two paths. The cases cover random
sizes, zeros in ly or uy, zero-width boxes, and A passing through zero.

Run with:  python test_compute_max_error_tensor.py   (from forward_mode_tensorized_src/)
Exit status is 0 when every case passes and 1 otherwise.
"""
import torch

from precise_transformer import compute_max_error


def make_case(n, lx, ux, ly, uy, A, B, C):
    """Package the inputs in the two coefficient formats compute_max_error takes.

    Returns the list of (A, B, C) tuples (original path), the [n, 3] tensor
    (vectorized path), and the box bounds unchanged.
    """
    ABCs = [(A[i], B[i], C[i]) for i in range(n)]
    ABCs_tensor = torch.stack((A, B, C), dim=1)
    return ABCs, ABCs_tensor, lx, ux, ly, uy


def make_random_case(n, seed, ly_lo=-2.0, ly_width=4.0):
    """Build random boxes with ly in [ly_lo, ly_lo + ly_width) and random A, B, C.

    The default range for ly includes negative values and values near zero, so
    the A / ly and A / uy divisions are exercised with a realistic mix of
    denominators.
    """
    g = torch.Generator().manual_seed(seed)

    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 4 + 1e-3

    ly = torch.rand(n, generator=g) * ly_width + ly_lo
    uy = ly + torch.rand(n, generator=g) * 4 + 1e-3

    A = torch.rand(n, generator=g) * 4 - 2
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1

    return make_case(n, lx, ux, ly, uy, A, B, C)


def safe_abs_diff(a, b):
    """Absolute difference that treats matching infinities as zero difference.

    Plain subtraction of two equal infinities gives NaN, which would make an
    agreeing result look like a mismatch in the printed diagnostic.
    """
    both_inf = torch.isinf(a) & torch.isinf(b) & (torch.sign(a) == torch.sign(b))
    diff = (a - b).abs()
    return torch.where(both_inf, torch.zeros_like(diff), diff)


def _as_tensor(result):
    """Normalize the return value of compute_max_error to an [n] tensor.

    The original path returns a list of 0-dim tensors, while the fully
    vectorized path returns an [n] tensor directly. Converting both lets the
    comparison ignore that difference.
    """
    return result if torch.is_tensor(result) else torch.stack(result)


def compare(name, ABCs, ABCs_tensor, lx, ux, ly, uy, atol=1e-4, rtol=1e-3):
    """Run both paths on one case, print the result, return pass/fail."""
    old_result = _as_tensor(compute_max_error(lx, ux, ly, uy, ABCs))
    new_result = _as_tensor(compute_max_error(lx, ux, ly, uy, ABCs, ABCs_tensor))

    ok = torch.allclose(old_result, new_result, atol=atol, rtol=rtol, equal_nan=True)
    max_abs_diff = safe_abs_diff(old_result, new_result).max().item()

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:45s} max_abs_diff={max_abs_diff:.3e}")

    if not ok:
        print("  old:", old_result)
        print("  new:", new_result)

    return ok


def main():
    """Run all test cases and return the process exit status."""
    torch.manual_seed(0)
    all_ok = True

    # 1. Plain random cases of various sizes. ly and uy may be negative, zero
    #    or positive, which exercises the A / ly and A / uy divisions of the
    #    nonlinear-boundary check with a realistic mix of denominators.
    for i, n in enumerate([1, 2, 5, 17, 100, 500]):
        ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=i)
        all_ok &= compare(f"random n={n}", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 2. ly / uy containing an exact zero
    n = 20
    g = torch.Generator().manual_seed(100)
    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 4 + 1e-3
    ly = torch.linspace(-2.0, 2.0, n)  # includes exact 0
    uy = ly + torch.rand(n, generator=g) * 3 + 1e-3
    A = torch.rand(n, generator=g) * 4 - 2
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_case(n, lx, ux, ly, uy, A, B, C)
    all_ok &= compare("ly/uy containing zero", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 3. degenerate intervals: lx == ux
    n = 20
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=300)
    ux = lx.clone()
    all_ok &= compare("degenerate lx==ux", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 4. degenerate intervals: lx == ux and ly == uy together
    n = 20
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=400)
    ux = lx.clone()
    uy = ly.clone()
    all_ok &= compare("degenerate lx==ux and ly==uy", ABCs, ABCs_tensor, lx, ux, ly, uy)

    # 5. A sweeping through 0, mixed sign
    n = 50
    g = torch.Generator().manual_seed(500)
    lx = torch.rand(n, generator=g) * 4 - 2
    ux = lx + torch.rand(n, generator=g) * 4 + 1e-3
    ly = torch.rand(n, generator=g) * 3 + 0.2
    uy = ly + torch.rand(n, generator=g) * 3 + 1e-3
    A = torch.linspace(-2.0, 2.0, n)
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_case(n, lx, ux, ly, uy, A, B, C)
    all_ok &= compare("A sweeping through 0, mixed sign", ABCs, ABCs_tensor, lx, ux, ly, uy)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
