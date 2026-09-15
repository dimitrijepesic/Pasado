"""
Unit-level correctness check for compute_max_error's optional ABCs_tensor
fast path against the default ABCs path. Now that check_nonlinear_boundary_tensor
is wired in alongside check_corners_tensor, ABCs_tensor=... exercises BOTH
vectorized sub-checks at once, so this compares the fully combined result.

Does not touch get_lipschitz.py / lipschitz.sh.

Run with:  python test_compute_max_error_tensor.py
"""
import torch

from precise_transformer import compute_max_error


def make_case(n, lx, ux, ly, uy, A, B, C):
    ABCs = [(A[i], B[i], C[i]) for i in range(n)]
    ABCs_tensor = torch.stack((A, B, C), dim=1)
    return ABCs, ABCs_tensor, lx, ux, ly, uy


def make_random_case(n, seed, ly_lo=-2.0, ly_width=4.0):
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
    both_inf = torch.isinf(a) & torch.isinf(b) & (torch.sign(a) == torch.sign(b))
    diff = (a - b).abs()
    return torch.where(both_inf, torch.zeros_like(diff), diff)


def _as_tensor(result):
    # compute_max_error returns a list of 0-dim tensors on the original path
    # and, since the tensor-max cleanup, a [n] tensor on the fully vectorized
    # one. Normalize both so the comparison is shape-agnostic.
    return result if torch.is_tensor(result) else torch.stack(result)


def compare(name, ABCs, ABCs_tensor, lx, ux, ly, uy, atol=1e-4, rtol=1e-3):
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
    torch.manual_seed(0)
    all_ok = True

    # 1. plain random cases of various sizes (ly/uy allowed to be negative,
    #    zero, or positive - this exercises the nonlinear-boundary path's
    #    A/ly, A/uy division for a realistic mix of denominators)
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
