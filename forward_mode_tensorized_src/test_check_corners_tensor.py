import torch

from precise_transformer import check_corners, check_corners_tensor


def make_random_case(n, seed):
    """Random boxes and plane coefficients, as a list of tuples (original) and an [n, 3] tensor (vectorized)."""
    g = torch.Generator().manual_seed(seed)

    lx = torch.rand(n, generator=g) * 4 - 2      # in [-2, 2)
    width_x = torch.rand(n, generator=g) * 4 + 1e-3
    ux = lx + width_x

    ly = torch.rand(n, generator=g) * 4 - 2
    width_y = torch.rand(n, generator=g) * 4 + 1e-3
    uy = ly + width_y

    A = torch.rand(n, generator=g) * 2 - 1
    B = torch.rand(n, generator=g) * 2 - 1
    C = torch.rand(n, generator=g) * 2 - 1

    # ABCs in the original list-of-tuples-of-0-dim-tensors format
    ABCs = [(A[i], B[i], C[i]) for i in range(n)]
    # same data as a [n, 3] tensor, for the vectorized function
    ABCs_tensor = torch.stack((A, B, C), dim=1)

    return ABCs, ABCs_tensor, lx, ux, ly, uy


def run_case(n, seed, atol=1e-6, rtol=1e-5):
    """Compare both versions on one random case. The tolerances are loose because the vectorized one reorders operations."""
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed)

    old_result = torch.stack(check_corners(ABCs, lx, ux, ly, uy))
    new_result = check_corners_tensor(ABCs_tensor, lx, ux, ly, uy)

    ok = torch.allclose(old_result, new_result, atol=atol, rtol=rtol)
    max_abs_diff = (old_result - new_result).abs().max().item()

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] n={n:5d} seed={seed}  max_abs_diff={max_abs_diff:.3e}")

    if not ok:
        print("  old:", old_result)
        print("  new:", new_result)

    return ok


def main():
    torch.manual_seed(0)

    # Sizes cover the smallest batch, odd sizes, and a larger layer.
    sizes = [1, 2, 5, 17, 100, 1000]
    all_ok = True
    for i, n in enumerate(sizes):
        all_ok &= run_case(n, seed=i)

    # Degenerate box: lx == ux and ly == uy (zero-width intervals), so all four
    # corners coincide.
    n = 8
    ABCs, ABCs_tensor, lx, ux, ly, uy = make_random_case(n, seed=123)
    ux = lx.clone()
    uy = ly.clone()
    old_result = torch.stack(check_corners(ABCs, lx, ux, ly, uy))
    new_result = check_corners_tensor(ABCs_tensor, lx, ux, ly, uy)
    ok = torch.allclose(old_result, new_result, atol=1e-6, rtol=1e-5)
    print(f"[{'PASS' if ok else 'FAIL'}] degenerate box (lx==ux, ly==uy)")
    all_ok &= ok

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
