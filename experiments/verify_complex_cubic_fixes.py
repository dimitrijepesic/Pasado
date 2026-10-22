"""Demonstrates and tests the two fixes for the complex Cardano cubic solver:

1. Heuristic threshold + residual check:
   - If |Im(x)| <= imag_tol (e.g., 1e-6 or 1e-7), round Im(x) to 0.
   - Plug the real part r = Re(x) back into P(r) = 2*r^3 - 3*r^2 + r - y
     and verify that |P(r)| <= res_tol.

2. Analytical floating-point rounding estimate inspired by Mine (2004):
   - Propagate first-order floating-point rounding error estimates through Cardano's
     formula and the cubic polynomial evaluation P(r) = 2*r^3 - 3*r^2 + r - y to
     compute per-element thresholds delta_imag(y) and delta_res(r, y). Note that
     because complex pow/sqrt library ULP bounds are approximated by an effective
     operation count (k_ops = 12), this serves as an analytical estimate rather than
     a formally proved sound interval enclosure.
"""
import math
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))

import precise_transformer as pt


def cubic_residual(r, y):
    """Evaluate |2*r^3 - 3*r^2 + r - y| = |sigma''(x) - y| for s = r."""
    return (2.0 * (r ** 3) - 3.0 * (r ** 2) + r - y).abs()


def inverse_poly_tensor_exact_exp(y, complex_dtype=torch.complex64):
    """Same Cardano formula as inverse_poly_tensor, but with 1.0/3.0 instead of 0.33333."""
    Q = torch.tensor(-1.0 / 12.0, dtype=y.dtype, device=y.device)
    R = (0.25 * y).to(complex_dtype)
    J = torch.tensor(1.0j, dtype=complex_dtype, device=y.device)
    sqrt_arg = ((Q ** 3) + (R ** 2)).to(complex_dtype)

    one_third = 1.0 / 3.0
    S = (R + torch.sqrt(sqrt_arg)) ** one_third
    T = (R - torch.sqrt(sqrt_arg)) ** one_third

    x1 = S + T + 0.5
    sqrt_3_by_2 = math.sqrt(3.0) / 2.0
    x2 = -0.5 * (S + T) + 0.5 + (sqrt_3_by_2 * J) * (S - T)
    x3 = -0.5 * (S + T) + 0.5 - (sqrt_3_by_2 * J) * (S - T)
    return (x1, x2, x3)


# ---------------------------------------------------------------------------
# Method 1: Fixed threshold on |Im(x)| + cubic residual check |P(r)| <= res_tol
# ---------------------------------------------------------------------------
def filter_complex_root_fixed_tol(xk, y, imag_tol=1e-6, res_tol=2e-5):
    """Round imaginary part to 0 if |Im(xk)| <= imag_tol, then verify |P(Re(xk))| <= res_tol."""
    r = xk.real.to(y.dtype)
    imag_ok = xk.imag.abs() <= imag_tol
    res_ok = cubic_residual(r, y) <= res_tol
    valid = imag_ok & res_ok
    return torch.where(valid, r, torch.full_like(r, float("nan")))


# ---------------------------------------------------------------------------
# Method 2: Analytical FP rounding error estimate inspired by Mine (2004)
# ---------------------------------------------------------------------------
def mine_cardano_rounding_bounds(r, y, complex_dtype=torch.complex64, legacy_exp=True):
    """Compute analytical estimates (delta_imag, delta_res) inspired by Mine (2004) FP semantics.

    In Mine (2004), each elementary FP operation op(a, b) introduces relative error
    eps_mach plus subnormal floor eta_min:
        |fl(a op b) - (a op b)| <= eps_mach * |a op b| + eta_min

    For Cardano on u^3 - u/4 - y/2 = 0 in the casus irreducibilis (|y| <= 1/(6*sqrt(3))):
      - |S| = |T| = sqrt(|Q|) = 1 / sqrt(12) ~= 0.288675
      - We approximate the accumulated complex sqrt, add, pow(1/3), and linear combination
        error using an effective count of k_ops = 12 roundings of size eps_c * |S|
        (heuristic estimate; a formal proof would require platform-specific ULP bounds
        for complex csqrt/cpow).
      - If the legacy exponent 0.33333 is used instead of 1/3, there is an additional
        exponent truncation error: |z^0.33333 - z^(1/3)| <= |1/3 - 0.33333| * |z^(1/3)| * |ln(z)|.
    """
    real_equiv = torch.float32 if complex_dtype == torch.complex64 else torch.float64
    eps_c = torch.finfo(real_equiv).eps          # 1.192e-7 (fp32) or 2.220e-16 (fp64)
    eta_c = torch.finfo(real_equiv).tiny         # subnormal/tiny floor

    # Modulus |S| = |T| = 1 / sqrt(12) for real-root regime
    mod_S = 1.0 / math.sqrt(12.0)

    # 12 elementary operations in Cardano (R, Q^3+R^2, csqrt, R+-W, cpow, S+-T, *sqrt(3)/2)
    k_ops = 12.0
    delta_cardano = k_ops * eps_c * mod_S + eta_c

    if legacy_exp:
        # Legacy code uses x ** 0.33333 instead of 1/3 -> |0.33333 - 1/3| = 1e-5 / 3
        exp_err = abs(1.0 / 3.0 - 0.33333)
        # |ln(S)| = sqrt((0.5*ln(1/12))^2 + theta^2) <= sqrt(1.242^2 + pi^2) ~= 3.38
        delta_exp = exp_err * mod_S * 3.5
    else:
        delta_exp = 0.0

    # 1) Bound on |Im(x_k)| for a true real root:
    delta_imag = 2.0 * delta_cardano

    # 2) Bound on root error |r_tilde - r| = delta_r:
    delta_r = 2.0 * (delta_cardano + delta_exp)

    # 3) Propagate [r - delta_r, r + delta_r] through P(r) = 2*r^3 - 3*r^2 + r - y
    #    using interval arithmetic + per-operation rounding (Mine 2004):
    eps_y = torch.finfo(y.dtype).eps
    abs_r = r.abs()
    poly_cond = 6.0 * (abs_r ** 2) + 6.0 * abs_r + 1.0
    eval_round = 6.0 * eps_y * (2.0 * (abs_r ** 3) + 3.0 * (abs_r ** 2) + abs_r + y.abs())
    delta_res = poly_cond * delta_r + eval_round + eta_c

    return delta_imag, delta_res


def filter_complex_root_mine(xk, y, complex_dtype=torch.complex64, legacy_exp=True):
    """Filter complex root using Mine (2004) FP rounding bounds on Im(xk) and P(Re(xk))."""
    r = xk.real.to(y.dtype)
    delta_imag, delta_res = mine_cardano_rounding_bounds(
        r, y, complex_dtype=complex_dtype, legacy_exp=legacy_exp
    )
    imag_ok = xk.imag.abs() <= delta_imag
    res_ok = cubic_residual(r, y) <= delta_res
    valid = imag_ok & res_ok
    return torch.where(valid, r, torch.full_like(r, float("nan")))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    l = -1.0 / (6.0 * math.sqrt(3.0))
    u = -l

    # 1000 valid y values inside [l, u] and 200 invalid y values outside [l, u]
    y_valid = torch.linspace(l * 0.9999, u * 0.9999, 1000, dtype=torch.float64, device=device)
    y_invalid = torch.cat([
        torch.linspace(l * 2.0, l * 1.01, 100, dtype=torch.float64, device=device),
        torch.linspace(u * 1.01, u * 2.0, 100, dtype=torch.float64, device=device),
    ])

    if device.type == "cuda":
        # On actual CUDA, complex64 Cardano naturally produces ~1e-8i rounding noise
        x_valid_legacy = list(pt.inverse_poly_tensor(y_valid))
        x_valid_exact = list(inverse_poly_tensor_exact_exp(y_valid))
    else:
        # On CPU, simulate GPU complex64 noise on valid roots (~1e-8 to 4e-8 imaginary residual)
        torch.manual_seed(7)
        gpu_imag_noise = (torch.randn(1000, dtype=torch.float32) * 2e-8).to(torch.complex64) * 1j
        x_valid_legacy = [xk + gpu_imag_noise for xk in pt.inverse_poly_tensor(y_valid)]
        x_valid_exact = [xk + gpu_imag_noise for xk in inverse_poly_tensor_exact_exp(y_valid)]
    x_invalid_legacy = pt.inverse_poly_tensor(y_invalid)

    # Compare how many valid roots (out of 3000) and invalid complex roots are kept:
    orig_kept = sum(torch.isreal(xk).sum().item() for xk in x_valid_legacy)
    m1_legacy_kept = sum(
        torch.isfinite(filter_complex_root_fixed_tol(xk, y_valid, imag_tol=1e-6, res_tol=2e-5)).sum().item()
        for xk in x_valid_legacy
    )
    m1_exact_kept = sum(
        torch.isfinite(filter_complex_root_fixed_tol(xk, y_valid, imag_tol=1e-6, res_tol=1e-6)).sum().item()
        for xk in x_valid_exact
    )
    m2_legacy_kept = sum(
        torch.isfinite(filter_complex_root_mine(xk, y_valid, legacy_exp=True)).sum().item()
        for xk in x_valid_legacy
    )
    m2_exact_kept = sum(
        torch.isfinite(filter_complex_root_mine(xk, y_valid, legacy_exp=False)).sum().item()
        for xk in x_valid_exact
    )

    # On out-of-domain y (outside [-1/(6*sqrt(3)), +1/(6*sqrt(3))]), no root lies in [0, 1].
    # Because legacy Cardano uses principal complex cube roots for S and T, for y < l
    # all three outputs x1, x2, x3 have non-zero imaginary parts (none is a real root);
    # checking x2 and x3 verifies that neither filter accepts non-real Cardano outputs.
    # (Note: for soundness of the upper bound, only false negatives—dropped valid roots—can
    # cause under-approximation; extra candidate points in [lx, ux] only increase max|E|.)
    m1_false_pos = sum(
        torch.isfinite(filter_complex_root_fixed_tol(xk, y_invalid, imag_tol=1e-6, res_tol=2e-5)).sum().item()
        for xk in x_invalid_legacy[1:]
    )
    m2_false_pos = sum(
        torch.isfinite(filter_complex_root_mine(xk, y_invalid, legacy_exp=True)).sum().item()
        for xk in x_invalid_legacy[1:]
    )

    sample_r = x_valid_legacy[0].real.to(torch.float64)
    d_im_leg, d_res_leg = mine_cardano_rounding_bounds(sample_r, y_valid, legacy_exp=True)
    d_im_ex, d_res_ex = mine_cardano_rounding_bounds(sample_r, y_valid, legacy_exp=False)

    print(f"=== Summary on 3000 valid roots (device={device}) ===")
    print(f"Original torch.isreal(x)            kept: {orig_kept:4d} / 3000")
    print(f"Method 1 (fixed tol, 0.33333)       kept: {m1_legacy_kept:4d} / 3000")
    print(f"Method 1 (fixed tol, 1.0/3.0)       kept: {m1_exact_kept:4d} / 3000")
    print(f"Method 2 (Mine FP bound, 0.33333)   kept: {m2_legacy_kept:4d} / 3000  (delta_imag={d_im_leg:.2e}, max delta_res={d_res_leg.max():.2e})")
    print(f"Method 2 (Mine FP bound, 1.0/3.0)   kept: {m2_exact_kept:4d} / 3000  (delta_imag={d_im_ex:.2e}, max delta_res={d_res_ex.max():.2e})")
    print(f"False positives on 400 true complex roots: Method 1 = {m1_false_pos}, Method 2 = {m2_false_pos}")

    # End-to-end test on 3layer, 4layer, 5layer, big (image 0, eps_idx 0, float64)
    from measure_split_sweep import load_network_and_images, build_forward_precise, eval_splits_once

    torch.set_default_dtype(torch.float64)
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True

    eps0 = 10 ** (-2 / 4) * 2
    orig_inv_sig_2nd = pt.inverse_sigmoid_2nd_deriv

    def make_patched_inv_sig(mode):
        def patched(y):
            if mode == "legacy_isreal":
                pt.USE_REAL_CUBIC = False
                return orig_inv_sig_2nd(y)
            if mode == "viete":
                x1, x2, x3 = pt.inverse_poly_real_tensor(y)
                return (pt.inv_sigmoid(x1), pt.inv_sigmoid(x2), pt.inv_sigmoid(x3))
            if mode == "m1_legacy":
                x1, x2, x3 = pt.inverse_poly_tensor(y)
                r1 = filter_complex_root_fixed_tol(x1, y, imag_tol=1e-6, res_tol=2e-5)
                r2 = filter_complex_root_fixed_tol(x2, y, imag_tol=1e-6, res_tol=2e-5)
                r3 = filter_complex_root_fixed_tol(x3, y, imag_tol=1e-6, res_tol=2e-5)
                return (pt.inv_sigmoid(r1), pt.inv_sigmoid(r2), pt.inv_sigmoid(r3))
            if mode == "m1_exact":
                x1, x2, x3 = inverse_poly_tensor_exact_exp(y)
                r1 = filter_complex_root_fixed_tol(x1, y, imag_tol=1e-6, res_tol=1e-6)
                r2 = filter_complex_root_fixed_tol(x2, y, imag_tol=1e-6, res_tol=1e-6)
                r3 = filter_complex_root_fixed_tol(x3, y, imag_tol=1e-6, res_tol=1e-6)
                return (pt.inv_sigmoid(r1), pt.inv_sigmoid(r2), pt.inv_sigmoid(r3))
            if mode == "m2_mine_legacy":
                x1, x2, x3 = pt.inverse_poly_tensor(y)
                r1 = filter_complex_root_mine(x1, y, legacy_exp=True)
                r2 = filter_complex_root_mine(x2, y, legacy_exp=True)
                r3 = filter_complex_root_mine(x3, y, legacy_exp=True)
                return (pt.inv_sigmoid(r1), pt.inv_sigmoid(r2), pt.inv_sigmoid(r3))
            if mode == "m2_mine_exact":
                x1, x2, x3 = inverse_poly_tensor_exact_exp(y)
                r1 = filter_complex_root_mine(x1, y, legacy_exp=False)
                r2 = filter_complex_root_mine(x2, y, legacy_exp=False)
                r3 = filter_complex_root_mine(x3, y, legacy_exp=False)
                return (pt.inv_sigmoid(r1), pt.inv_sigmoid(r2), pt.inv_sigmoid(r3))
            raise ValueError(mode)
        return patched

    print(f"\n=== End-to-End lc_precise across networks (device={device}, image_id=0, eps=0.632456, float64) ===")
    print(
        f"{'Network':8s} | {'Legacy isreal':>14s} | {'Viete (real)':>14s} | "
        f"{'M1 (0.33333)':>14s} | {'M1 (1/3)':>14s} | {'M2 Mine(0.333)':>14s} | {'M2 Mine(1/3)':>14s}"
    )
    print("-" * 103)
    for net_name in ["3layer", "4layer", "5layer", "big"]:
        net, imgs = load_network_and_images(net_name, device, torch.float64, 1)
        _, img = imgs[0]
        fwd = build_forward_precise(pt, net, net_name)
        vals = {}
        for mode in ["legacy_isreal", "viete", "m1_legacy", "m1_exact", "m2_mine_legacy", "m2_mine_exact"]:
            pt.inverse_sigmoid_2nd_deriv = make_patched_inv_sig(mode)
            vals[mode] = eval_splits_once(fwd, img, eps0, 1, 7, device, torch.float64)
        pt.inverse_sigmoid_2nd_deriv = orig_inv_sig_2nd
        print(
            f"{net_name:8s} | {vals['legacy_isreal']:14.6f} | {vals['viete']:14.6f} | {vals['m1_legacy']:14.6f} | "
            f"{vals['m1_exact']:14.6f} | {vals['m2_mine_legacy']:14.6f} | {vals['m2_mine_exact']:14.6f}"
        )


if __name__ == "__main__":
    main()
