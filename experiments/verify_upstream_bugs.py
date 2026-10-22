"""Demonstrates the two upstream bugs in Pasado's forward-mode zonotope code.

Each bug is toggled by monkeypatching, so this script shows the effect of the
fixes but does not test the repository code; experiments/test_soundness.py does.

1. Zonotope.__neg__ and DualZonotope.__neg__ (SimpleZono.py) negate centers
   (-self.centers) but fail to negate generators (returning +self.generators
   instead of -self.generators).
   - Consequence 1a (hazed input): In Section_5_4/get_lipschitz.py,
       hazed_zono = (one - zono_eps) * img_f + zono_eps
     places generator coefficient +g_t * (1 + img_f) instead of +g_t * (1 - img_f)
     on the real part, for every sub-interval [l, r] including l = 0. The input
     becomes the segment from (1 - r) * img_f + l to (1 - l) * img_f + r, so after
     the first affine layer the real zonotope fails to enclose the true network
     activations along the haze path, and lc_precise can drop BELOW the true
     directional derivative |d logit / dt| (unsound bound).
   - Consequence 1b (SigmoidDualZonotope): In SimpleZono.py,
       sigmoid_deriv = sigmoid_real * (-sigmoid_real + 1)
     computes first-order generator (s_0 + (1 - s_0)) * g = 1 * g instead of
     (1 - 2 * s_0) * g.

2. check_nonlinear_boundary and _max_objective_over_x_candidates
   (precise_transformer.py) maximize E(x, y) = sigma'(x)*y - (A*x + B*y + C)
   instead of |E(x, y)|, ignoring negative approximation errors at interior
   critical points on the y = ly and y = uy edges (unlike check_corners, which
   checks both +E and -E).
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))
sys.path.insert(0, HERE)

import SimpleZono as sz
import precise_transformer as pt
from Duals import DualIntervalTensor
from measure_split_sweep import (
    load_network_and_images,
    build_forward_precise,
    eval_splits_once,
)


def sample_max_directional_derivative(net, img, eps_val, n_samples=1001):
    """Sample max_{t in [0, eps], k} |d net((1-t)*img + t)_k / dt| via autograd."""
    ts = torch.linspace(0.0, eps_val, n_samples, dtype=img.dtype, device=img.device)
    v = 1.0 - img
    max_deriv = 0.0
    for t in ts:
        x = ((1.0 - t) * img + t).detach().requires_grad_(True)
        out = net(x.unsqueeze(0)).squeeze(0)
        for k in range(out.shape[0]):
            g = torch.autograd.grad(out[k], x, retain_graph=True)[0]
            val = abs(torch.dot(g, v).item())
            if val > max_deriv:
                max_deriv = val
    return max_deriv


def set_bug_state(fix_neg: bool, fix_abs: bool):
    """Monkeypatch SimpleZono and precise_transformer to toggle each fix."""
    if fix_neg:
        sz.Zonotope.__neg__ = lambda self: sz.Zonotope(-self.centers, -self.generators)
        sz.DualZonotope.__neg__ = lambda self: sz.DualZonotope(
            -self.real.centers, -self.real.generators,
            -self.dual.centers, -self.dual.generators,
        )
    else:
        sz.Zonotope.__neg__ = lambda self: sz.Zonotope(-self.centers, self.generators)
        sz.DualZonotope.__neg__ = lambda self: sz.DualZonotope(
            -self.real.centers, self.real.generators,
            -self.dual.centers, self.dual.generators,
        )

    def max_obj(A, B, C, xs, ys):
        sig = torch.sigmoid(xs)
        sigmoid_prime_y = (1.0 - sig) * (sig * ys)
        planar = A.unsqueeze(1) * xs + B.unsqueeze(1) * ys + C.unsqueeze(1)
        evaluation = sigmoid_prime_y - planar
        if fix_abs:
            evaluation = evaluation.abs()
        evaluation = torch.nan_to_num(evaluation, -float("Inf"))
        return torch.max(evaluation, dim=1).values

    pt._max_objective_over_x_candidates = max_obj


def main():
    torch.set_default_dtype(torch.float64)
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True
    pt.USE_REAL_CUBIC = True

    print("=== 1. Minimal algebraic check of Bug 1 (Zonotope.__neg__) ===")
    l_val, r_val = 0.2, 0.4
    img_2px = torch.tensor([0.1, 0.8])
    one = torch.tensor([1.0])
    true_lo = (1.0 - l_val) * img_2px + l_val
    true_hi = (1.0 - r_val) * img_2px + r_val
    print(f"True haze endpoints for t in [{l_val}, {r_val}] on img={img_2px.tolist()}:")
    print(f"  at t={l_val} (eps_1 = -1): {true_lo.tolist()}")
    print(f"  at t={r_val} (eps_1 = +1): {true_hi.tolist()}")

    for label, fix_neg in [("Upstream buggy (+generators)", False), ("Fixed (-generators)", True)]:
        set_bug_state(fix_neg=fix_neg, fix_abs=True)
        ie = DualIntervalTensor(real_l=torch.tensor([l_val]), real_u=torch.tensor([r_val]))
        ie.e1_l[0] = 1.0
        ie.e1_u[0] = 1.0
        ze = sz.HyperDualIntervalToDualZonotope(ie)
        hz = (one - ze) * img_2px + ze
        c, g = hz.real.centers, hz.real.generators[0]
        print(f"  {label}:")
        print(f"    at eps_1 = -1: {(c - g).tolist()}")
        print(f"    at eps_1 = +1: {(c + g).tolist()}")

    print("\n=== 2. Soundness check against sampled autograd derivative (4layer, image 0, eps_idx=8) ===")
    eps_schedule = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    eps8 = eps_schedule[8]
    net4, imgs4 = load_network_and_images("4layer", torch.device("cpu"), torch.float64, 1)
    _, img0 = imgs4[0]
    true_max = sample_max_directional_derivative(net4, img0, eps8, n_samples=1001)
    print(f"Sampled autograd lower bound max |d logit / dt| = {true_max:.6f}")

    fwd4 = build_forward_precise(pt, net4, "4layer")
    for label, fn, fa in [
        ("Upstream (both bugs present)", False, False),
        ("Fix 1 only (__neg__ fixed)  ", True, False),
        ("Fix 2 only (abs(E) fixed)   ", False, True),
        ("Both fixes applied          ", True, True),
    ]:
        set_bug_state(fix_neg=fn, fix_abs=fa)
        lc1 = eval_splits_once(fwd4, img0, eps8, 1, 7, torch.device("cpu"), torch.float64)
        lc16 = eval_splits_once(fwd4, img0, eps8, 16, 7, torch.device("cpu"), torch.float64)
        sound_16 = "SOUND (>= true_max)" if lc16 >= true_max else "UNSOUND (< true_max!)"
        print(f"  {label} | n_splits=1: {lc1:10.6f} | n_splits=16: {lc16:10.6f}  [{sound_16}]")

    print("\n=== 3. Impact on big network (image 0, eps_idx=0, eps=0.632456) ===")
    eps0 = eps_schedule[0]
    net_big, imgs_big = load_network_and_images("big", torch.device("cpu"), torch.float64, 1)
    _, img_b0 = imgs_big[0]
    true_max_big = sample_max_directional_derivative(net_big, img_b0, eps0, n_samples=201)
    print(f"Sampled autograd lower bound max |d logit / dt| = {true_max_big:.6f}")
    fwd_big = build_forward_precise(pt, net_big, "big")

    print(f"{'n_splits':>8s} | {'Upstream buggy':>16s} | {'Fix 1 (__neg__)':>16s} | {'Both fixes':>16s}")
    print("-" * 65)
    for s in [1, 2, 4, 8, 16]:
        set_bug_state(False, False)
        v_bug = eval_splits_once(fwd_big, img_b0, eps0, s, 7, torch.device("cpu"), torch.float64)
        set_bug_state(True, False)
        v_f1 = eval_splits_once(fwd_big, img_b0, eps0, s, 7, torch.device("cpu"), torch.float64)
        set_bug_state(True, True)
        v_both = eval_splits_once(fwd_big, img_b0, eps0, s, 7, torch.device("cpu"), torch.float64)
        assert v_both >= true_max_big, f"Unsound bound at n_splits={s}: {v_both} < {true_max_big}"
        print(f"{s:8d} | {v_bug:16.6f} | {v_f1:16.6f} | {v_both:16.6f}")

    # Restore fixed state
    set_bug_state(True, True)
    print("\nALL CHECKS PASSED (both fixes restore soundness >= sampled autograd max).")


if __name__ == "__main__":
    main()
