"""Soundness checks that run against the repository code as it is (no monkeypatching).

1. Zonotope and DualZonotope negation: Z + (-Z) has zero generators, and the hazed
   input (1 - t) * img + t has the true endpoints at eps = -1 and eps = +1.
2. compute_max_error is at least the largest |sigma'(x) * y - (A x + B y + C)| found
   by brute force on a dense grid over the box, for both the vectorized and the
   original per-neuron path. A sound error term may be larger, never smaller.
3. lc_precise is at least the largest |d logit / dt| sampled with autograd along the
   haze path, on (network, image, epsilon, n_splits) cases that were below it before
   the negation and |E| fixes.

Run: python experiments/test_soundness.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))
sys.path.insert(0, HERE)

import precise_transformer as pt
from Duals import DualIntervalTensor
from SimpleZono import DualZonotope, HyperDualIntervalToDualZonotope, Zonotope

EPS_SCHEDULE = [10 ** (-k / 4) * 2 for k in range(2, 18)]
CPU = torch.device("cpu")


def report(ok, name, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    return ok


def set_flags(vectorized, real_cubic):
    pt.USE_VECTORIZED_PRECISE = vectorized
    pt.USE_VECTORIZED_BOUNDARY = vectorized
    pt.USE_BATCHED_LSTSQ = vectorized
    pt.USE_BATCHED_GRID = vectorized
    pt.USE_REAL_CUBIC = real_cubic


def check_negation():
    all_ok = True
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(5, generator=g)
    generators = torch.randn(3, 5, generator=g)

    z = Zonotope(centers.clone(), generators.clone())
    diff = z + (-z)
    worst = diff.generators.abs().max().item()
    all_ok &= report(worst == 0.0, "Zonotope: Z + (-Z) has zero generators", f"max|gen|={worst:.3e}")

    dz = DualZonotope(centers.clone(), generators.clone(), 2 * centers, -generators)
    ddiff = dz + (-dz)
    worst = max(ddiff.real.generators.abs().max().item(), ddiff.dual.generators.abs().max().item())
    all_ok &= report(worst == 0.0, "DualZonotope: DZ + (-DZ) has zero generators", f"max|gen|={worst:.3e}")

    img = torch.tensor([0.0, 0.1, 0.8, 1.0])
    for l_val, r_val in [(0.0, 0.3), (0.2, 0.4)]:
        ie = DualIntervalTensor(real_l=torch.tensor([l_val]), real_u=torch.tensor([r_val]))
        ie.e1_l[0] = 1
        ie.e1_u[0] = 1
        zono_eps = HyperDualIntervalToDualZonotope(ie)
        hazed = (torch.tensor([1.0]) - zono_eps) * img + zono_eps
        c = hazed.real.centers
        gen = hazed.real.generators.sum(dim=0)
        true_lo = (1 - l_val) * img + l_val
        true_hi = (1 - r_val) * img + r_val
        err = max((c - gen - true_lo).abs().max().item(), (c + gen - true_hi).abs().max().item())
        all_ok &= report(err < 1e-12, f"hazed input endpoints, t in [{l_val}, {r_val}]", f"max|err|={err:.3e}")
    return all_ok


def realistic_boxes(n, seed):
    """Boxes and regression planes built the same way the precise transformer builds them."""
    g = torch.Generator().manual_seed(seed)
    lx = torch.rand(n, generator=g) * 8 - 4
    ux = lx + torch.rand(n, generator=g) * 3 + 1e-3
    ly = torch.rand(n, generator=g) * 4 - 2
    uy = ly + torch.rand(n, generator=g) * 2 + 1e-3
    grid = pt.get_linspace_batched(lx, ux, ly, uy)
    xs, ys = grid[..., 0], grid[..., 1]
    zs = (1.0 - torch.sigmoid(xs)) * (torch.sigmoid(xs) * ys)
    R = pt.lin_reg_tensor_batched(grid, zs)
    A, B, C = R[:, 1], R[:, 2], R[:, 0]
    return lx, ux, ly, uy, A, B, C


def brute_force_max_abs_error(lx, ux, ly, uy, A, B, C, nx=2001, ny=41):
    t_x = torch.linspace(0, 1, nx)
    t_y = torch.linspace(0, 1, ny)
    best = torch.zeros_like(lx)
    for ty in t_y:
        y = (ly + ty * (uy - ly)).unsqueeze(1)
        x = lx.unsqueeze(1) + t_x.unsqueeze(0) * (ux - lx).unsqueeze(1)
        s = torch.sigmoid(x)
        e = (1.0 - s) * (s * y) - (A.unsqueeze(1) * x + B.unsqueeze(1) * y + C.unsqueeze(1))
        best = torch.maximum(best, e.abs().max(dim=1).values)
    return best


def check_max_error():
    all_ok = True
    for seed in range(3):
        lx, ux, ly, uy, A, B, C = realistic_boxes(400, seed)
        brute = brute_force_max_abs_error(lx, ux, ly, uy, A, B, C)
        abcs_tensor = torch.stack((A, B, C), dim=1)
        abcs = [(A[i], B[i], C[i]) for i in range(A.shape[0])]
        for vectorized in (True, False):
            set_flags(vectorized, real_cubic=True)
            if vectorized:
                err = pt.compute_max_error(lx, ux, ly, uy, None, abcs_tensor)
            else:
                err = torch.stack(pt.compute_max_error(lx, ux, ly, uy, abcs))
            short = (brute - err) / brute.clamp_min(1e-300)
            worst = short.max().item()
            n_bad = int((short > 1e-12).sum().item())
            path = "vectorized" if vectorized else "original"
            all_ok &= report(n_bad == 0, f"max error >= brute force, {path}, seed {seed}",
                             f"neurons below={n_bad}/400 worst shortfall={worst:.3e}")
    return all_ok


def sampled_max_derivative(net, img, l_val, r_val, n_samples=4001):
    ts = torch.linspace(l_val, r_val, n_samples)
    x = (1 - ts[:, None]) * img[None, :] + ts[:, None]
    _, d = torch.func.jvp(net, (x,), ((1 - img)[None, :].expand_as(x),))
    return d.abs().max().item()


def check_end_to_end():
    from measure_split_sweep import build_forward_precise, eval_splits_once, load_network_and_images

    # (network, position among correctly classified images, eps index, n_splits, vectorized)
    cases = [
        ("3layer", 0, 8, 8, True),
        ("3layer", 2, 8, 16, True),
        ("4layer", 0, 8, 4, True),
        ("4layer", 2, 15, 1, True),
        ("big", 1, 8, 4, True),
        ("3layer", 0, 8, 8, False),
    ]
    all_ok = True
    loaded = {}
    for net_name, img_pos, eps_idx, n_splits, vectorized in cases:
        if net_name not in loaded:
            loaded[net_name] = load_network_and_images(net_name, CPU, torch.float64, 3)
        net, images = loaded[net_name]
        image_id, img = images[img_pos]
        eps = EPS_SCHEDULE[eps_idx]
        set_flags(vectorized, real_cubic=True)
        lc = eval_splits_once(build_forward_precise(pt, net, net_name), img, eps, n_splits, 7, CPU, torch.float64)
        sampled = sampled_max_derivative(net, img, 0.0, eps)
        path = "vectorized" if vectorized else "original"
        all_ok &= report(lc >= sampled, f"{net_name} img {image_id} eps_idx {eps_idx} n_splits {n_splits} {path}",
                         f"lc/sampled_max={lc / sampled:.6f}")
    return all_ok


def main():
    torch.set_default_dtype(torch.float64)
    all_ok = True
    print("--- negation ---")
    all_ok &= check_negation()
    print("--- approximation error over the box ---")
    all_ok &= check_max_error()
    print("--- lc_precise against sampled autograd derivative ---")
    all_ok &= check_end_to_end()
    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
