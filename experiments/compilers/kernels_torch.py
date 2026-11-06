"""Isolated Pasado kernels for the compiler comparison (results/compilers_2026-11/PROTOCOL.md).

Every kernel calls the library function itself; only the inputs are built here, from a
fixed seed, in float64. make(kernel, size, device) -> (fn, inputs).
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import batched_splits as bs  # noqa: E402
import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402

pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
KERNELS = ("fit", "viete", "bounds", "complex_cubic", "lstsq", "sigmoid")
DEFAULT_SIZES = {"fit": ["1024", "65536"], "viete": ["65536"], "bounds": ["small", "big"],
                 "complex_cubic": ["65536"], "lstsq": ["1024"], "sigmoid": ["3layer_S1", "big_S8"]}
BOUNDS_CONFIGS = {"small": (64, 202, 100), "big": (8, 2050, 1024)}  # S, G, N


def boxes(n, g):
    """Realistic boxes: x centres ~ N(0, 3), widths 1e-6..3; y narrower; 10% with ly == uy."""
    dt = torch.float64
    cx = torch.randn(n, generator=g, dtype=dt) * 3
    wx = 10 ** (torch.rand(n, generator=g, dtype=dt) * 6.5 - 6)
    cy = torch.randn(n, generator=g, dtype=dt) * 0.5
    wy = 10 ** (torch.rand(n, generator=g, dtype=dt) * 6 - 6)
    wy = torch.where(torch.rand(n, generator=g) < 0.1, torch.zeros_like(wy), wy)
    return cx - wx / 2, cx + wx / 2, cy - wy / 2, cy + wy / 2


def fit(grid, z):
    return planar_fit.fit_planes(grid, z)


def viete(y):
    return torch.stack(pt.inverse_sigmoid_2nd_deriv(y))


def bounds(c, rg, w):
    lb, ub = bs._bounds(c @ w, rg @ w)
    return torch.stack((lb, ub))


def complex_cubic(y):
    return torch.stack([torch.view_as_real(r) for r in pt.inverse_poly_tensor(y)])


def lstsq(grid, z):
    return pt.lin_reg_tensor_batched(grid, z)


def sigmoid(rc, rg, dc, dg, noise):
    out, _ = bs.precise_sigmoid(bs.BatchedDualZonotope(rc, rg, dc, dg), noise)
    return out.rc, out.rg, out.dc, out.dg


def make(kernel, size, device):
    """size: an integer n, or a config name; "name:k" scales S of a config by k."""
    g = torch.Generator().manual_seed(0)
    dev = torch.device(device)
    size, _, factor = str(size).partition(":")
    factor = int(factor or 1)
    if kernel in ("fit", "lstsq"):
        grid = pt.get_linspace_batched(*boxes(int(size), g))
        gx, gy = grid[..., 0], grid[..., 1]
        z = (1.0 - torch.sigmoid(gx)) * (torch.sigmoid(gx) * gy)
        return (fit if kernel == "fit" else lstsq), [grid.to(dev), z.to(dev)]
    if kernel in ("viete", "complex_cubic"):
        y = (torch.rand(int(size), generator=g, dtype=torch.float64) * 2 - 1) * 0.12
        return (viete if kernel == "viete" else complex_cubic), [y.to(dev)]
    if kernel == "bounds":
        S, G, N = BOUNDS_CONFIGS[size]
        S *= factor
        rg = torch.randn(S, G, N, generator=g, dtype=torch.float64) * 1e-2
        w = torch.randn(N, N, generator=g, dtype=torch.float64) / N ** 0.5
        c = torch.randn(S, N, generator=g, dtype=torch.float64)
        return bounds, [c.to(dev), rg.to(dev), w.to(dev)]
    if kernel == "sigmoid":
        import measure_split_sweep as mss
        net_name, s_txt = size.split("_S")
        S = int(s_txt)
        net, images = mss.load_network_and_images(net_name, dev, torch.float64, 1)
        layers = bs.extract_affine_layers(net, net_name)
        eps = 10 ** (-2 / 4) * 2
        lows = torch.arange(S, dtype=torch.float64, device=dev) * (eps / S)
        z = bs.affine(bs.hazed_input(images[0][1], lows, lows + eps / S), *layers[0])
        np.random.seed(7)
        noise = bs._perturbation(S, layers[0][0].shape[1], 7, torch.float64, dev)
        return sigmoid, [z.rc, z.rg, z.dc, z.dg, noise.contiguous()]
    raise ValueError(kernel)
