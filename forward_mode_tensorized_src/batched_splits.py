"""Interval splitting with all sub-intervals in one batched forward pass.

--n-splits divides [0, epsilon] into S pieces and bounds the Lipschitz constant on
each. The original code runs those pieces one after another in a Python loop. The
pieces are independent and have the same shapes, so this module stacks them on a
leading batch axis instead:

    centers     [S, N]        generators     [S, G, N]

and runs one pass of the precise forward analysis over the whole stack. Per split
the arithmetic is the same as in the sequential code (SimpleZono.py and
precise_transformer.py), so the per-split bounds agree with it up to floating
point reordering inside matmul and sum.

Only the precise path for fully connected sigmoid networks is covered. The code
stays outside precise_transformer.py so the sequential path is untouched and can
serve as the reference.

Differences from the sequential code, all numerically irrelevant:
  * The "all lower bounds equal all upper bounds" shortcut in SigmoidZonotope is
    not taken. The general formula gives the same bounds there.
  * Zero rows that the sequential code appends only to align generator counts are
    not materialised when they would be multiplied away.

Memory grows linearly with S (the sequential loop reuses the same buffers), so
`chunk` caps how many splits are processed at once.
"""
import numpy as np
import torch

import precise_transformer as pt


class BatchedDualZonotope:
    """S independent dual zonotopes over N variables with G noise symbols each."""

    __slots__ = ("rc", "rg", "dc", "dg")

    def __init__(self, rc, rg, dc, dg):
        self.rc = rc  # real centers      [S, N]
        self.rg = rg  # real generators   [S, G, N]
        self.dc = dc  # dual centers      [S, N]
        self.dg = dg  # dual generators   [S, G, N]


def extract_affine_layers(net, network_name):
    """[(weight.T, bias), ...] for the hidden layers followed by the output layer.

    The tensors are detached, so later changes to the module (net.to(...), training)
    do not reach them.
    """
    if network_name == "big":
        modules = [net[i] for i in (1, 3, 5, 7, 9)]
    else:
        layers = int(network_name[0])
        names = ["fc1", "fc2"] + (["fc3"] if layers >= 4 else []) + (["fc4"] if layers == 5 else [])
        names.append("fc_final")
        modules = [getattr(net, n) for n in names]
    return [(m.weight.detach().T, m.bias.detach()) for m in modules]


def hazed_input(img, lows, highs):
    """(1 - eps) * img + eps for eps in each [lows[s], highs[s]], as a batched dual zonotope.

    Written out in the operation order of the sequential expression
    (one - zono_eps) * img + zono_eps, so each split gets the same values.
    The two noise symbols are the real one (the interval of eps) and the dual one,
    whose generators are zero because the dual part of eps is the exact value 1.
    """
    mid = (highs + lows) / 2.0
    rad = (highs - lows) / 2.0
    S, N = lows.shape[0], img.shape[0]
    rc = (1.0 - mid).unsqueeze(1) * img + mid.unsqueeze(1)
    rg = img.new_zeros((S, 2, N))
    rg[:, 0, :] = (-rad).unsqueeze(1) * img + rad.unsqueeze(1)
    dc = (-img + 1.0).unsqueeze(0).expand(S, N).contiguous()
    dg = img.new_zeros((S, 2, N))
    return BatchedDualZonotope(rc, rg, dc, dg)


def affine(z, weight_t, bias):
    """AffineDualZonotope(z, weight_t) + bias. The dual part gets no bias."""
    return BatchedDualZonotope(z.rc @ weight_t + bias, z.rg @ weight_t,
                               z.dc @ weight_t, z.dg @ weight_t)


def _bounds(centers, generators):
    radius = generators.abs().sum(dim=1)
    return centers - radius, centers + radius


def _perturbation(S, N, seed, dtype, device):
    """The noise added to coefficient A ([S, N]).

    With a seed the sequential protocol reseeds np.random before every split, so
    every split sees the same stream. The caller reseeds once per forward pass and
    the N values drawn here are shared by all splits. Without a seed each (split,
    neuron) pair draws its own value.
    """
    if seed is not None:
        values = torch.as_tensor(np.random.normal(0, .0001, size=N), dtype=dtype, device=device)
        return values.unsqueeze(0).expand(S, N)
    values = torch.as_tensor(np.random.normal(0, .0001, size=(S, N)), dtype=dtype, device=device)
    return values


def precise_sigmoid(z, seed):
    """PreciseSigmoidDualZonotope for S splits at once. Adds 2N noise symbols."""
    S, G0, N = z.rg.shape
    dtype, device = z.rc.dtype, z.rc.device

    lx, ux = _bounds(z.rc, z.rg)
    ly, uy = _bounds(z.dc, z.dg)

    # SigmoidZonotope on the real part
    s_lb, s_ub = torch.sigmoid(lx), torch.sigmoid(ux)
    lam = torch.min(s_lb * (1 - s_lb), s_ub * (1 - s_ub))
    mu1 = 0.5 * (s_ub + s_lb - lam * (ux + lx))
    mu2 = 0.5 * (s_ub - s_lb - lam * (ux - lx))
    sig_c = lam * z.rc + mu1

    # Planar regression of sigmoid'(x) * y on the 5 x 5 grid of every (split, neuron)
    # box. The boxes are independent, so [S, N] flattens to one batch of S*N.
    lx_f, ux_f, ly_f, uy_f = (t.reshape(-1) for t in (lx, ux, ly, uy))
    grid = pt.get_linspace_batched(lx_f, ux_f, ly_f, uy_f)  # [S*N, 25, 2]
    gx, gy = grid[..., 0], grid[..., 1]
    targets = (1.0 - torch.sigmoid(gx)) * (torch.sigmoid(gx) * gy)
    coef = pt.lin_reg_tensor_batched(grid, targets)  # [S*N, 3], intercept first
    As = coef[:, 1] + _perturbation(S, N, seed, dtype, device).reshape(-1)
    Bs = coef[:, 2]
    Cs = coef[:, 0]
    assert bool(((As > 0) | (As < 0)).all())

    # worst-case approximation error, same three checks as compute_max_error
    abc = torch.stack((As, Bs, Cs), dim=1)
    corners = pt.check_corners_tensor(abc, lx_f, ux_f, ly_f, uy_f)
    err_ly, err_uy = pt.check_nonlinear_boundary_tensor(lx_f, ux_f, ly_f, uy_f, abc)
    errors = torch.maximum(torch.maximum(corners, err_ly), err_uy).reshape(S, N)

    As, Bs, Cs = As.reshape(S, N), Bs.reshape(S, N), Cs.reshape(S, N)
    dual_c = As * z.rc + Bs * z.dc + Cs

    # Generator layout per split, G0 + 2N rows:
    #   real: [lam * rg | diag(mu2) | 0]      dual: [A * rg + B * dg | 0 | diag(errors)]
    new_rg = z.rg.new_zeros((S, G0 + 2 * N, N))
    new_rg[:, :G0, :] = lam.unsqueeze(1) * z.rg
    new_rg[:, G0:G0 + N, :].diagonal(dim1=1, dim2=2).copy_(mu2)

    new_dg = z.rg.new_zeros((S, G0 + 2 * N, N))
    new_dg[:, :G0, :] = As.unsqueeze(1) * z.rg + Bs.unsqueeze(1) * z.dg
    new_dg[:, G0 + N:, :].diagonal(dim1=1, dim2=2).copy_(errors)

    return BatchedDualZonotope(sig_c, new_rg, dual_c, new_dg)


def _forward_chunk(layers, img, lows, highs, seed):
    z = hazed_input(img, lows, highs)
    if seed is not None:
        np.random.seed(seed)
    for weight_t, bias in layers[:-1]:
        z = precise_sigmoid(affine(z, weight_t, bias), seed)
    weight_t, bias = layers[-1]
    z = affine(z, weight_t, bias)
    lb, ub = _bounds(z.dc, z.dg)
    return torch.maximum(lb.abs(), ub.abs()).amax(dim=1)  # [S]


def split_bounds(layers, img, epsilon, n_splits, seed=None, chunk=None):
    """Lipschitz bound of every sub-interval of [0, epsilon], as an [n_splits] tensor.

    layers comes from extract_affine_layers. seed matches the sequential
    protocol (np.random reseeded before each split); None draws fresh noise.
    chunk limits how many splits go through the network together; None means all.
    The maximum of the result is the bound for the whole interval.
    """
    dtype, device = img.dtype, img.device
    delta = epsilon / n_splits
    # floats first, as in the sequential code, so the endpoints are identical
    lows_f = [i * delta for i in range(n_splits)]
    highs_f = [low + delta for low in lows_f]
    lows = torch.tensor(lows_f, dtype=dtype, device=device)
    highs = torch.tensor(highs_f, dtype=dtype, device=device)

    chunk = n_splits if chunk is None else max(1, min(chunk, n_splits))
    with torch.no_grad():
        parts = [_forward_chunk(layers, img, lows[i:i + chunk], highs[i:i + chunk], seed)
                 for i in range(0, n_splits, chunk)]
    return torch.cat(parts)


def lipschitz_bound(layers, img, epsilon, n_splits, seed=None, chunk=None):
    """max over the splits, as a Python float."""
    return split_bounds(layers, img, epsilon, n_splits, seed, chunk).max().item()
