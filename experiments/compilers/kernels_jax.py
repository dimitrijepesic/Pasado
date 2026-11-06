"""JAX ports of the isolated kernels of kernels_torch.py, line by line from the library
(results/compilers_2026-11/PROTOCOL.md). float64 needs jax_enable_x64, set on import.

fit_planes  <- forward_mode_tensorized_src/planar_fit.py (same operations, same order)
viete       <- precise_transformer.inverse_poly_real_tensor + inv_sigmoid
bounds      <- batched_splits.affine + _bounds
complex_cubic <- precise_transformer.inverse_poly_tensor (complex64, as in the original)
"""
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

PORTED = ("fit", "viete", "bounds", "complex_cubic")


# ---- planar_fit ---------------------------------------------------------------------------
def _grid_axes(grid):
    s = math.isqrt(grid.shape[1])
    return grid[:, ::s, 0], grid[:, :s, 1]


def _centred(v):
    r = v - v[:, :1]
    rm = r.mean(axis=1, keepdims=True)
    return r - rm, v[:, 0] + rm[:, 0]


def _pow2_scale(v):
    m = jnp.abs(v).max(axis=1)
    _, e = jnp.frexp(m)
    return jnp.ldexp(jnp.ones_like(m), e)


def _stats(grid):
    xs, ys = _grid_axes(grid)
    dx, xbar = _centred(xs)
    dy, ybar = _centred(ys)
    ax, ay = _pow2_scale(dx), _pow2_scale(dy)
    dxn, dyn = dx / ax[:, None], dy / ay[:, None]
    nxx = (dxn * dxn).sum(1)
    nyy = (dyn * dyn).sum(1)
    xdeg = (xs[:, 0] == xs[:, -1]) | ((dx * dx).sum(1) == 0)
    ydeg = (ys[:, 0] == ys[:, -1]) | ((dy * dy).sum(1) == 0)
    return dict(s=xs.shape[1], xbar=xbar, ybar=ybar, ax=ax, ay=ay, dxn=dxn, dyn=dyn,
                nxx=nxx, nyy=nyy, xdeg=xdeg, ydeg=ydeg)


def _safe(den, deg):
    return jnp.where(deg, jnp.ones_like(den), den)


def _min_norm(k, a0, b0, xbar, ybar, xdeg, ydeg):
    zero = jnp.zeros_like(k)
    u = jnp.where(xdeg, xbar, zero)
    v = jnp.where(ydeg, ybar, zero)
    m = jnp.maximum(jnp.ones_like(k), jnp.maximum(jnp.abs(u), jnp.abs(v)))
    inv, un, vn = 1.0 / m, u / m, v / m
    denom = inv * inv + un * un + vn * vn
    km = k / m
    C = km * inv / denom
    A = jnp.where(xdeg, km * un / denom, a0)
    B = jnp.where(ydeg, km * vn / denom, b0)
    return jnp.stack((C, A, B), axis=1)


def _fit_targets(st, z, xdeg, ydeg):
    n, s = z.shape[0], st["s"]
    z3 = z.reshape(n, s, s)
    zbar = z.mean(1)
    px = (z3 - z3[:, :1, :]).mean(2)
    px = px - px.mean(1, keepdims=True)
    py = (z3 - z3[:, :, :1]).mean(1)
    py = py - py.mean(1, keepdims=True)
    sx, sy = _pow2_scale(px), _pow2_scale(py)
    zero = jnp.zeros_like(zbar)
    a0 = jnp.where(xdeg, zero, (st["dxn"] * (px / sx[:, None])).sum(1)
                   / _safe(st["nxx"], xdeg) * sx / _safe(st["ax"], xdeg))
    b0 = jnp.where(ydeg, zero, (st["dyn"] * (py / sy[:, None])).sum(1)
                   / _safe(st["nyy"], ydeg) * sy / _safe(st["ay"], ydeg))
    k = zbar - a0 * st["xbar"] - b0 * st["ybar"]
    return _min_norm(k, a0, b0, st["xbar"], st["ybar"], xdeg, ydeg)


def _gram_eigenvalues(xbar, ybar, sx, sy, steps=8):
    one, zero = jnp.ones_like(xbar), jnp.zeros_like(xbar)
    m = jnp.maximum(jnp.maximum(one, jnp.maximum(jnp.abs(xbar), jnp.abs(ybar))), jnp.maximum(sx, sy))
    c2, a2, b2 = (one / m) ** 2, (xbar / m) ** 2, (ybar / m) ** 2
    p2, q2 = (sx / m) ** 2, (sy / m) ** 2
    e1 = c2 + a2 + b2 + p2 + q2
    e2 = p2 * (c2 + b2) + q2 * (c2 + a2) + p2 * q2
    e3 = c2 * p2 * q2
    l1 = e1
    for _ in range(steps):
        f = ((l1 - e1) * l1 + e2) * l1 - e3
        df = (3.0 * l1 - 2.0 * e1) * l1 + e2
        l1 = l1 - jnp.where(df > 0, f / _safe(df, df <= 0), zero)
    prod = e3 / l1
    total = jnp.maximum((e2 - prod) / l1, 0)
    disc = jnp.maximum(total * total - 4.0 * prod, 0)
    l2 = 0.5 * (total + jnp.sqrt(disc))
    l3 = jnp.where(l2 > 0, prod / _safe(l2, l2 <= 0), zero)
    return l1, l2, l3, p2 / (c2 + a2), q2 / (c2 + b2)


def fit_planes(grid, z):
    st = _stats(grid)
    rcond = jnp.finfo(grid.dtype).eps * max(grid.shape[1], 3)
    sx = st["ax"] * jnp.sqrt(st["nxx"] / st["s"])
    sy = st["ay"] * jnp.sqrt(st["nyy"] / st["s"])
    l1, l2, l3, tx, ty = _gram_eigenvalues(st["xbar"], st["ybar"], sx, sy)
    r2 = rcond * rcond
    drop2, drop3, pick_x = l2 <= r2 * l1, l3 <= r2 * l1, tx <= ty
    cx, cy = drop2 | (drop3 & pick_x), drop2 | (drop3 & ~pick_x)
    return _fit_targets(st, z, st["xdeg"] | cx, st["ydeg"] | cy)


# ---- cubic ---------------------------------------------------------------------------------
def viete(y, tol=1e-7):
    arg = 6.0 * math.sqrt(3.0) * y
    phi = jnp.arccos(jnp.clip(arg, -1.0, 1.0)) / 3.0
    phi = jnp.where(jnp.abs(arg) <= 1.0 + tol, phi, jnp.full_like(phi, jnp.nan))
    scale, two_pi_3 = 1.0 / math.sqrt(3.0), 2.0 * math.pi / 3.0
    roots = [scale * jnp.cos(phi - k * two_pi_3) + 0.5 for k in (0, 1, 2)]
    return jnp.stack([jnp.log(s / (1.0 - s)) for s in roots])


def complex_cubic(y):
    Q = jnp.float32(-1 / 12.)
    R = (0.25 * y).astype(jnp.complex64)
    J = jnp.complex64(1j)
    sqrt_arg = (Q ** 3 + R ** 2).astype(jnp.complex64)
    S = (R + jnp.sqrt(sqrt_arg)) ** 0.33333
    T = (R - jnp.sqrt(sqrt_arg)) ** 0.33333
    x1 = S + T + 0.5
    s32 = jnp.sqrt(jnp.float32(3.)) / 2.
    x2 = -0.5 * (S + T) + 0.5 + (s32 * J) * (S - T)
    x3 = -0.5 * (S + T) + 0.5 - (s32 * J) * (S - T)
    return jnp.stack([jnp.stack((r.real, r.imag), axis=-1) for r in (x1, x2, x3)])


# ---- zonotope bounds -----------------------------------------------------------------------
def bounds(c, rg, w):
    centers, generators = c @ w, rg @ w
    radius = jnp.abs(generators).sum(axis=1)
    return jnp.stack((centers - radius, centers + radius))


FUNCS = dict(fit=fit_planes, viete=viete, bounds=bounds, complex_cubic=complex_cubic)
