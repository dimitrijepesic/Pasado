"""In float32 the computed max error D can fall below the true maximum, with gelsd.

One box recorded from the batched forward pass of 3layer (float32 bounds and the
float32 plane gelsd fitted there, noise on A included). The precise transformer
computes D from the corners and from the roots of sigmoid''(x) = A / y on the two
y-edges, with the real (Viete) cubic. Here A / y = 2.3e-7, so one root is
s = sigmoid(x) = 2.3e-7, and the float32 formula s = cos(...) / sqrt(3) + 0.5 loses
most of its digits to cancellation: s comes out 31% too large, x = log(s / (1 - s))
lands at -15.03 instead of -15.30, and E is evaluated away from its maximum.

The same float32 coefficients through the float64 pipeline give the true maximum,
so the shortfall comes from the float32 root, not from the plane.

Run:
  python experiments/reproduce_f32_root_shortfall.py
"""
import os
import sys

import mpmath
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "forward_mode_tensorized_src"))

import precise_transformer as pt  # noqa: E402

# exact float32 values
BOX = dict(lx=-27.481807708740234, ux=-12.281240463256836,
           ly=-48.06841278076172, uy=-48.06841278076172,
           A=-1.0905865565291606e-05, B=5.8534105846774764e-06, C=-1.2177254404832638e-07)


def pipeline(dtype):
    t = {k: torch.tensor([v], dtype=dtype) for k, v in BOX.items()}
    abc = torch.stack((t["A"], t["B"], t["C"]), dim=1)
    corners = pt.check_corners_tensor(abc, t["lx"], t["ux"], t["ly"], t["uy"]).item()
    e_ly, e_uy = pt.check_nonlinear_boundary_tensor(t["lx"], t["ux"], t["ly"], t["uy"], abc)
    roots = pt.inverse_poly_real_tensor(t["A"] / t["ly"])
    return max(corners, e_ly.item(), e_uy.item()), [r.item() for r in roots]


def true_maximum():
    mpmath.mp.dps = 50
    A, B, C, y0 = (mpmath.mpf(BOX[k]) for k in ("A", "B", "C", "ly"))

    def err(x):
        s = 1 / (1 + mpmath.exp(-x))
        return abs(s * (1 - s) * y0 - (A * x + B * y0 + C))

    best, at = max((err(mpmath.mpf(BOX[k])), BOX[k]) for k in ("lx", "ux"))
    for z in mpmath.polyroots([2, -3, 1, -A / y0], extraprec=200):
        z = mpmath.mpc(z)
        if abs(z.imag) < mpmath.mpf(10) ** -30 and 0 < z.real < 1:
            x = mpmath.log(z.real / (1 - z.real))
            if BOX["lx"] <= x <= BOX["ux"] and err(x) > best:
                best, at = err(x), (float(z.real), float(x))
    return float(best), at


def main():
    pt.USE_REAL_CUBIC = True
    true_d, at = true_maximum()
    print(f"true maximum (50 digits)       D = {true_d:.10e}  at s, x = {at}")
    for dtype in (torch.float32, torch.float64):
        d, roots = pipeline(dtype)
        print(f"{str(dtype):14s} pipeline       D = {d:.10e}  shortfall {max(0.0, true_d - d) / true_d:.2e}"
              f"  roots s = {roots}")
    d32, _ = pipeline(torch.float32)
    d64, _ = pipeline(torch.float64)
    reproduced = d32 < true_d * (1 - 1e-3) and abs(d64 - true_d) <= 1e-12 * true_d
    print("REPRODUCED" if reproduced else "NOT REPRODUCED")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
