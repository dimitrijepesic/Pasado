"""Finds the first place where the precise forward pass on a device differs from CPU.

It records the inputs and outputs of the regression, the corner check and the nonlinear
boundary check on both devices, and compares them call by call. For the first boundary
check whose result differs, it feeds the same CPU inputs to the device and compares the
steps of the cubic root solve.

Run: python experiments/diagnose_device_diff.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--dtype", choices=["float32", "float64"], default=None)
ap.add_argument("--epsilon", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--tol", type=float, default=1e-9, help="relative difference that counts as different")
args = ap.parse_args()
dtype = args.dtype or ("float32" if args.device.startswith("mps") else "float64")

import torchvision                                   # noqa: E402
import torchvision.transforms as transforms          # noqa: E402
from SimpleZono import *                             # noqa: E402,F401,F403
from Duals import *                                  # noqa: E402,F401,F403
import precise_transformer as pt                     # noqa: E402
from model import FCN                                # noqa: E402

# HyperDuals.py sets the default dtype to float64 on import, so set it after the imports.
torch.set_default_dtype(getattr(torch, dtype))
pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = True
pt.USE_REAL_CUBIC = False

REAL_LSTSQ = pt.lin_reg_tensor_batched
ORIG_CORNERS = pt.check_corners_tensor
ORIG_BOUNDARY = pt.check_nonlinear_boundary_tensor
L = -1.0 / (6.0 * np.sqrt(3.0))
U = -L


def cpu(x):
    if isinstance(x, tuple):
        return tuple(cpu(t) for t in x)
    return x.detach().cpu().clone()


def compare(a, b):
    """(largest relative difference over finite entries, entries whose NaN/inf pattern differs)."""
    if isinstance(a, tuple):
        rs = [compare(x, y) for x, y in zip(a, b)]
        return max(r[0] for r in rs), sum(r[1] for r in rs)
    a, b = a.double(), b.double()
    both = torch.isfinite(a) & torch.isfinite(b)
    pattern = int((torch.isfinite(a) != torch.isfinite(b)).sum() + (torch.isnan(a) != torch.isnan(b)).sum())
    if not both.any():
        return 0.0, pattern
    d = (a[both] - b[both]).abs() / b[both].abs().clamp_min(1e-300)
    return d.max().item(), pattern


def run(device):
    """One precise forward pass; returns the recorded calls and the final bound."""
    rec = {"lstsq": [], "corners": [], "boundary": []}

    def lstsq(x, z):
        out = REAL_LSTSQ(x.cpu(), z.cpu()).to(x.device)
        rec["lstsq"].append(((cpu(x), cpu(z)), cpu(out)))
        return out

    def corners(*a):
        out = ORIG_CORNERS(*a)
        rec["corners"].append((cpu(tuple(a)), cpu(out)))
        return out

    def boundary(*a):
        out = ORIG_BOUNDARY(*a)
        rec["boundary"].append((cpu(tuple(a)), cpu(out)))
        return out

    pt.lin_reg_tensor_batched, pt.check_corners_tensor, pt.check_nonlinear_boundary_tensor = lstsq, corners, boundary
    try:
        net = FCN(3)
        net.load_state_dict(torch.load(os.path.join(ROOT, "Section_5_4", "trained", "model_3layer.pth"),
                                       map_location="cpu"))
        net = net.eval().to(getattr(torch, dtype)).to(device)
        testset = torchvision.datasets.MNIST(root=os.path.join(ROOT, "Section_5_4", "MNIST_Data"), train=False,
                                             download=True, transform=transforms.ToTensor())
        img = testset[0][0].flatten().to(getattr(torch, dtype)).to(device)
        ie = DualIntervalTensor(real_l=torch.tensor([0.0], device=device),
                                real_u=torch.tensor([args.epsilon], device=device))
        ie.e1_l[0] = 1
        ie.e1_u[0] = 1
        ze = HyperDualIntervalToDualZonotope(ie)
        x = (torch.tensor([1], device=device) - ze) * img + ze
        np.random.seed(args.seed)
        with torch.no_grad():
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
            x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
            x = pt.AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
        lc = torch.max(torch.maximum(x.dual.get_lb().abs(), x.dual.get_ub().abs())).item()
    finally:
        pt.lin_reg_tensor_batched, pt.check_corners_tensor, pt.check_nonlinear_boundary_tensor = (
            REAL_LSTSQ, ORIG_CORNERS, ORIG_BOUNDARY)
    return rec, lc


def stage_report(cpu_args, device):
    """Steps of the cubic root solve for one boundary call, on CPU and on the device."""
    lx, ux, ly, uy, abc = cpu_args
    for tag, y in (("ly", ly), ("uy", uy)):
        per = {}
        for dev in ("cpu", device):
            to = lambda t: t.to(dev)
            A = to(abc)[:, 0]
            r = pt.filter_range(L, U, pt.inf_to_nan((A / to(y)).clone()))
            roots_c = pt.inverse_poly_tensor(r.clone())
            roots = pt.inverse_sigmoid_2nd_deriv(r.clone())
            filt = [pt.filter_range(to(lx), to(ux), t.clone()) for t in roots]
            ok = ~torch.isnan(r)
            per[dev] = dict(
                r=r.cpu(), roots_c=[t.cpu() for t in roots_c], roots=[t.cpu() for t in roots],
                in_range=int(ok.sum()),
                nonreal=[int((~torch.isreal(t) & ok).sum()) for t in roots_c],
                valid=[int((~torch.isnan(t)).sum()) for t in roots],
                kept=[int((~torch.isnan(t)).sum()) for t in filt])
        c, d = per["cpu"], per[device]
        print(f"  [{tag}] neurons with A/y in range: cpu {c['in_range']}, {device} {d['in_range']}")
        print(f"       roots with nonzero imaginary part (cpu | {device}): {c['nonreal']} | {d['nonreal']}")
        print(f"       roots in (0,1) after inverse sigmoid (cpu | {device}): {c['valid']} | {d['valid']}")
        print(f"       roots kept inside [lx,ux] (cpu | {device}): {c['kept']} | {d['kept']}")
        for k in range(3):
            re = compare(c["roots_c"][k].real, d["roots_c"][k].real)
            im = compare(c["roots_c"][k].imag, d["roots_c"][k].imag)
            print(f"       root {k}: complex64 value rel diff re {re[0]:.1e} im {im[0]:.1e}")


def main():
    cpu_rec, cpu_lc = run("cpu")
    dev_rec, dev_lc = run(args.device)
    print(f"dtype {dtype}: lc cpu {cpu_lc:.6f}, {args.device} {dev_lc:.6f}, "
          f"relative difference {abs(cpu_lc - dev_lc) / cpu_lc:.2e}\n")

    print("call by call (relative difference of inputs / outputs; NaN-inf pattern mismatches in brackets)")
    first = None
    for name in ("lstsq", "corners", "boundary"):
        for i, ((ci, co), (di, do)) in enumerate(zip(cpu_rec[name], dev_rec[name])):
            ri, pi = compare(ci, di)
            ro, po = compare(co, do)
            bad = max(ri, ro) > args.tol or pi or po
            print(f"  {name:9s} layer {i}: inputs {ri:.1e} [{pi}]  outputs {ro:.1e} [{po}]  {'DIFFERENT' if bad else 'same'}")
            if name == "boundary" and bad and first is None:
                first = (i, ci, co)
    if first is None:
        print("\nno boundary call differs beyond the tolerance")
        return
    i, ci, co = first
    print(f"\nboundary check of layer {i}: same CPU inputs on {args.device}, compared with the CPU result")
    out = pt.check_nonlinear_boundary_tensor(*[t.to(args.device) for t in ci])
    r, p = compare(cpu(out), co)
    print(f"  outputs: relative difference {r:.1e}, pattern mismatches {p}")
    print("  steps of the cubic root solve:")
    stage_report(ci, args.device)


if __name__ == "__main__":
    main()
