"""Runs the precise forward pass on a device and compares it with the same pass on CPU.

Any tensor that is created on the wrong device makes PyTorch raise "Expected all
tensors to be on the same device", so a clean run means the device plumbing is complete.
It works with mps (float32 only) and with cuda (float64).

lstsq has no GPU version for gelsd, so for now it is run on CPU inside this script.
Remove lin_reg_on_cpu once the real CPU bridge is in precise_transformer.py.

On mps, a torch.cat of a CPU and a GPU tensor can crash the interpreter (exit code 139)
instead of raising an error. On cuda it raises a normal RuntimeError.

Run: python experiments/check_device_plumbing.py --device mps
     python experiments/check_device_plumbing.py --device cuda
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
ap.add_argument("--device", default="mps")
ap.add_argument("--dtype", choices=["float32", "float64"], default=None,
                help="default: float32 on mps, float64 otherwise")
ap.add_argument("--network", choices=["3layer"], default="3layer")
ap.add_argument("--epsilon", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

dtype = args.dtype or ("float32" if args.device.startswith("mps") else "float64")

import torchvision                                   # noqa: E402
import torchvision.transforms as transforms          # noqa: E402
from SimpleZono import *                             # noqa: E402,F401,F403
from Duals import *                                  # noqa: E402,F401,F403
import precise_transformer as pt                     # noqa: E402
from model import FCN                                # noqa: E402

# HyperDuals.py sets the default dtype to float64 when it is imported, so the dtype
# for this check has to be set after the imports.
torch.set_default_dtype(getattr(torch, dtype))

pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = True
pt.USE_REAL_CUBIC = False

_real_lin_reg = pt.lin_reg_tensor_batched


def lin_reg_on_cpu(x_batch, zs_batch):
    """Stand-in for the CPU bridge: solve on CPU, return on the input's device."""
    return _real_lin_reg(x_batch.cpu(), zs_batch.cpu()).to(x_batch.device)


def build_net(device):
    net = FCN(3)
    net.load_state_dict(torch.load(os.path.join(ROOT, "Section_5_4", "trained", "model_3layer.pth"),
                                   map_location="cpu"))
    net.eval()
    return net.to(getattr(torch, dtype)).to(device)


def precise_forward(net, img, device):
    """The 3layer precise forward pass of get_lipschitz.py, on `device`."""
    interval_eps = DualIntervalTensor(real_l=torch.tensor([0.0], device=device),
                                      real_u=torch.tensor([args.epsilon], device=device))
    interval_eps.e1_l[0] = 1
    interval_eps.e1_u[0] = 1
    zono_eps = HyperDualIntervalToDualZonotope(interval_eps)
    x = (torch.tensor([1], device=device) - zono_eps) * img + zono_eps
    np.random.seed(args.seed)
    with torch.no_grad():
        x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
        x = pt.PreciseSigmoidDualZonotope(pt.AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
        return pt.AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data


def bound(out):
    lb, ub = out.dual.get_lb(), out.dual.get_ub()
    return torch.max(torch.maximum(lb.abs(), ub.abs())).item()


def main():
    testset = torchvision.datasets.MNIST(root=os.path.join(ROOT, "Section_5_4", "MNIST_Data"),
                                         train=False, download=True, transform=transforms.ToTensor())
    img = testset[0][0].flatten().to(getattr(torch, dtype))
    device = torch.device(args.device)

    cpu_out = precise_forward(build_net("cpu"), img, "cpu")
    print(f"dtype {dtype}; CPU reference lc = {bound(cpu_out):.6f}")

    pt.lin_reg_tensor_batched = lin_reg_on_cpu
    try:
        dev_out = precise_forward(build_net(device), img.to(device), device)
    except RuntimeError as e:
        print(f"FAIL on {device}: {str(e).splitlines()[0][:150]}")
        return 1

    tensors = {"real centers": dev_out.real.centers, "real generators": dev_out.real.generators,
               "dual centers": dev_out.dual.centers, "dual generators": dev_out.dual.generators}
    ok = True
    for name, t in tensors.items():
        on_device = t.device.type == device.type
        ok &= on_device
        print(f"  {name:16s} on {t.device}  {'ok' if on_device else 'WRONG DEVICE'}")
    rel = abs(bound(dev_out) - bound(cpu_out)) / abs(bound(cpu_out))
    tol = 1e-3 if dtype == "float32" else 1e-9
    print(f"{device} lc = {bound(dev_out):.6f}, relative difference to CPU {rel:.2e} (limit {tol:.0e})")
    ok &= rel < tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
