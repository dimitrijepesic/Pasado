"""Measures how much the real (Viete) cubic solver moves lc_precise, and whether it stays valid.

For each network it runs get_lipschitz.py (precise only, seeded) with the complex64 cubic
solver and with the real one, and compares the 16 averaged bounds. As a check that both
are valid, it also computes, with autograd, the true largest |dF_j/dt| of each image over
t in [0, eps], averaged over the same images. Any valid bound must be at least that.

Run: python experiments/measure_real_cubic_shift.py [--networks 3layer big]
"""
import argparse
import ast
import os
import subprocess
import sys

import torch
import torchvision
import torchvision.transforms as transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEC = os.path.join(ROOT, "Section_5_4")
sys.path.insert(0, SEC)
from model import FCN, FCNBig                                  # noqa: E402

FLAGS = {"PASADO_VECTORIZED_PRECISE": "1", "PASADO_VEC_BOUNDARY": "1",
         "PASADO_BATCHED_LSTSQ": "1", "PASADO_BATCHED_GRID": "1"}
EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def lc_precise(network, real_cubic, seed):
    """lc_precise for all 16 epsilons and 30 images, from get_lipschitz.py."""
    env = {**os.environ, **FLAGS, "PASADO_REAL_CUBIC": "1" if real_cubic else "0"}
    p = subprocess.run([sys.executable, "get_lipschitz.py", "--network", network, "--precise-only",
                        "--seed", str(seed), "--no-save"], cwd=SEC, env=env, capture_output=True, text=True)
    p.check_returncode()
    line = next(l for l in p.stdout.splitlines() if l.startswith("lc_precise "))
    return ast.literal_eval(line[len("lc_precise "):])


def true_mean_max(network, n_t=41, n_images=30):
    """Mean over the images of max over t in [0, eps] and over outputs of |dF/dt|, per epsilon."""
    torch.set_default_dtype(torch.float64)
    layers = {"3layer": 3, "4layer": 4, "5layer": 5}
    net = FCNBig() if network == "big" else FCN(layers[network])
    net.load_state_dict(torch.load(os.path.join(SEC, "trained", f"model_{network}.pth"), map_location="cpu"))
    net.eval()
    correct = torch.load(os.path.join(SEC, "trained", f"indices_{network}.pth"), map_location="cpu")
    # the file is a list of 0-dim tensors; turn them into plain ints so set lookup works
    correct = {int(i) for i in correct}
    testset = torchvision.datasets.MNIST(root=os.path.join(SEC, "MNIST_Data"), train=False, download=True,
                                         transform=transforms.ToTensor())
    imgs = []
    for i in range(len(testset)):
        if i in correct:
            imgs.append(testset[i][0].flatten().double())
            if len(imgs) == n_images:
                break
    f = lambda x: net(x.view(1, -1)).squeeze(0)
    out = []
    for eps in EPS:
        total = 0.0
        for img in imgs:
            d = 1.0 - img                                     # x(t) = img + t * (1 - img)
            best = 0.0
            for t in torch.linspace(0, eps, n_t):
                _, jv = torch.func.jvp(f, ((img + t * d).detach(),), (d,))
                best = max(best, jv.abs().max().item())
            total += best
        out.append(total / len(imgs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    for net in args.networks:
        cplx = lc_precise(net, False, args.seed)
        real = lc_precise(net, True, args.seed)
        true = true_mean_max(net)
        shift = [abs(a - b) / abs(a) for a, b in zip(cplx, real)]
        print(f"{net}: complex vs real cubic, relative difference of lc_precise over 16 epsilons: "
              f"max {max(shift):.1e}, median {sorted(shift)[len(shift) // 2]:.1e}")
        for name, lc in (("complex", cplx), ("real", real)):
            ratio = min(a / b for a, b in zip(lc, true))
            print(f"   {name:8s} lowest lc / true maximum over the epsilons: {ratio:.3f}  "
                  f"({'valid everywhere' if ratio >= 1 else 'BELOW the true maximum somewhere'})")


if __name__ == "__main__":
    main()
