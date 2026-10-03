"""Probes behind the lstsq analysis: shapes and duplicate design matrices on a real
run, batched vs loop lstsq, gelsd vs the default driver on rank-deficient boxes,
and whether torch.linspace matches simple affine formulas. Changes no source."""
import hashlib
import os
import sys
from collections import Counter

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(REPO, "Section_5_4"))

from SimpleZono import *          # noqa: F401,F403
from Duals import *               # noqa: F401,F403
import precise_transformer as pt
from precise_transformer import PreciseSigmoidDualZonotope, AffineDualZonotope
from model import FCN

SEED = 12345


# ---------------------------------------------------------------- [shapes]
def probe_shapes():
    import torchvision
    import torchvision.transforms as transforms

    net = FCN(3)
    net.load_state_dict(torch.load(
        os.path.join(REPO, "Section_5_4", "trained", "model_3layer.pth"),
        map_location="cpu"))
    net.eval()
    testset = torchvision.datasets.MNIST(
        root=os.path.join(REPO, "Section_5_4", "MNIST_Data"), train=False,
        download=True, transform=transforms.ToTensor())
    correct_indices = torch.load(
        os.path.join(REPO, "Section_5_4", "trained", "indices_3layer.pth"),
        map_location="cpu")
    img = None
    idx = 0
    for (image, _) in torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False):
        if idx in correct_indices:
            img = image.flatten()
            break
        idx += 1

    shapes = Counter()
    dtypes = Counter()
    xplusone_hashes = Counter()
    orig = pt.lin_reg_tensor

    def recording(x, zs):
        shapes[(tuple(x.shape), tuple(zs.shape))] += 1
        dtypes[(str(x.dtype), str(zs.dtype))] += 1
        xp = torch.cat((torch.ones(x.size(0), 1), x), 1)
        xplusone_hashes[hashlib.sha1(xp.numpy().tobytes()).hexdigest()] += 1
        return orig(x, zs)

    def make_hazed(img_f, epsilon):
        ie = DualIntervalTensor(real_l=torch.tensor([0.0]), real_u=torch.tensor([epsilon]))
        ie.e1_l[0] = 1
        ie.e1_u[0] = 1
        return (torch.tensor([1]) - HyperDualIntervalToDualZonotope(ie)) * img_f \
            + HyperDualIntervalToDualZonotope(ie)

    pt.lin_reg_tensor = recording
    try:
        for eps in (2.0, 2e-3):
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            x = make_hazed(img, eps)
            x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
            x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
    finally:
        pt.lin_reg_tensor = orig

    total = sum(shapes.values())
    dupes = {h: c for h, c in xplusone_hashes.items() if c > 1}
    print(f"[shapes] calls={total} (2 layers x 100 neurons x 2 eps)")
    for (xs, zs), c in shapes.items():
        print(f"[shapes]   x{list(xs)} zs{list(zs)}  x{c}")
    for (dx, dz), c in dtypes.items():
        print(f"[shapes]   dtype x={dx} zs={dz}  x{c}")
    print(f"[shapes] distinct xplusone matrices: {len(xplusone_hashes)}/{total}  "
          f"exact duplicates: {sum(c - 1 for c in dupes.values())}")


# ---------------------------------------------------------------- [batched]
def probe_batched():
    g = torch.Generator().manual_seed(SEED)
    n, m = 64, 25
    A = torch.randn(n, m, 3, generator=g, dtype=torch.float64)
    A[:, :, 0] = 1.0  # intercept column, like xplusone
    B = torch.randn(n, m, 1, generator=g, dtype=torch.float64)

    for driver in ("gelsd", None):
        try:
            sol = torch.linalg.lstsq(A, B, driver=driver).solution
            loop = torch.stack([torch.linalg.lstsq(A[i], B[i], driver=driver).solution
                                for i in range(n)])
            d = (sol - loop).abs().max().item()
            print(f"[batched] driver={driver!s:6s} batched OK  shape={list(sol.shape)}  "
                  f"max|batched-loop|={d:.3e}")
        except Exception as e:
            print(f"[batched] driver={driver!s:6s} FAILED: {type(e).__name__}: {e}")


# ---------------------------------------------------------------- [drivers]
def probe_drivers():
    g = torch.Generator().manual_seed(SEED + 1)
    m = 25
    # well-conditioned system (random box grid, like a healthy neuron)
    x = torch.rand(m, 2, generator=g, dtype=torch.float64) * 4 - 2
    Awell = torch.cat((torch.ones(m, 1), x), 1)
    b = torch.randn(m, generator=g, dtype=torch.float64)

    s_gelsd = torch.linalg.lstsq(Awell, b, driver="gelsd").solution
    s_none = torch.linalg.lstsq(Awell, b).solution
    print(f"[drivers] well-conditioned: |gelsd-default|={(s_gelsd - s_none).abs().max():.3e}")

    # rank-deficient: degenerate box lx==ux -> x column constant -> rank 2
    xd = x.clone()
    xd[:, 0] = 0.7
    Adef = torch.cat((torch.ones(m, 1), xd), 1)
    r_gelsd = torch.linalg.lstsq(Adef, b, driver="gelsd")
    r_none = torch.linalg.lstsq(Adef, b)
    sd, sn = r_gelsd.solution, r_none.solution
    resid_d = (Adef @ sd - b).norm().item()
    resid_n = (Adef @ sn - b).norm().item()
    print(f"[drivers] rank-deficient (lx==ux): rank(gelsd)={int(r_gelsd.rank)}  "
          f"|gelsd-default|={(sd - sn).abs().max():.3e}  "
          f"residual gelsd={resid_d:.6f} default={resid_n:.6f}  "
          f"|coef| gelsd={sd.norm():.4f} default={sn.norm():.4f}")


# ---------------------------------------------------------------- [grid]
def probe_grid():
    g = torch.Generator().manual_seed(SEED + 2)
    mism_a = mism_b = mism_c = 0
    trials = 2000
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    for _ in range(trials):
        l = (torch.rand((), generator=g, dtype=torch.float64) * 8 - 4).item()
        u = l + (torch.rand((), generator=g, dtype=torch.float64) * 4).item()
        ref = torch.linspace(l, u, steps=5)
        alt_a = l + (u - l) / 4.0 * torch.arange(5, dtype=torch.float64)
        alt_b = l + (u - l) * t
        mism_a += 0 if torch.equal(ref, alt_a) else 1
        mism_b += 0 if torch.equal(ref, alt_b) else 1
        xs = torch.linspace(l, u, steps=5)
        ys = torch.linspace(u, l, steps=5)
        cp = torch.cartesian_prod(xs, ys)
        alt = torch.stack((xs.repeat_interleave(5), ys.repeat(5)), dim=1)
        mism_c += 0 if torch.equal(cp, alt) else 1
    print(f"[grid] linspace vs l+(u-l)/4*arange: {mism_a}/{trials} mismatched trials")
    print(f"[grid] linspace vs l+(u-l)*[0,.25,.5,.75,1]: {mism_b}/{trials} mismatched trials")
    print(f"[grid] cartesian_prod vs repeat_interleave/repeat: {mism_c}/{trials} mismatched trials")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  default_dtype={torch.get_default_dtype()}  "
          f"platform={sys.platform}")
    probe_batched()
    probe_drivers()
    probe_grid()
    probe_shapes()
