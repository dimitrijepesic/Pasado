"""End-to-end old-vs-new correctness harness for the precise Lipschitz path.

Reproduces exactly the precise forward pass of Section_5_4/get_lipschitz.py for
the 3layer network on a handful of (image, epsilon) pairs, and compares the
ORIGINAL per-neuron check path (PASADO_VECTORIZED_PRECISE off) against the
vectorized path (on).

Determinism: the only randomness in the precise path is np.random.normal inside
sigmoid_prime_product_tensor (a per-neuron perturbation of A applied *before* the
corner/boundary branch). Re-seeding np.random (and torch) identically before each
forward makes the ABCs identical between the two paths, so the two runs differ
only by floating-point reordering inside the corner/boundary checks (and its
propagation through later layers). Everything is float64, as in the benchmark.

Writes logs/correctness_results.csv.
"""
import csv
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(REPO, "Section_5_4"))

torch.set_default_dtype(torch.float64)

import torchvision
import torchvision.transforms as transforms

from SimpleZono import *          # noqa: F401,F403
from Duals import *               # noqa: F401,F403
import precise_transformer as pt
from precise_transformer import (
    PreciseSigmoidDualZonotope, AffineDualZonotope,
)
from model import FCN

CSV_PATH = os.path.join(
    REPO, "logs",
    "correctness_results_batched.csv" if "--batched" in sys.argv
    else "correctness_results.csv")
SEED = 12345
# tolerance justified by float64 op reordering propagated through 2 sigmoid layers
ATOL = 1e-9
RTOL = 1e-9


def build_net():
    net = FCN(3)
    net.load_state_dict(torch.load(
        os.path.join(REPO, "Section_5_4", "trained", "model_3layer.pth"),
        map_location="cpu"))
    net.eval()
    return net


def forward_zono_precise(net, x):
    x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
    x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
    x = AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data
    return x


def make_hazed(img_f, epsilon):
    interval_eps = DualIntervalTensor(real_l=torch.tensor([0.0]),
                                      real_u=torch.tensor([epsilon]))
    interval_eps.e1_l[0] = 1
    interval_eps.e1_u[0] = 1
    zono_eps = HyperDualIntervalToDualZonotope(interval_eps)
    return (torch.tensor([1]) - zono_eps) * img_f + zono_eps


# --batched: the "new" side additionally enables the batched-lstsq path, so the
# comparison becomes old-loop-everything vs vectorized-checks+batched-regression.
BATCHED = "--batched" in sys.argv


def run_precise(net, img_f, epsilon, vectorized):
    """Run one precise forward with the selector set; seeded for determinism."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    pt.USE_VECTORIZED_PRECISE = vectorized
    pt.USE_VECTORIZED_BOUNDARY = vectorized
    pt.USE_BATCHED_LSTSQ = vectorized and BATCHED
    with torch.no_grad():
        out = forward_zono_precise(net, make_hazed(img_f, epsilon))
    lb = out.dual.get_lb()
    ub = out.dual.get_ub()
    lc = torch.max(torch.maximum(torch.abs(lb), torch.abs(ub))).item()
    return {
        "lc": lc,
        "dual_centers": out.dual.centers.detach().clone(),
        "dual_generators": out.dual.generators.detach().clone(),
        "lb": lb.detach().clone(),
        "ub": ub.detach().clone(),
    }


def cmp_tensor(a, b):
    if a.shape != b.shape:
        return dict(shape_match=False, max_abs=float("nan"), max_rel=float("nan"),
                    dtype_match=(a.dtype == b.dtype), device_match=(a.device == b.device),
                    naninf_pos_match=False, close=False)
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-300)
    # NaN/Inf *positions* must agree exactly, not just their existence
    naninf_pos = bool(torch.equal(torch.isnan(a), torch.isnan(b))
                      and torch.equal(torch.isinf(a), torch.isinf(b)))
    try:
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
        close = True
    except AssertionError:
        close = False
    return dict(shape_match=True,
                max_abs=diff.max().item(),
                max_rel=(diff / denom).max().item(),
                dtype_match=(a.dtype == b.dtype),
                device_match=(a.device == b.device),
                naninf_pos_match=naninf_pos,
                close=close)


def main():
    net = build_net()

    testset = torchvision.datasets.MNIST(
        root=os.path.join(REPO, "Section_5_4", "MNIST_Data"), train=False,
        download=True, transform=transforms.ToTensor())
    correct_indices = torch.load(
        os.path.join(REPO, "Section_5_4", "trained", "indices_3layer.pth"),
        map_location="cpu")

    test_range = [10 ** (-k / 4) * 2 for k in range(2, 18)]
    epsilons = [("largest", test_range[0]),
                ("middle", test_range[len(test_range) // 2]),
                ("smallest", test_range[-1])]

    # first few correct-index images
    n_images = 3
    imgs = []
    idx = 0
    for (image, _) in torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False):
        if idx in correct_indices:
            imgs.append(image.flatten())
            if len(imgs) == n_images:
                break
        idx += 1

    rows = []
    all_pass = True
    for img_i, img_f in enumerate(imgs):
        for eps_name, eps in epsilons:
            old = run_precise(net, img_f, eps, vectorized=False)
            new = run_precise(net, img_f, eps, vectorized=True)

            lc_abs = abs(old["lc"] - new["lc"])
            lc_rel = lc_abs / max(abs(old["lc"]), 1e-300)
            gen = cmp_tensor(old["dual_generators"], new["dual_generators"])
            cen = cmp_tensor(old["dual_centers"], new["dual_centers"])
            lb = cmp_tensor(old["lb"], new["lb"])
            ub = cmp_tensor(old["ub"], new["ub"])

            nan_old = bool(torch.isnan(old["dual_generators"]).any() or torch.isinf(old["dual_generators"]).any())
            nan_new = bool(torch.isnan(new["dual_generators"]).any() or torch.isinf(new["dual_generators"]).any())

            checks = [gen, cen, lb, ub]
            ok = (lc_abs <= ATOL + RTOL * abs(old["lc"])
                  and all(c["shape_match"] for c in checks)
                  and all(c["dtype_match"] for c in checks)
                  and all(c["device_match"] for c in checks)
                  and all(c["naninf_pos_match"] for c in checks)
                  and all(c["close"] for c in checks))
            all_pass = all_pass and ok

            def add(name, old_v, new_v, a, r):
                rows.append({
                    "case": f"img{img_i}", "network": "3layer",
                    "epsilon": f"{eps:.6g}", "eps_name": eps_name, "output": name,
                    "old_value": f"{old_v:.12g}", "new_value": f"{new_v:.12g}",
                    "abs_diff": f"{a:.3e}", "rel_diff": f"{r:.3e}",
                    "pass": "PASS" if ok else "FAIL",
                })

            add("lc_bound", old["lc"], new["lc"], lc_abs, lc_rel)
            add("dual_generators_maxabs", 0.0, 0.0, gen["max_abs"], gen["max_rel"])
            add("dual_centers_maxabs", 0.0, 0.0, cen["max_abs"], cen["max_rel"])
            add("dual_lb_maxabs", 0.0, 0.0, lb["max_abs"], lb["max_rel"])
            add("dual_ub_maxabs", 0.0, 0.0, ub["max_abs"], ub["max_rel"])

            print(f"[{'PASS' if ok else 'FAIL'}] img{img_i} eps={eps_name:8s} "
                  f"lc_old={old['lc']:.10g} lc_new={new['lc']:.10g} "
                  f"lc_absdiff={lc_abs:.2e} gen_absdiff={gen['max_abs']:.2e} "
                  f"shapes={old['dual_generators'].shape}=={new['dual_generators'].shape} "
                  f"dtype={old['dual_generators'].dtype} "
                  f"device={old['dual_generators'].device} "
                  f"naninf_pos_match={all(c['naninf_pos_match'] for c in checks)} "
                  f"nan={nan_old}/{nan_new}")

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "case", "network", "epsilon", "eps_name", "output",
            "old_value", "new_value", "abs_diff", "rel_diff", "pass"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {CSV_PATH} ({len(rows)} rows)")
    print("ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
