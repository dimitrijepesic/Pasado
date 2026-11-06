"""Tests of the preliminary outward rounding of D (batched_splits.USE_OUTWARD_D,
results/outward_rounding_prototype_2026-11/PROTOCOL.md).

1. Switch off: split bounds bit-identical to batched_splits.py before the prototype (1d754c9).
2. Switch on: D_out >= the exact maximum on the 4992 archived boxes of the D contract.
3. Switch on: every bound >= the bound with the switch off and >= the sampled autograd maximum.

Run: python experiments/test_outward_d.py
"""
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "forward_mode_tensorized_src"))
sys.path.insert(0, HERE)
import batched_splits as bs  # noqa: E402
import measure_split_sweep as mss  # noqa: E402
import precise_transformer as pt  # noqa: E402
from check_sweep_soundness import sampled_max_derivative  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


REF_COMMIT = "1d754c9"  # origin/gpu-hybrid, the last commit before USE_OUTWARD_D existed


def head_module():
    """batched_splits.py at REF_COMMIT, imported under another name."""
    src = subprocess.run(["git", "-C", ROOT, "show", f"{REF_COMMIT}:forward_mode_tensorized_src/batched_splits.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "batched_splits_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("batched_splits_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    torch.set_default_dtype(torch.float64)
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
    dev = torch.device("cpu")
    head = head_module()
    cases = [("3layer", 1), ("3layer", 8), ("5layer", 4), ("big", 2)]
    for net_name, S in cases:
        net, images = mss.load_network_and_images(net_name, dev, torch.float64, 2)
        layers = bs.extract_affine_layers(net, net_name)
        for image_id, img in images:
            for ei in (0, 8):
                bs.USE_OUTWARD_D = False
                off = bs.split_bounds(layers, img, EPS[ei], S, seed=7)
                ref = head.split_bounds(layers, img, EPS[ei], S, seed=7)
                check(f"off == {REF_COMMIT} bit for bit {net_name} S={S} img={image_id} eps={ei}", torch.equal(off, ref))
                bs.USE_OUTWARD_D = True
                on = bs.split_bounds(layers, img, EPS[ei], S, seed=7)
                bs.USE_OUTWARD_D = False
                check(f"on >= off {net_name} S={S} img={image_id} eps={ei}", bool((on >= off).all()),
                      f"(max rel increase {float(((on - off) / off).max()):.2e})")
        # sampled autograd maximum with the switch on, one image, eps index 0
        import check_sweep_soundness as css
        net64 = css.load_network(net_name)
        img = images[0][1]
        bs.USE_OUTWARD_D = True
        on = float(bs.split_bounds(layers, img, EPS[0], S, seed=7).max())
        bs.USE_OUTWARD_D = False
        smax = sampled_max_derivative(net64, img, EPS[0], 4001)
        check(f"on >= sampled max {net_name} S={S}", on >= smax, f"({on:.6g} vs {smax:.6g})")

    # coverage on the archived boxes of the D contract
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "compiled_sigmoid_v3_cuda_2026-10-29",
                                                 "gate_d_contract.csv"))))
    col = lambda k: torch.tensor([float(r[f"eager_{k}"]) for r in rows])  # noqa: E731
    A, B, C, lx, ux, ly, uy = (col(k) for k in ("A", "B", "C", "lx", "ux", "ly", "uy"))
    abc = torch.stack((A, B, C), dim=1)
    e_ly, e_uy = pt.check_nonlinear_boundary_tensor(lx, ux, ly, uy, abc)
    D = torch.maximum(torch.maximum(pt.check_corners_tensor(abc, lx, ux, ly, uy), e_ly), e_uy)
    D_out = bs.outward_error(D, abc, lx, ux, ly, uy)
    D_true = col("D_true")
    below_before, below_after = int((D < D_true).sum()), int((D_out < D_true).sum())
    check("coverage: D_out >= exact maximum on every archived box", below_after == 0,
          f"(below before: {below_before} of {len(rows)}, after: {below_after}; "
          f"median D_out/D - 1: {float(((D_out - D) / D).median()):.2e})")
    print("ALL PASS" if not FAIL else f"FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
