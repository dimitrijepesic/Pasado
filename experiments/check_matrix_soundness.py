"""Checks every per-image bound of a device-matrix half (postfix_device_matrix.py) against the
sampled autograd maximum of |dF/dt| along the haze path (4001 points).

  python experiments/check_matrix_soundness.py DIR [--configs full_f64_real] [--out DIR/soundness.csv]

Reports, per network and analysis (precise, zonotope, interval), the smallest ratio
bound / sampled maximum and the number of rows below 1.
"""
import argparse
import csv
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_sweep_soundness import load_network, sampled_max_derivative  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--configs", nargs="+", default=["full_f64_real"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torchvision
    import torchvision.transforms as transforms
    testset = torchvision.datasets.MNIST(root=os.path.join(os.path.dirname(HERE), "Section_5_4", "MNIST_Data"),
                                         train=False, download=True, transform=transforms.ToTensor())
    out, bad = [], 0
    for path in sorted(glob.glob(os.path.join(args.dir, "*.csv"))):
        name = os.path.basename(path)[:-4]
        network, config = name.split("_", 1)
        if config not in args.configs:
            continue
        net = load_network(network)
        rows = list(csv.DictReader(open(path)))
        cache = {}
        for r in rows:
            key = (int(r["image_id"]), int(r["eps_index"]))
            if key not in cache:
                img = testset[key[0]][0].flatten().to(torch.float64)
                cache[key] = sampled_max_derivative(net, img, EPS[key[1]], 4001)
            for col in ("lc_precise", "lc_zono", "lc_interval"):
                if r.get(col) in (None, ""):
                    continue
                ratio = float(r[col]) / cache[key]
                bad += ratio < 1
                out.append(dict(network=network, config=config, analysis=col, image_id=key[0], eps_index=key[1],
                                bound=float(r[col]), sampled_max=cache[key], ratio=ratio))
        for col in ("lc_precise", "lc_zono", "lc_interval"):
            sel = [o for o in out if o["network"] == network and o["config"] == config and o["analysis"] == col]
            if sel:
                print(f"{network:6} {config:20} {col:11} rows {len(sel):4}  smallest bound/sampled max "
                      f"{min(o['ratio'] for o in sel):.6f}  below: {sum(o['ratio'] < 1 for o in sel)}", flush=True)
    if args.out:
        with open(args.out, "w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("ALL SOUND" if bad == 0 else f"{bad} ROWS BELOW THE SAMPLED MAXIMUM")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
