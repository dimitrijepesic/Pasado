"""Checks a split sweep CSV against a sampled autograd lower bound.

For every (network, image_id, epsilon) in the CSV written by measure_split_sweep.py,
the largest |d logit / dt| along the haze path (1 - t) * img + t, t in [0, epsilon],
is sampled with forward-mode autograd on CPU in float64. A sound bound satisfies
lc_precise >= sampled maximum, because the sampled maximum can only be smaller
than the true one. Exit code 1 if any row is below it.

Run:
  python experiments/check_sweep_soundness.py results/.../split_sweep_cpu_fp64.csv
  python experiments/check_sweep_soundness.py sweep.csv --out sweep_soundness.csv
"""
import argparse
import csv
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Section_5_4"))


def load_network(network_name):
    from model import FCN, FCNBig

    net = FCNBig() if network_name == "big" else FCN(int(network_name[0]))
    weights = os.path.join(ROOT, "Section_5_4", "trained", f"model_{network_name}.pth")
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    return net.eval().to(dtype=torch.float64)


def sampled_max_derivative(net, img, eps, n_samples):
    ts = torch.linspace(0.0, eps, n_samples, dtype=torch.float64)
    x = (1 - ts[:, None]) * img[None, :] + ts[:, None]
    _, d = torch.func.jvp(net, (x,), ((1 - img)[None, :].expand_as(x),))
    return d.abs().max().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv_files", nargs="+", help="CSV files written by measure_split_sweep.py")
    ap.add_argument("--samples", type=int, default=4001,
                    help="points on [0, epsilon] for the autograd maximum; default: 4001")
    ap.add_argument("--out", default=None, help="optional CSV with one row per input row")
    args = ap.parse_args()

    import torchvision
    import torchvision.transforms as transforms

    testset = torchvision.datasets.MNIST(
        root=os.path.join(ROOT, "Section_5_4", "MNIST_Data"),
        train=False, download=True, transform=transforms.ToTensor(),
    )

    nets = {}
    sampled = {}
    out_rows = []
    n_bad = 0
    for path in args.csv_files:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        worst = float("inf")
        file_bad = 0
        for row in rows:
            net_name = row["network"]
            image_id = int(row["image_id"])
            eps = float(row["epsilon"])
            key = (net_name, image_id, eps)
            if key not in sampled:
                if net_name not in nets:
                    nets[net_name] = load_network(net_name)
                img = testset[image_id][0].flatten().to(torch.float64)
                sampled[key] = sampled_max_derivative(nets[net_name], img, eps, args.samples)
            lc = float(row["lc_precise"])
            ratio = lc / sampled[key]
            sound = lc >= sampled[key]
            worst = min(worst, ratio)
            if not sound:
                file_bad += 1
                print(f"  BELOW: {net_name} image {image_id} eps_idx {row['epsilon_index']} "
                      f"n_splits {row['n_splits']}: lc={lc:.6f} sampled_max={sampled[key]:.6f}")
            out_rows.append({
                "file": os.path.basename(path), "network": net_name, "image_id": image_id,
                "epsilon_index": row["epsilon_index"], "epsilon": eps, "n_splits": row["n_splits"],
                "lc_precise": lc, "sampled_max_deriv": sampled[key], "ratio": ratio, "sound": sound,
            })
        n_bad += file_bad
        print(f"{path}: rows={len(rows)} min lc/sampled_max={worst:.6f} below={file_bad}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"Wrote {len(out_rows)} rows to {args.out}")

    print("ALL SOUND" if n_bad == 0 else f"{n_bad} ROWS BELOW THE SAMPLED MAXIMUM")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
