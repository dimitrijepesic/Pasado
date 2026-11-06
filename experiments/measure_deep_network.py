"""Bounds of big and deep8 on the same images (results/deep_network_2026-11/PROTOCOL.md).

  python experiments/measure_deep_network.py --deep-dir DIR --out DIR/bounds.csv [--device cuda]
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_split_sweep as mss  # noqa: E402
import batched_splits as bs  # noqa: E402
import precise_transformer as pt  # noqa: E402
from check_sweep_soundness import sampled_max_derivative  # noqa: E402
from train_deep_network import deep8  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
SEC = os.path.join(os.path.dirname(HERE), "Section_5_4")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deep-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--eps-indices", type=int, nargs="+", default=[0, 4, 8, 12, 15])
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="local test only: skip the 90% accuracy rule")
    args = ap.parse_args()
    training = json.load(open(os.path.join(args.deep_dir, "training.json")))
    if training["test_accuracy"] < 0.9 and not args.smoke:
        raise SystemExit(f"deep8 test accuracy {training['test_accuracy']:.4f} < 0.9: not used (PROTOCOL.md)")
    torch.set_default_dtype(torch.float64)
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
    bs.USE_AUTO_PLANE_FIT = True
    dev = torch.device(args.device)
    nets = {}
    big, _ = mss.load_network_and_images("big", torch.device("cpu"), torch.float64, 1)
    d8 = deep8()
    d8.load_state_dict(torch.load(os.path.join(args.deep_dir, "model_deep8.pth"), map_location="cpu"))
    for name, net in (("big", big), ("deep8", d8.to(torch.float64).eval())):
        lin = [m for m in net if isinstance(m, nn.Linear)]
        nets[name] = (net, [(m.weight.detach().T.to(dev), m.bias.detach().to(dev)) for m in lin])
    big_ok = {int(i) for i in torch.load(os.path.join(SEC, "trained", "indices_big.pth"), map_location="cpu")}
    deep_ok = {int(i) for i in torch.load(os.path.join(args.deep_dir, "indices_deep8.pth"), map_location="cpu")}
    import torchvision
    import torchvision.transforms as transforms
    testset = torchvision.datasets.MNIST(root=os.path.join(SEC, "MNIST_Data"), train=False, download=True,
                                         transform=transforms.ToTensor())
    ids = [i for i in range(len(testset)) if i in big_ok and i in deep_ok][:args.num_images]
    rows = []
    for name, (net, layers) in nets.items():
        for i in ids:
            img = testset[i][0].flatten().to(torch.float64)
            for ei in args.eps_indices:
                sampled = sampled_max_derivative(net, img, EPS[ei], 4001)
                for S in args.splits:
                    bs.lipschitz_bound(layers, img.to(dev), EPS[ei], S, 7, args.chunk)  # warm-up
                    times = []
                    for _ in range(3):
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        b = bs.lipschitz_bound(layers, img.to(dev), EPS[ei], S, 7, args.chunk)
                        times.append(time.perf_counter() - t0)
                    rows.append(dict(network=name, image_id=i, eps_index=ei, n_splits=S, bound=b, sampled_max=sampled,
                                     looseness=b / sampled, sound=b >= sampled, time_s=statistics.median(times)))
            with open(args.out, "w", newline="") as handle:
                w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(name, "done", flush=True)
    for ei in args.eps_indices:
        for S in args.splits:
            sel = lambda n: [r for r in rows if r["network"] == n and r["eps_index"] == ei and r["n_splits"] == S]  # noqa: E731
            b, d = sel("big"), sel("deep8")
            mb, md = statistics.mean(r["bound"] for r in b), statistics.mean(r["bound"] for r in d)
            lb, ld = statistics.mean(r["looseness"] for r in b), statistics.mean(r["looseness"] for r in d)
            print(f"eps[{ei}] S={S}: bound big {mb:.4g} deep8 {md:.4g} (x{md / mb:.2f}); looseness big {lb:.3f} "
                  f"deep8 {ld:.3f} (x{ld / lb:.2f}); all sound: {all(r['sound'] for r in b + d)}")


if __name__ == "__main__":
    main()
