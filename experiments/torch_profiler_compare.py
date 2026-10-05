"""Compares the old, vectorized and batched precise-path variants with torch.profiler
on one image and one epsilon. Optional output directories can store operator
tables and traces.

Run: python experiments/torch_profiler_compare.py"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

torch.set_default_dtype(torch.float64)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "forward_mode_tensorized_src"))
sys.path.insert(0, os.path.join(REPO, "Section_5_4"))

import torchvision
import torchvision.transforms as transforms

from SimpleZono import *          # noqa: F401,F403
from Duals import *               # noqa: F401,F403
import precise_transformer as pt
from precise_transformer import PreciseSigmoidDualZonotope, AffineDualZonotope
from model import FCN

SEED = 12345
EPS = 2e-3  # one mid-range epsilon


def load_case():
    net = FCN(3)
    net.load_state_dict(torch.load(
        os.path.join(REPO, "Section_5_4", "trained", "model_3layer.pth"),
        map_location="cpu"))
    net.eval()
    testset = torchvision.datasets.MNIST(
        root=os.path.join(REPO, "Section_5_4", "MNIST_Data"), train=False,
        download=False, transform=transforms.ToTensor())
    ci = {
        int(index)
        for index in torch.load(os.path.join(REPO, "Section_5_4", "trained",
                                             "indices_3layer.pth"), map_location="cpu")
    }
    idx = 0
    for (image, _) in torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=False):
        if idx in ci:
            return net, image.flatten()
        idx += 1
    raise RuntimeError("no correctly classified image found")


def make_hazed(img_f, epsilon):
    ie = DualIntervalTensor(real_l=torch.tensor([0.0]),
                            real_u=torch.tensor([epsilon]))
    ie.e1_l[0] = 1
    ie.e1_u[0] = 1
    z = HyperDualIntervalToDualZonotope(ie)
    return (torch.tensor([1]) - z) * img_f + z


def forward_precise(net, x):
    x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc1.weight.T) + net.fc1.bias.data)
    x = PreciseSigmoidDualZonotope(AffineDualZonotope(x, net.fc2.weight.T) + net.fc2.bias.data)
    return AffineDualZonotope(x, net.fc_final.weight.T) + net.fc_final.bias.data


VARIANTS = {
    "old":     dict(vec=False, batched=False),
    "vec":     dict(vec=True, batched=False),
    "batched": dict(vec=True, batched=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="optional directory for text operator tables")
    ap.add_argument("--trace-dir", default=None,
                    help="optional directory for Chrome trace JSON files")
    args = ap.parse_args()
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    net, img = load_case()
    have_batched = hasattr(pt, "USE_BATCHED_LSTSQ") and hasattr(pt, "lin_reg_tensor_batched")
    summary = []
    for name, cfg in VARIANTS.items():
        if cfg["batched"] and not have_batched:
            print(f"[{name}] skipped - batched path not wired into precise_transformer")
            continue
        pt.USE_VECTORIZED_PRECISE = cfg["vec"]
        pt.USE_VECTORIZED_BOUNDARY = cfg["vec"]
        if have_batched:
            pt.USE_BATCHED_LSTSQ = cfg["batched"]

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        forward_precise(net, make_hazed(img, EPS))  # warm-up outside profiler

        np.random.seed(SEED)
        torch.manual_seed(SEED)
        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=True) as prof:
            forward_precise(net, make_hazed(img, EPS))

        ka = prof.key_averages()
        total_calls = sum(e.count for e in ka)
        total_self_us = sum(e.self_cpu_time_total for e in ka)
        tiny = {e.key: e.count for e in ka
                if e.key in ("aten::linspace", "aten::cartesian_prod", "aten::cat",
                             "aten::linalg_lstsq", "aten::sigmoid", "aten::max",
                             "aten::stack", "aten::item", "aten::ones")}
        summary.append((name, total_calls, total_self_us / 1e3, tiny))

        table = ka.table(sort_by="self_cpu_time_total", row_limit=25)
        saved = []
        if args.out_dir:
            out_txt = os.path.join(args.out_dir, f"torch_profiler_{name}.txt")
            with open(out_txt, "w") as f:
                f.write(f"variant={name}  (1 image, eps={EPS}, 3layer precise forward)\n")
                f.write(f"total operator calls: {total_calls}\n\n")
                f.write(table)
            saved.append(out_txt)
        if args.trace_dir:
            trace = os.path.join(args.trace_dir, f"torch_profiler_{name}.json")
            prof.export_chrome_trace(trace)
            saved.append(trace)
        suffix = f" -> {', '.join(saved)}" if saved else " (no files written)"
        print(f"[{name}] ops={total_calls:7d}  self_cpu={total_self_us / 1e3:8.1f} ms"
              f"{suffix}")

    print("\nper-op call counts (tiny-op dispatch check):")
    keys = sorted({k for _, _, _, t in summary for k in t})
    header = "op".ljust(24) + "".join(n.rjust(12) for n, _, _, _ in summary)
    print(header)
    for k in keys:
        row = k.ljust(24)
        for _, _, _, t in summary:
            row += str(t.get(k, 0)).rjust(12)
        print(row)


if __name__ == "__main__":
    main()
