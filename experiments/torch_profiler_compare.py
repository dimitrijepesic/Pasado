"""Reduced torch.profiler comparison of the three precise-path variants (Phase 9).

Workload: ONE image, ONE epsilon, full 3layer precise forward (both sigmoid
layers) - not the whole benchmark. Profiled per variant:

    old      PASADO_VECTORIZED_PRECISE-off per-neuron checks + per-neuron lstsq
    vec      vectorized corner/boundary checks + per-neuron lstsq
    batched  vectorized checks + batched lstsq (only if wired into pt)

Outputs, per variant:
  - logs/torch_profiler_<variant>.txt : operator table (name, calls, self CPU,
    total CPU) sorted by self CPU, top 25, plus total op-call count
  - profiles/torch_profiler_<variant>.json : small Chrome trace
  - a compact stdout summary comparing tiny-op counts across variants.

Run on an idle machine:
    .venv/Scripts/python.exe experiments/torch_profiler_compare.py
"""
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
        download=True, transform=transforms.ToTensor())
    ci = torch.load(os.path.join(REPO, "Section_5_4", "trained",
                                 "indices_3layer.pth"), map_location="cpu")
    idx = 0
    for (image, _) in torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=False):
        if idx in ci:
            return net, image.flatten()
        idx += 1


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
        out_txt = os.path.join(REPO, "logs", f"torch_profiler_{name}.txt")
        with open(out_txt, "w") as f:
            f.write(f"variant={name}  (1 image, eps={EPS}, 3layer precise forward)\n")
            f.write(f"total operator calls: {total_calls}\n\n")
            f.write(table)
        prof.export_chrome_trace(
            os.path.join(REPO, "profiles", f"torch_profiler_{name}.json"))
        print(f"[{name}] ops={total_calls:7d}  self_cpu={total_self_us / 1e3:8.1f} ms  "
              f"-> {out_txt}")

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
