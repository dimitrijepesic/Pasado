"""Colab/GPU port of the Pasado-vs-auto_LiRPA local-Lipschitz comparison.

Why this exists as a separate script. The comparison runs fine locally for the
100-wide networks (3/4/5layer), but not for FFNNBig (1024-wide): auto_LiRPA's
Jacobian path calls _expand_jacobian, which explicitly turns OFF every sparse
option --

    # auto_LiRPA/jacobian.py, _expand_jacobian
    if self.jacobian_start_nodes:
        # Disable unstable options
        self.bound_opts.update({
            'sparse_intermediate_bounds': False,
            'sparse_conv_intermediate_bounds': False,
            'sparse_intermediate_bounds_with_ibp': False,
            'sparse_features_alpha': False,
            'sparse_spec_alpha': False,
        })

-- so intermediate bounds are computed against dense 1024x1024 identity specs
(the library itself warns: "Creating an identity matrix with size 1024x1024 ...
This may indicate poor performance"). One solve peaks near 10.7 GB, and the
memory is never returned to the OS, so a 15.7 GB laptop cannot do two solves in
one process and barely does one. An A100 (80 GB) has room.

What is compared. Pasado Section 5.4 bounds dF/dt for the haze family
x(t) = img + t*(1-img), t in [0, eps], reporting
max_j max(|lb_j|, |ub_j|) averaged over 30 correctly-classified test images.
We express the same haze map inside the model and ask auto_LiRPA for the
Jacobian w.r.t. the scalar t, which is exactly dF/dt -- same quantity, no
conversion factor.

Fairness notes: auto_LiRPA runs its default optimize=True (CROWN-Optimized),
its strongest standard mode; float64 throughout, matching Pasado. Runtime is
NOT comparable here (their GPU vs Pasado's CPU) -- tightness is the metric.

Colab usage:
    !git clone https://github.com/uiuc-arc/Pasado.git
    !pip install git+https://github.com/Verified-Intelligence/auto_LiRPA.git
    !python Pasado/experiments/colab_compare_pasado_lirpa.py --network big
"""
import argparse
import csv
import os
import time

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def find_pasado_root():
    """Locate Section_5_4 whether we're run from inside the repo or beside it."""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "Section_5_4"),
                 os.path.join(os.getcwd(), "Pasado", "Section_5_4"),
                 os.path.join(os.getcwd(), "Section_5_4")):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    raise FileNotFoundError("Section_5_4 not found; clone uiuc-arc/Pasado first")


SEC = find_pasado_root()


def build_network(name):
    if name == "big":
        net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024), nn.Sigmoid(),
            nn.Linear(1024, 1024), nn.Sigmoid(),
            nn.Linear(1024, 1024), nn.Sigmoid(),
            nn.Linear(1024, 1024), nn.Sigmoid(),
            nn.Linear(1024, 10),
        )
        net.load_state_dict(torch.load(
            os.path.join(SEC, "trained", f"model_{name}.pth"),
            map_location="cpu"))
    else:
        layers = {"3layer": 3, "4layer": 4, "5layer": 5}[name]
        mods = [nn.Flatten(), nn.Linear(784, 100), nn.Sigmoid(),
                nn.Linear(100, 100), nn.Sigmoid()]
        if layers >= 4:
            mods += [nn.Linear(100, 100), nn.Sigmoid()]
        if layers == 5:
            mods += [nn.Linear(100, 100), nn.Sigmoid()]
        mods += [nn.Linear(100, 10)]
        net = nn.Sequential(*mods)
        state = torch.load(os.path.join(SEC, "trained", f"model_{name}.pth"),
                           map_location="cpu")
        seq_linears = [i for i, m in enumerate(net) if isinstance(m, nn.Linear)]
        fc_names = ["fc1", "fc2", "fc3", "fc4"][:len(seq_linears) - 1] + ["fc_final"]
        net.load_state_dict({
            f"{i}.{p}": state[f"{fc}.{p}"]
            for i, fc in zip(seq_linears, fc_names) for p in ("weight", "bias")})
    return net.to(torch.float64).eval()


def load_mnist_test():
    """torchvision on Colab; ToTensor() == raw uint8 / 255, same as Pasado."""
    import torchvision
    import torchvision.transforms as T
    ds = torchvision.datasets.MNIST(
        root=os.path.join(SEC, "MNIST_Data"), train=False, download=True,
        transform=T.ToTensor())
    return ds


class HazeJacobian(nn.Module):
    """t is the perturbed scalar; img rides along unperturbed."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, img):
        from auto_LiRPA.jacobian import JacobianOP
        x = img + t * (1.0 - img)
        return JacobianOP.apply(self.net(x), t)


def main():
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="big",
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    print(f"device={dev}"
          + (f" ({torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB)"
             if dev == "cuda" else ""))

    net = build_network(args.network).to(dev)
    ds = load_mnist_test()
    correct = set(int(i) for i in torch.load(
        os.path.join(SEC, "trained", f"indices_{args.network}.pth"),
        map_location="cpu"))
    selected = [i for i in range(len(ds)) if i in correct][:args.num_images]

    def saved(kind):
        p = os.path.join(SEC, "results", f"{kind}_{args.network}.pth")
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None
    ref = {k: saved(k) for k in
           ("lc_intervals", "lc_zonos", "lc_precise", "time_precise")}

    out_path = os.path.join(os.getcwd(), f"results_{args.network}_colab.csv")
    lc_sums = [0.0] * len(EPSILONS)
    t_sums = [0.0] * len(EPSILONS)
    counts = [0] * len(EPSILONS)

    for img_no, idx in enumerate(selected, start=1):
        img = ds[idx][0].flatten().unsqueeze(0).to(torch.float64).to(dev)
        for ei, eps in enumerate(EPSILONS):
            # Fresh module per (image, epsilon). BoundedModule accumulates
            # cached intermediate bounds and alpha parameters on every
            # compute_jacobian_bounds call and never releases them; with the
            # sparse options force-disabled on the Jacobian path (see the
            # module docstring) that is ~10 GB per call on `big`. Reusing one
            # module across epsilons is what OOM-killed the 16 GB local runs,
            # and 16 accumulated calls would threaten even an 80 GB A100.
            model = BoundedModule(HazeJacobian(net),
                                  (torch.zeros(1, 1, device=dev), img),
                                  device=dev)
            t0 = torch.full((1, 1), eps / 2, device=dev)
            bt = BoundedTensor(t0, PerturbationLpNorm(
                norm=float("inf"), eps=eps / 2))
            if dev == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            lb, ub = model.compute_jacobian_bounds((bt, img))
            if dev == "cuda":
                torch.cuda.synchronize()
            t_sums[ei] += time.perf_counter() - begin
            lc_sums[ei] += torch.maximum(lb.abs(), ub.abs()).max().item()
            counts[ei] += 1
            del model, lb, ub
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(f"  img {img_no} eps={eps:.6f} "
                  f"lirpa={lc_sums[ei]/counts[ei]:.6g} "
                  f"({t_sums[ei]/counts[ei]:.1f}s avg)", flush=True)

        rows = [{
            "epsilon": e,
            "images_averaged": counts[k],
            "pasado_interval": ref["lc_intervals"][k] if ref["lc_intervals"] else None,
            "pasado_zono": ref["lc_zonos"][k] if ref["lc_zonos"] else None,
            "pasado_precise": ref["lc_precise"][k] if ref["lc_precise"] else None,
            "lirpa_opt": lc_sums[k] / counts[k],
            "pasado_time_precise": ref["time_precise"][k] if ref["time_precise"] else None,
            "lirpa_time": t_sums[k] / counts[k],
        } for k, e in enumerate(EPSILONS)]
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        print(f"--- image {img_no}/{len(selected)} ---", flush=True)
        for r in rows:
            pp = r["pasado_precise"]
            ratio = (r["lirpa_opt"] / pp) if pp else float("nan")
            print(f"  eps={r['epsilon']:.6f}  precise={pp:.6g}  "
                  f"lirpa={r['lirpa_opt']:.6g}  ratio={ratio:.3f}  "
                  f"t={r['lirpa_time']:.2f}s", flush=True)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
