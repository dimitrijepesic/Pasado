"""Compares Pasado's bound on dF/dt with auto_LiRPA's Jacobian bound for the haze
family x(t) = img + t*(1-img), on 30 correctly classified test images.
Needs a GPU for the big network (auto_LiRPA uses about 10 GB there). Only the
tightness is compared, not the runtime.

Colab: python Pasado/experiments/colab_compare_pasado_lirpa.py --network big"""
import argparse
import csv
import gc
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="big",
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None,
                    help="output CSV path; default: results_<network>_colab.csv")
    args = ap.parse_args()

    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

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

    # The saved Pasado numbers are averages over exactly 30 images, so only
    # --num-images 30 gives a comparable ratio (image 0 alone is 245.2 on big,
    # well above the 30-image average of 168.1).
    if args.num_images != 30:
        print(f"\n*** WARNING: --num-images {args.num_images} != 30. The saved "
              f"Pasado values are 30-image averages, so the printed ratios are "
              f"NOT apples-to-apples and must not be reported. Use 30. ***\n",
              flush=True)

    def saved(kind):
        p = os.path.join(SEC, "results", f"{kind}_{args.network}.pth")
        return torch.load(p, map_location="cpu") if os.path.exists(p) else None
    ref = {k: saved(k) for k in
           ("lc_intervals", "lc_zonos", "lc_precise", "time_precise")}

    out_path = os.path.abspath(
        args.out or os.path.join(os.getcwd(), f"results_{args.network}_colab.csv")
    )
    lc_sums = [0.0] * len(EPSILONS)
    t_sums = [0.0] * len(EPSILONS)
    counts = [0] * len(EPSILONS)

    for img_no, idx in enumerate(selected, start=1):
        img = ds[idx][0].flatten().unsqueeze(0).to(torch.float64).to(dev)
        for ei, eps in enumerate(EPSILONS):
            # Fresh network and BoundedModule for every (image, epsilon). auto_LiRPA keeps
            # the bound graph of each call alive (about 10 GB per call on big), so reusing
            # one module ran out of memory.
            model = BoundedModule(HazeJacobian(build_network(args.network).to(dev)),
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
            # gc.collect() first, then empty_cache(): the bound graph is a reference cycle,
            # so only the garbage collector frees it.
            del model, lb, ub
            gc.collect()
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
