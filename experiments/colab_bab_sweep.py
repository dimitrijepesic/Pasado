"""auto_LiRPA + Branch-and-Bound swept over networks x epsilons x budgets, on GPU.

Establishes where the equal-budget crossover with Pasado sits across the whole
benchmark, not just at the one point measured so far (5layer, eps=0.2, crossover
at ~7 subproblems). That point was the most favourable one for Pasado -- it is
where the single-pass gap peaked -- so the crossover elsewhere is expected to be
at a smaller budget, and this sweep is what shows it.

Only the auto_LiRPA side runs here. The Pasado side is a CPU job
(`get_lipschitz.py --n-splits k`, cheap) and is merged afterwards; keeping them
apart avoids running Pasado's float64 CPU code inside a CUDA-default process.

Design notes that matter:

* **Best-first bisection.** t in [0, eps] is the whole perturbed domain (a
  scalar), so bisecting it is complete branching; there are no ReLU neurons to
  split in a sigmoid network. At each step the sub-interval with the largest
  current bound is split, since only that one can hold the global max. The
  running max over live sub-intervals is sound at every budget, so one pass
  yields the whole curve.

* **Fresh module per solve.** auto_LiRPA caches intermediate bounds and alpha
  parameters per call and never frees them; the bound graph is a reference
  cycle, so `gc.collect()` is required and `empty_cache()` alone is a no-op.
  Without this an 80 GB A100 fills after six solves.

* **Resumable.** Colab sessions die. Every (network, epsilon) result is appended
  to the CSV as soon as it completes, and a rerun skips pairs already present,
  so progress survives a disconnect.

Colab:
    !git clone https://github.com/uiuc-arc/Pasado.git
    !mkdir -p Pasado/experiments
    !pip install -q --ignore-requires-python git+https://github.com/Verified-Intelligence/auto_LiRPA.git
    !wget -q -O Pasado/experiments/sweep.py \\
      https://raw.githubusercontent.com/dimitrijepesic/Pasado/colab-gpu-wip/experiments/colab_bab_sweep.py
    !python Pasado/experiments/sweep.py --networks 3layer 4layer 5layer --eps-indices 0 1 2 3 4 5 6
"""
import argparse
import csv
import gc
import os
import time

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

EPSILONS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def find_section_5_4():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "..", "Section_5_4"),
                 os.path.join(os.getcwd(), "Pasado", "Section_5_4"),
                 os.path.join(os.getcwd(), "Section_5_4")):
        if os.path.isdir(cand):
            return os.path.normpath(cand)
    raise FileNotFoundError("Section_5_4 not found; clone uiuc-arc/Pasado first")


SEC = find_section_5_4()


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
            os.path.join(SEC, "trained", f"model_{name}.pth"), map_location="cpu"))
    else:
        depth = {"3layer": 3, "4layer": 4, "5layer": 5}[name]
        mods = [nn.Flatten(), nn.Linear(784, 100), nn.Sigmoid(),
                nn.Linear(100, 100), nn.Sigmoid()]
        if depth >= 4:
            mods += [nn.Linear(100, 100), nn.Sigmoid()]
        if depth == 5:
            mods += [nn.Linear(100, 100), nn.Sigmoid()]
        mods += [nn.Linear(100, 10)]
        net = nn.Sequential(*mods)
        state = torch.load(os.path.join(SEC, "trained", f"model_{name}.pth"),
                           map_location="cpu")
        lin_idx = [i for i, m in enumerate(net) if isinstance(m, nn.Linear)]
        fc = ["fc1", "fc2", "fc3", "fc4"][:len(lin_idx) - 1] + ["fc_final"]
        net.load_state_dict({f"{i}.{p}": state[f"{f_}.{p}"]
                             for i, f_ in zip(lin_idx, fc)
                             for p in ("weight", "bias")})
    return net.to(torch.float64).eval()


def load_test_images():
    import torchvision
    import torchvision.transforms as T
    return torchvision.datasets.MNIST(
        root=os.path.join(SEC, "MNIST_Data"), train=False, download=True,
        transform=T.ToTensor())


class HazeJacobian(nn.Module):
    """dF/dt for x(t) = img + t*(1 - img) -- exactly Pasado's Section 5.4 quantity."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, img):
        from auto_LiRPA.jacobian import JacobianOP
        return JacobianOP.apply(self.net(img + t * (1.0 - img)), t)


def solve_interval(network, img, t_lo, t_hi, dev):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    center, radius = (t_lo + t_hi) / 2.0, (t_hi - t_lo) / 2.0
    # Rebuilding the net too, not just the BoundedModule: tracing attaches bound
    # state to the traced modules, so a shared net keeps old graphs reachable.
    model = BoundedModule(HazeJacobian(build_network(network).to(dev)),
                          (torch.zeros(1, 1, device=dev), img), device=dev)
    bt = BoundedTensor(torch.full((1, 1), center, device=dev),
                       PerturbationLpNorm(norm=float("inf"), eps=radius))
    lb, ub = model.compute_jacobian_bounds((bt, img))
    out = torch.maximum(lb.abs(), ub.abs()).max().item()
    del model, lb, ub
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def bab_curve(network, img, eps, budget, dev):
    """{solves_used: bound} for best-first bisection of t in [0, eps]."""
    b0 = solve_interval(network, img, 0.0, eps, dev)
    domains = [(b0, 0.0, eps)]
    curve, solves = {1: b0}, 1
    while solves + 2 <= budget:
        i = max(range(len(domains)), key=lambda k: domains[k][0])
        _, lo, hi = domains.pop(i)
        mid = (lo + hi) / 2.0
        domains.append((solve_interval(network, img, lo, mid, dev), lo, mid))
        domains.append((solve_interval(network, img, mid, hi, dev), mid, hi))
        solves += 2
        curve[solves] = max(d[0] for d in domains)
    return curve


FIELDS = ["network", "eps_index", "epsilon", "solves", "images_averaged",
          "lirpa_bab", "pasado_precise_1_solve", "seconds_per_image"]


def load_done(path):
    """(network, eps_index) pairs already finished, so a rerun resumes."""
    if not os.path.exists(path):
        return set(), []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["network"], int(r["eps_index"])) for r in rows}, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=["3layer", "4layer", "5layer"],
                    choices=["3layer", "4layer", "5layer", "big"])
    # Indices 7..15 are omitted by default: there every method has converged and
    # the ratio is flat at ~1, so they cost time without carrying information.
    ap.add_argument("--eps-indices", nargs="+", type=int,
                    default=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--budget", type=int, default=9,
                    help="max solves per image; schedule is 1,3,5,...")
    ap.add_argument("--num-images", type=int, default=30)
    ap.add_argument("--out", default="bab_sweep.csv")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.num_images != 30:
        print(f"\n*** WARNING: --num-images {args.num_images} != 30. Pasado's "
              f"reference numbers are 30-image averages, so ratios computed "
              f"against them will not be comparable. ***\n", flush=True)

    dev = args.device
    print(f"device={dev}" + (f" ({torch.cuda.get_device_name(0)})"
                             if dev == "cuda" else ""))
    ds = load_test_images()
    done, rows = load_done(args.out)
    if done:
        print(f"resuming: {len(done)} (network, eps) pairs already in {args.out}")

    schedule = list(range(1, args.budget + 1, 2))

    for network in args.networks:
        correct = set(int(i) for i in torch.load(
            os.path.join(SEC, "trained", f"indices_{network}.pth"),
            map_location="cpu"))
        selected = [i for i in range(len(ds)) if i in correct][:args.num_images]
        ref_path = os.path.join(SEC, "results", f"lc_precise_{network}.pth")
        ref = torch.load(ref_path, map_location="cpu") if os.path.exists(ref_path) else None

        for ei in args.eps_indices:
            if (network, ei) in done:
                print(f"skip {network} eps_index={ei} (already done)")
                continue
            eps = EPSILONS[ei]
            sums = {k: 0.0 for k in schedule}
            begin = time.perf_counter()

            for img_no, idx in enumerate(selected, start=1):
                img = ds[idx][0].flatten().unsqueeze(0).to(torch.float64).to(dev)
                curve = bab_curve(network, img, eps, args.budget, dev)
                for k in schedule:
                    sums[k] += curve[k]
                if img_no % 10 == 0:
                    print(f"  {network} eps={eps:.5g}: {img_no}/{len(selected)} "
                          f"images, K=1 {sums[1]/img_no:.4g} -> "
                          f"K={schedule[-1]} {sums[schedule[-1]]/img_no:.4g}",
                          flush=True)

            secs = (time.perf_counter() - begin) / len(selected)
            pas = ref[ei] if ref else None
            new = [{
                "network": network, "eps_index": ei, "epsilon": eps,
                "solves": k, "images_averaged": len(selected),
                "lirpa_bab": sums[k] / len(selected),
                "pasado_precise_1_solve": pas,
                "seconds_per_image": secs,
            } for k in schedule]
            rows.extend(new)
            # Appended as soon as the pair finishes, so a dead session costs at
            # most the pair in flight.
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

            summary = "  ".join(f"K={r['solves']}:{r['lirpa_bab']:.4g}" for r in new)
            print(f"[{network} eps={eps:.5g}] {secs:.1f}s/img  {summary}", flush=True)

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
