"""Share of the CPU part (gelsd fallback and transfers) in one precise pass, for the
sequential loop and the batched path (section B of results/postfix_rerun_2026-11/PROTOCOL.md).

Every call of precise_transformer.lin_reg_tensor_batched is replaced by a copy of it that
times its steps (device-to-host copies, design matrix, CPU gelsd, host-to-device copy),
and every call of planar_fit.fit_planes (the closed form, used by --mode batched_auto from
512 boxes per call) is timed as a whole. The instrumented pass synchronizes the device
around every step, so the same configuration is also timed without instrumentation,
alternating with the instrumented passes. The instrumented replacement is checked to give
the same bound as the library function before any timing.

  python experiments/measure_cpu_share.py --device cuda --out cpu_share.csv
"""
import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_split_sweep as mss  # noqa: E402  (also puts the source dirs on sys.path)

import batched_splits  # noqa: E402
import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
MODES = ("sequential", "batched_G", "batched_auto")


class Steps:
    """Accumulates step times (ms) of the CPU bridge and of the closed form in one pass."""

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.t = dict(d2h_ms=0.0, design_ms=0.0, gelsd_ms=0.0, h2d_ms=0.0, closed_form_ms=0.0,
                      gelsd_calls=0, closed_form_calls=0, boxes=0)


def instrument(steps):
    orig_lin_reg = pt.lin_reg_tensor_batched
    orig_fit = planar_fit.fit_planes
    stamp = lambda: mss.stamp(steps.device)  # noqa: E731

    def timed_lin_reg(x_batch, zs_batch):
        dev = x_batch.device
        t0 = stamp()
        if dev.type != "cpu":
            x_batch, zs_batch = x_batch.cpu(), zs_batch.cpu()
        t1 = time.perf_counter()
        if zs_batch.dim() == 2:
            zs_batch = zs_batch.unsqueeze(-1)
        ones = torch.ones(x_batch.shape[0], x_batch.shape[1], 1,
                          dtype=x_batch.dtype, device=x_batch.device)
        xplusone = torch.cat((ones, x_batch), dim=2)
        t2 = time.perf_counter()
        R = torch.linalg.lstsq(xplusone, zs_batch, driver='gelsd').solution
        t3 = time.perf_counter()
        out = R.squeeze(-1).to(dev)
        t4 = stamp()
        s = steps.t
        s["d2h_ms"] += (t1 - t0) * 1e3
        s["design_ms"] += (t2 - t1) * 1e3
        s["gelsd_ms"] += (t3 - t2) * 1e3
        s["h2d_ms"] += (t4 - t3) * 1e3
        s["gelsd_calls"] += 1
        s["boxes"] += int(x_batch.shape[0])
        return out

    def timed_fit(grid, targets):
        t0 = stamp()
        out = orig_fit(grid, targets)
        t1 = stamp()
        steps.t["closed_form_ms"] += (t1 - t0) * 1e3
        steps.t["closed_form_calls"] += 1
        steps.t["boxes"] += int(grid.shape[0])
        return out

    def install():
        pt.lin_reg_tensor_batched = timed_lin_reg
        planar_fit.fit_planes = timed_fit

    def remove():
        pt.lin_reg_tensor_batched = orig_lin_reg
        planar_fit.fit_planes = orig_fit

    return install, remove


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--networks", nargs="+", default=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--num-images", type=int, default=3)
    ap.add_argument("--eps-index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--big-chunk", type=int, default=None,
                    help="splits per batched pass on big; default: 64 with >= 70 GiB free, else 32")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
    big_chunk = args.big_chunk
    if big_chunk is None:
        free = torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else 0
        big_chunk = 64 if free >= 70 * 2 ** 30 else 32
    steps = Steps(device)
    install, remove = instrument(steps)
    eps = EPS[args.eps_index]
    rows = []
    for network in args.networks:
        net, images = mss.load_network_and_images(network, device, dtype, args.num_images)
        forward = mss.build_forward_precise(pt, net, network)
        layers = batched_splits.extract_affine_layers(net, network)
        for mode in args.modes:
            batched_splits.USE_AUTO_PLANE_FIT = mode == "batched_auto"
            for S in args.splits:
                chunk = big_chunk if network == "big" else None

                def one_pass(img):
                    if mode == "sequential":
                        return mss.eval_splits_once(forward, img, eps, S, args.seed, device, dtype)
                    return batched_splits.lipschitz_bound(layers, img, eps, S, args.seed, chunk)

                # the instrumented replacement must not change the bound
                ref = one_pass(images[0][1])
                install()
                try:
                    got = one_pass(images[0][1])
                finally:
                    remove()
                if got != ref:
                    raise SystemExit(f"instrumented bound differs: {network} {mode} S={S}: {got} vs {ref}")
                for _ in range(args.warmup):
                    one_pass(images[0][1])
                for image_id, img in images:
                    for rep in range(args.repeats):
                        t0 = mss.stamp(device)
                        bound = one_pass(img)
                        plain_ms = (mss.stamp(device) - t0) * 1e3
                        steps.reset()
                        install()
                        try:
                            t0 = mss.stamp(device)
                            one_pass(img)
                            instr_ms = (mss.stamp(device) - t0) * 1e3
                        finally:
                            remove()
                        s = dict(steps.t)
                        cpu_ms = s["d2h_ms"] + s["design_ms"] + s["gelsd_ms"] + s["h2d_ms"]
                        rows.append(dict(network=network, mode=mode, n_splits=S,
                                         chunk=chunk or S, image_id=image_id, repeat=rep,
                                         eps_index=args.eps_index, bound=bound,
                                         pass_ms_uninstrumented=plain_ms,
                                         pass_ms_instrumented=instr_ms, cpu_part_ms=cpu_ms,
                                         **s))
                sub = [r for r in rows if r["network"] == network and r["mode"] == mode
                       and r["n_splits"] == S]
                med = lambda k: statistics.median(r[k] for r in sub)  # noqa: E731
                print(f"{network:6} {mode:12} S={S:3}  pass {med('pass_ms_uninstrumented'):9.2f} ms "
                      f"(instr. {med('pass_ms_instrumented'):9.2f})  cpu part {med('cpu_part_ms'):8.2f} ms "
                      f"= {100 * med('cpu_part_ms') / med('pass_ms_instrumented'):5.1f}% of instr. pass  "
                      f"[gelsd {med('gelsd_ms'):.2f}, copies {med('d2h_ms') + med('h2d_ms'):.2f}, "
                      f"closed form {med('closed_form_ms'):.2f}]", flush=True)
                with open(args.out, "w", newline="") as handle:  # rewritten after every cell
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
    batched_splits.USE_AUTO_PLANE_FIT = False


if __name__ == "__main__":
    np.random.seed(0)
    main()
