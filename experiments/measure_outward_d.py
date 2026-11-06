"""Cost and effect of the preliminary outward rounding of D
(results/outward_rounding_prototype_2026-11/PROTOCOL.md, items 2 and 4).

  python experiments/measure_outward_d.py --device cpu --out DIR
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "compilers"))
import measure_split_sweep as mss  # noqa: E402
import batched_splits as bs  # noqa: E402
import precise_transformer as pt  # noqa: E402
import kernels_torch as kt  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]


def effect(dev, out):
    rows = []
    for net_name in ("3layer", "4layer", "5layer", "big"):
        net, images = mss.load_network_and_images(net_name, dev, torch.float64, 3)
        layers = bs.extract_affine_layers(net, net_name)
        for image_id, img in images:
            for ei in (0, 8):
                for S in (1, 64):
                    chunk = 8 if net_name == "big" else None
                    bs.USE_OUTWARD_D = False
                    off = bs.lipschitz_bound(layers, img, EPS[ei], S, seed=7, chunk=chunk)
                    bs.USE_OUTWARD_D = True
                    on = bs.lipschitz_bound(layers, img, EPS[ei], S, seed=7, chunk=chunk)
                    bs.USE_OUTWARD_D = False
                    rows.append(dict(network=net_name, image_id=image_id, eps_index=ei, n_splits=S,
                                     bound_off=off, bound_on=on, rel_increase=(on - off) / off))
        print(net_name, "largest relative increase", max(r["rel_increase"] for r in rows if r["network"] == net_name),
              flush=True)
    json.dump(rows, open(os.path.join(out, "effect.json"), "w"), indent=1)
    return rows


def cost(dev, out, rounds):
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    results = {}
    # as in results/compiled_sigmoid_v3_cuda_2026-10-29: the closed-form plane fit inside, since
    # gelsd (lstsq on the CPU) cannot be part of a full graph
    bs.USE_CLOSED_FORM_FIT = True
    for size in ("3layer_S1", "3layer_S64", "big_S8"):
        fn, inputs = kt.make("sigmoid", size, str(dev))
        compiled = torch.compile(kt.sigmoid, fullgraph=True)
        variants = {}
        for flag in (False, True):
            for mode, f in (("eager", fn), ("compiled", compiled)):
                def run(f=f, flag=flag):
                    bs.USE_OUTWARD_D = flag
                    return f(*inputs)
                t0 = time.perf_counter(); run(); sync()
                first = time.perf_counter() - t0
                for _ in range(10):
                    run()
                variants[(mode, flag)] = dict(run=run, first_call_s=first, times=[])
        keys = list(variants)
        for r in range(rounds):
            for k in keys[r % len(keys):] + keys[:r % len(keys)]:  # rotating order
                sync(); t0 = time.perf_counter(); variants[k]["run"](); sync()
                variants[k]["times"].append(time.perf_counter() - t0)
        bs.USE_OUTWARD_D = False
        res = {f"{m}_{'on' if fl else 'off'}": dict(median_ms=1e3 * statistics.median(v["times"]),
                                                    min_ms=1e3 * min(v["times"]), first_call_s=v["first_call_s"])
               for (m, fl), v in variants.items()}
        # outputs of compiled with the switch on against eager with the switch on
        bs.USE_OUTWARD_D = True
        e, c = fn(*inputs), compiled(*inputs)
        bs.USE_OUTWARD_D = False
        res["compiled_vs_eager_on_max_rel"] = max(float((a - b).abs().max() / b.abs().max()) for a, b in zip(c, e))
        results[size] = res
        print(size, json.dumps(res), flush=True)
    bs.USE_CLOSED_FORM_FIT = False
    json.dump(results, open(os.path.join(out, "cost.json"), "w"), indent=1)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-effect", action="store_true")
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    pt.USE_VECTORIZED_PRECISE = pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = True
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    np.random.seed(0)
    info = dict(device=str(dev), torch=torch.__version__, threads=torch.get_num_threads(), rounds=args.rounds,
                plane_fit_in_cost="closed form A2 (eager fit_planes) in both eager and compiled variants",
                plane_fit_in_effect="gelsd (default)")
    json.dump(info, open(os.path.join(args.out, "environment.json"), "w"), indent=1)
    if not args.skip_effect:
        effect(dev, args.out)
    cost(dev, args.out, args.rounds)


if __name__ == "__main__":
    main()
