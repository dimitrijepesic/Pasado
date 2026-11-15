"""Speedup ladder: every optimization step from the original Pasado code to the compiled
batched pass, timed and profiled on ONE machine in ONE process.

Steps (each adds one decision to the previous one):

  0 original          CPU   all selectors off, complex64 cubic, loop over the splits
  1 vectorized        CPU   + vectorized corner and boundary checks
  2 batched_lstsq     CPU   + one lstsq per layer and the batched sampling grid
  3 real_cubic        CPU   + real (Viete) cubic solver
  4 gpu_hybrid        GPU   the same code on the device, gelsd on the CPU, loop over the splits
  5 batched_G         GPU   all splits in one pass (batched_splits.py), gelsd on the CPU
  6 a2_eager          GPU   + closed-form plane fit planar_fit.fit_planes (PASADO_CLOSED_FORM_FIT)
  7 a2_compiled_fit   GPU   + fit_planes compiled by TorchInductor (PASADO_COMPILED_FIT)
  8 compiled_sigmoid  GPU   the whole precise_sigmoid compiled (fullgraph, eager fit inside)

"original" is the current code with every selector off, i.e. the upstream code plus the two
soundness fixes of b526762 (one abs per boundary check, a sign in __neg__).
"GPU" is --device. With --device cpu steps 5-8 run on the CPU and step 4 is skipped (it
would be step 3 again), which gives a CPU-only ladder and a local test.

Protocol. float64, seed 7. For every (network, n_splits) cell the steps are interleaved:
for every (image, epsilon) and repeat, each step runs one pass, in rotating order, so a
drift of the machine does not land on one step. The device is synchronized at every timer
boundary. The first pass of a step in a cell (the probe) is never timed; it carries the
compilation. A step whose passes are too long for --step-budget-s gets fewer repeats, then
fewer (image, epsilon) pairs; the plan actually used is in the output. Nothing is
extrapolated: speedups are ratios over the (image, epsilon) pairs both steps measured.

Per step and cell: time per pass, speedup over step 0 and over the previous step, the
bound and its largest relative difference from step 0 and from the previous step, the
smallest bound / sampled autograd maximum of |d logit / dt| (soundness), peak VRAM, the
first-pass time, and a profile:
  * stage breakdown of one instrumented pass (affine, sampling grid, plane fit, error
    checks, rest of the sigmoid, rest of the pass); the device is synchronized around
    every stage, so this pass is slower than the timed ones and is reported separately;
  * cProfile (top functions by cumulative and by own time);
  * torch.profiler (operator table; on CUDA also the number of kernels launched).

Output in --out: environment.json, cells/<network>_S<n>.json (a finished cell is skipped
on a re-run), profiles/<network>_S<n>_<step>.txt, passes.csv, summary.csv, summary.md.

  python experiments/measure_speedup_ladder.py --device cuda --out results/ladder_a100
  python experiments/measure_speedup_ladder.py --device cpu --networks 3layer --splits 1 4 \
      --num-images 1 --repeats 2 --out /tmp/ladder_test
  python experiments/measure_speedup_ladder.py --report-only --out results/ladder_a100
"""
import argparse
import cProfile
import csv
import io
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import measure_split_sweep as mss  # noqa: E402  (also puts the source dirs on sys.path)
import check_sweep_soundness as css  # noqa: E402

import batched_splits as bs  # noqa: E402
import planar_fit  # noqa: E402,F401
import precise_transformer as pt  # noqa: E402

EPS = [10 ** (-k / 4) * 2 for k in range(2, 18)]
DTYPE = torch.float64
ORIG_PRECISE_SIGMOID = bs.precise_sigmoid


def _step(i, name, on_cpu, path, pt_flags, closed=False, cfit=False, csig=False, text=""):
    return dict(id=i, name=name, on_cpu=on_cpu, path=path, pt=pt_flags, closed_form=closed,
                compiled_fit=cfit, compiled_sigmoid=csig, text=text)


#                                             vec bnd lstsq grid real
STEPS = [
    _step(0, "original", True, "sequential", (0, 0, 0, 0, 0),
          text="all selectors off, complex64 cubic, loop over the splits"),
    _step(1, "vectorized", True, "sequential", (1, 1, 0, 0, 0),
          text="+ vectorized corner and boundary checks"),
    _step(2, "batched_lstsq", True, "sequential", (1, 1, 1, 1, 0),
          text="+ one lstsq per layer, batched sampling grid"),
    _step(3, "real_cubic", True, "sequential", (1, 1, 1, 1, 1),
          text="+ real (Viete) cubic solver"),
    _step(4, "gpu_hybrid", False, "sequential", (1, 1, 1, 1, 1),
          text="same code on the device, gelsd on the CPU, loop over the splits"),
    _step(5, "batched_G", False, "batched", (1, 1, 1, 1, 1),
          text="all splits in one pass, gelsd on the CPU"),
    _step(6, "a2_eager", False, "batched", (1, 1, 1, 1, 1), closed=True,
          text="+ closed-form plane fit on the device"),
    _step(7, "a2_compiled_fit", False, "batched", (1, 1, 1, 1, 1), closed=True, cfit=True,
          text="+ plane fit compiled by TorchInductor"),
    _step(8, "compiled_sigmoid", False, "batched", (1, 1, 1, 1, 1), closed=True, csig=True,
          text="whole precise_sigmoid compiled (fullgraph)"),
]
STEP_BY_ID = {s["id"]: s for s in STEPS}


# ----------------------------------------------------------------------------- setup

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda", help="device of steps 4-8; default: cuda")
    ap.add_argument("--networks", nargs="+", default=["3layer", "4layer", "5layer", "big"],
                    choices=["3layer", "4layer", "5layer", "big"])
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--steps", type=int, nargs="+", default=[s["id"] for s in STEPS],
                    choices=[s["id"] for s in STEPS], help="step ids to run; default: all")
    ap.add_argument("--num-images", type=int, default=3)
    ap.add_argument("--eps-indices", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed passes per step and cell, the probe included; default: 2")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step-budget-s", type=float, default=180.0,
                    help="time one step may take in one cell before its repeats, then its "
                         "(image, epsilon) pairs, are reduced; default: 180")
    ap.add_argument("--profile-max-s", type=float, default=60.0,
                    help="no profile passes for a step whose pass is longer; default: 60")
    ap.add_argument("--no-profile", action="store_true", help="timing only")
    ap.add_argument("--big-chunk", type=int, default=None,
                    help="splits per batched pass on big; default: 64 on CUDA with >= 70 GiB "
                         "free, 32 on other CUDA devices, 8 on the CPU")
    ap.add_argument("--samples", type=int, default=4001,
                    help="points of the sampled autograd maximum; default: 4001")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads; default: torch's")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild passes.csv and the summaries from the cell files")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def git_commit():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=ROOT, text=True).strip()
        return head + (" (dirty)" if dirty else "")
    except Exception:
        return "unknown"


def environment(args, device):
    env = dict(
        date_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), commit=git_commit(),
        python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        os=platform.platform(), cpu=cpu_model(), cpu_count=os.cpu_count(),
        torch_threads=torch.get_num_threads(), device=str(device), dtype="float64",
        seed=args.seed, networks=args.networks, splits=args.splits, steps=args.steps,
        num_images=args.num_images, eps_indices=args.eps_indices, warmup=args.warmup,
        repeats=args.repeats, step_budget_s=args.step_budget_s, samples=args.samples,
        baseline="step 0 = current code with all selectors off (upstream code + b526762)")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        env.update(gpu=torch.cuda.get_device_name(device), vram_total_gib=total / 2 ** 30,
                   vram_free_gib=free / 2 ** 30, cuda=torch.version.cuda)
    return env


def default_big_chunk(device):
    if device.type == "cuda":
        return 64 if torch.cuda.mem_get_info(device)[0] >= 70 * 2 ** 30 else 32
    return 8


# ----------------------------------------------------------------------------- steps

def make_compiled_sigmoid():
    """The library's precise_sigmoid as one TorchInductor graph, one callable per cell.

    As in results/compiled_sigmoid_v3_cuda_2026-10-29: fullgraph, the eager fit_planes
    inside (no nested compile), and the generator dimension marked dynamic before every
    call, since it grows by 2N per layer and one graph has to serve every layer.
    """
    def core(rc, rg, dc, dg, noise):
        out, ok = ORIG_PRECISE_SIGMOID(bs.BatchedDualZonotope(rc, rg, dc, dg), noise)
        return out.rc, out.rg, out.dc, out.dg, ok

    compiled = torch.compile(core, fullgraph=True)

    def precise_sigmoid(z, noise):
        torch._dynamo.mark_dynamic(z.rg, 1)
        torch._dynamo.mark_dynamic(z.dg, 1)
        rc, rg, dc, dg, ok = compiled(z.rc, z.rg, z.dc, z.dg, noise)
        return bs.BatchedDualZonotope(rc, rg, dc, dg), ok

    return precise_sigmoid


def configure(step, compiled_sigmoid):
    """Every switch is set for every step, so no step inherits a value from another."""
    (pt.USE_VECTORIZED_PRECISE, pt.USE_VECTORIZED_BOUNDARY, pt.USE_BATCHED_LSTSQ,
     pt.USE_BATCHED_GRID, pt.USE_REAL_CUBIC) = (bool(v) for v in step["pt"])
    bs.USE_CLOSED_FORM_FIT = step["closed_form"]
    bs.USE_COMPILED_FIT = step["compiled_fit"]
    bs.USE_AUTO_PLANE_FIT = False
    bs.USE_OUTWARD_D = False
    bs.precise_sigmoid = compiled_sigmoid if step["compiled_sigmoid"] else ORIG_PRECISE_SIGMOID


def restore_library():
    configure(STEP_BY_ID[3], None)
    pt.USE_BATCHED_LSTSQ = pt.USE_BATCHED_GRID = pt.USE_REAL_CUBIC = False  # library defaults


class Side:
    """A network and its images on one device, with both ways to run a pass."""

    def __init__(self, network, device, num_images, seed, chunk):
        self.device = device
        self.net, self.images = mss.load_network_and_images(network, device, DTYPE, num_images)
        self.forward = mss.build_forward_precise(pt, self.net, network)
        self.layers = bs.extract_affine_layers(self.net, network)
        self.seed, self.chunk = seed, chunk

    def run(self, step, img, eps, n_splits):
        if step["path"] == "sequential":
            return mss.eval_splits_once(self.forward, img, eps, n_splits, self.seed, self.device, DTYPE)
        return bs.lipschitz_bound(self.layers, img, eps, n_splits, self.seed, self.chunk)


# ----------------------------------------------------------------------------- profile

class Stages:
    """Wall time per stage of one pass, the device synchronized around every stage."""

    SEQUENTIAL = [(pt, "AffineDualZonotope", "affine"), (pt, "PreciseSigmoidDualZonotope", "sigmoid"),
                  (pt, "get_linspace", "grid"), (pt, "get_linspace_batched", "grid"),
                  (pt, "lin_reg_tensor", "fit"), (pt, "lin_reg_tensor_batched", "fit"),
                  (pt, "compute_max_error", "error_check")]
    BATCHED = [(bs, "affine", "affine"), (bs, "precise_sigmoid", "sigmoid"),
               (pt, "get_linspace_batched", "grid"), (pt, "check_corners_tensor", "error_check"),
               (pt, "check_nonlinear_boundary_tensor", "error_check")]
    # Inside a compiled graph nothing can be timed: only the two outer stages.
    COMPILED = [(bs, "affine", "affine"), (bs, "precise_sigmoid", "sigmoid")]

    def __init__(self, device):
        self.device = device
        self.ms, self.calls, self.active, self.saved = {}, {}, set(), []

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            if name in self.active:  # a stage called from itself counts once
                return fn(*args, **kwargs)
            self.active.add(name)
            t0 = mss.stamp(self.device)
            try:
                return fn(*args, **kwargs)
            finally:
                self.ms[name] = self.ms.get(name, 0.0) + (mss.stamp(self.device) - t0) * 1e3
                self.calls[name] = self.calls.get(name, 0) + 1
                self.active.discard(name)
        return timed

    def install(self, step):
        if step["compiled_sigmoid"]:
            points = self.COMPILED
        else:
            points = self.SEQUENTIAL if step["path"] == "sequential" else self.BATCHED
        for module, attr, name in points:
            original = getattr(module, attr)
            self.saved.append((module, attr, original))
            setattr(module, attr, self.wrap(name, original))
        if step["path"] == "batched" and not step["compiled_sigmoid"]:
            choose = bs.plane_fit_for
            self.saved.append((bs, "plane_fit_for", choose))
            bs.plane_fit_for = lambda grid: self.wrap("fit", choose(grid))

    def remove(self):
        for module, attr, original in reversed(self.saved):
            setattr(module, attr, original)
        self.saved = []

    def result(self, total_ms, compiled):
        ms = self.ms
        inner = 0.0 if compiled else sum(ms.get(k, 0.0) for k in ("grid", "fit", "error_check"))
        out = dict(pass_ms=total_ms, affine_ms=ms.get("affine", 0.0), sigmoid_ms=ms.get("sigmoid", 0.0),
                   other_ms=total_ms - ms.get("affine", 0.0) - ms.get("sigmoid", 0.0),
                   calls=dict(self.calls))
        if not compiled:
            out.update(grid_ms=ms.get("grid", 0.0), fit_ms=ms.get("fit", 0.0),
                       error_check_ms=ms.get("error_check", 0.0),
                       sigmoid_rest_ms=ms.get("sigmoid", 0.0) - inner)
        return out


def stage_pass(step, side, compiled_sigmoid, img, eps, n_splits):
    configure(step, compiled_sigmoid)
    stages = Stages(side.device)
    stages.install(step)
    try:
        t0 = mss.stamp(side.device)
        bound = side.run(step, img, eps, n_splits)
        total = (mss.stamp(side.device) - t0) * 1e3
    finally:
        stages.remove()
    out = stages.result(total, step["compiled_sigmoid"])
    out["bound"] = bound
    return out


def cprofile_pass(step, side, compiled_sigmoid, img, eps, n_splits, rows=25):
    configure(step, compiled_sigmoid)
    profile = cProfile.Profile()
    profile.enable()
    side.run(step, img, eps, n_splits)
    mss.sync_device(side.device)
    profile.disable()
    text = []
    for key in ("cumulative", "tottime"):
        buf = io.StringIO()
        pstats.Stats(profile, stream=buf).strip_dirs().sort_stats(key).print_stats(rows)
        text.append(f"cProfile, sorted by {key}\n{buf.getvalue()}")
    return "\n".join(text)


def torch_profiler_pass(step, side, compiled_sigmoid, img, eps, n_splits, rows=30):
    from torch.profiler import ProfilerActivity, profile

    configure(step, compiled_sigmoid)
    cuda = side.device.type == "cuda"
    # Request CUDA activity whenever a GPU exists, also for the CPU steps: after one CPU-only
    # profile, later profiles in the same process record no device events (torch 2.14 on Colab,
    # see experiments/diag_profiler_cuda.py).
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    with profile(activities=activities) as prof:
        side.run(step, img, eps, n_splits)
        mss.sync_device(side.device)
    averages = prof.key_averages()
    table = None
    for key in (["self_cuda_time_total", "self_device_time_total"] if cuda else []) + ["self_cpu_time_total"]:
        try:
            table = averages.table(sort_by=key, row_limit=rows)
            break
        except Exception:
            continue
    counts = dict(operator_calls=sum(e.count for e in averages if e.key.startswith("aten::")))
    if cuda:
        # device events of the trace: kernels, and the copies between host and device
        try:
            dev_events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            counts["cuda_events"] = len(dev_events)
            counts["cuda_memcpy"] = sum(1 for e in dev_events if "emcpy" in e.name or "emset" in e.name)
            counts["cuda_kernels"] = counts["cuda_events"] - counts["cuda_memcpy"]
        except Exception as e:
            counts["cuda_events_error"] = f"{type(e).__name__}: {e}"
    return table or "", counts


def profile_step(step, side, compiled_sigmoid, img, eps, n_splits, header, path):
    stages = stage_pass(step, side, compiled_sigmoid, img, eps, n_splits)
    cprof = cprofile_pass(step, side, compiled_sigmoid, img, eps, n_splits)
    table, counts = torch_profiler_pass(step, side, compiled_sigmoid, img, eps, n_splits)
    lines = [header, "", "Stage breakdown of one instrumented pass (ms; device synchronized around every stage)"]
    for key in ("pass_ms", "affine_ms", "sigmoid_ms", "grid_ms", "fit_ms", "error_check_ms",
                "sigmoid_rest_ms", "other_ms"):
        if key in stages:
            share = 100.0 * stages[key] / stages["pass_ms"] if stages["pass_ms"] else 0.0
            lines.append(f"  {key:16s} {stages[key]:12.3f}  {share:6.1f}%")
    lines += [f"  calls: {stages['calls']}", f"  counts: {counts}", "", cprof,
              "torch.profiler", table]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return stages, counts


# ----------------------------------------------------------------------------- one cell

def plan_step(probe_s, n_pairs, pairs_per_image, args):
    """(extra warm-up passes, (image, epsilon) pairs, repeats) that fit the budget."""
    full = n_pairs * args.repeats
    if probe_s * (args.warmup - 1 + full) <= args.step_budget_s:
        return args.warmup - 1, n_pairs, args.repeats
    if probe_s * n_pairs <= args.step_budget_s:
        return 0, n_pairs, max(1, min(args.repeats, int(args.step_budget_s / (probe_s * n_pairs))))
    return 0, pairs_per_image, 1  # the first image only, one pass per epsilon


def run_cell(network, n_splits, sides, args, out_dir):
    device = sides["dev"].device
    pairs = [(i, e) for i in range(args.num_images) for e in args.eps_indices]
    compiled_sigmoid = make_compiled_sigmoid() if 8 in args.steps else None
    steps = [STEP_BY_ID[i] for i in sorted(args.steps)]
    if device.type == "cpu":
        steps = [s for s in steps if s["id"] != 4]  # on the CPU step 4 is step 3
    cell = dict(network=network, n_splits=n_splits, chunk=sides["dev"].chunk, steps={}, sampled={})

    def side_of(step):
        return sides["cpu"] if step["on_cpu"] else sides["dev"]

    def one_pass(step, pair):
        side = side_of(step)
        image_id, img = side.images[pair[0]]
        configure(step, compiled_sigmoid)
        if side.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(side.device)
        t0 = mss.stamp(side.device)
        bound = side.run(step, img, EPS[pair[1]], n_splits)
        seconds = mss.stamp(side.device) - t0
        peak = torch.cuda.max_memory_allocated(side.device) / 2 ** 20 if side.device.type == "cuda" else 0.0
        return dict(image_id=image_id, eps_index=pair[1], ms=seconds * 1e3, bound=bound, peak_mb=peak)

    def fail(step, error):
        rec = cell["steps"].setdefault(str(step["id"]), dict(name=step["name"], passes=[]))
        rec["error"] = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
        print(f"    step {step['id']} {step['name']}: FAILED {rec['error']}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # probe, plan and warm-up of every step
    live = []
    for step in steps:
        try:
            probe = one_pass(step, pairs[0])
            typical, done = probe, 1
            if step["compiled_fit"] or step["compiled_sigmoid"]:
                typical, done = one_pass(step, pairs[0]), 2  # the probe was the compilation
            warm, n_pairs, repeats = plan_step(typical["ms"] / 1e3, len(pairs), len(args.eps_indices), args)
            for _ in range(max(0, warm + 1 - done)):
                one_pass(step, pairs[0])
        except Exception as e:  # out of memory, no compiler, ...
            fail(step, e)
            continue
        cell["steps"][str(step["id"])] = dict(
            name=step["name"], device=str(side_of(step).device), first_pass_s=probe["ms"] / 1e3,
            plan=dict(extra_warmup=warm, pairs=n_pairs, repeats=repeats,
                      reduced=(n_pairs, repeats) != (len(pairs), args.repeats)),
            passes=[], error=None)
        live.append(step)
        print(f"    step {step['id']} {step['name']:17s} first pass {probe['ms'] / 1e3:9.3f} s, "
              f"plan: {n_pairs} pairs x {repeats} repeats", flush=True)

    # timed passes, interleaved in rotating order
    turn = 0
    for pair_index, pair in enumerate(pairs):
        for rep in range(args.repeats):
            order = live[turn % len(live):] + live[:turn % len(live)] if live else []
            turn += 1
            for step in order:
                rec = cell["steps"][str(step["id"])]
                if rec["error"] or pair_index >= rec["plan"]["pairs"] or rep >= rec["plan"]["repeats"]:
                    continue
                try:
                    row = one_pass(step, pair)
                except Exception as e:
                    fail(step, e)
                    continue
                row["rep"] = rep
                rec["passes"].append(row)

    # profile of every step: the first image and epsilon
    if not args.no_profile:
        for step in live:
            rec = cell["steps"][str(step["id"])]
            if rec["error"] or not rec["passes"]:
                continue
            typical = statistics.median(p["ms"] for p in rec["passes"]) / 1e3
            if typical > args.profile_max_s:
                rec["profile_skipped"] = f"pass of {typical:.1f} s is above --profile-max-s"
                continue
            side = side_of(step)
            image_id, img = side.images[0]
            header = (f"{network}, n_splits {n_splits}, step {step['id']} {step['name']} ({step['text']}), "
                      f"device {side.device}, image {image_id}, eps index {args.eps_indices[0]}")
            path = os.path.join(out_dir, "profiles", f"{network}_S{n_splits}_{step['id']}_{step['name']}.txt")
            try:
                rec["stages"], rec["counts"] = profile_step(
                    step, side, compiled_sigmoid, img, EPS[args.eps_indices[0]], n_splits, header, path)
            except Exception as e:
                rec["profile_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"

    # soundness reference: sampled autograd maximum of |d logit / dt| on the haze path
    cpu = sides["cpu"]
    for image_index, eps_index in pairs:
        image_id, img = cpu.images[image_index]
        cell["sampled"][f"{image_id}|{eps_index}"] = css.sampled_max_derivative(
            cpu.net, img, EPS[eps_index], args.samples)
    restore_library()
    return cell


# ----------------------------------------------------------------------------- report

def median_by_pair(rec):
    groups = {}
    for p in rec["passes"]:
        groups.setdefault((p["image_id"], p["eps_index"]), []).append(p)
    return {k: dict(ms=statistics.median(p["ms"] for p in v), bound=v[-1]["bound"],
                    peak_mb=max(p["peak_mb"] for p in v), n=len(v)) for k, v in groups.items()}


def compare(a, b):
    """Speedup of b over a and the largest relative bound difference, on their common pairs."""
    common = sorted(set(a) & set(b))
    if not common:
        return None, None
    speedup = sum(a[k]["ms"] for k in common) / sum(b[k]["ms"] for k in common)
    rel = max(abs(b[k]["bound"] - a[k]["bound"]) / abs(a[k]["bound"]) for k in common)
    return speedup, rel


def summarize(cells):
    rows = []
    for cell in cells:
        ids = sorted(int(i) for i in cell["steps"])
        stats = {i: median_by_pair(cell["steps"][str(i)]) for i in ids}
        measured = [set(st) for st in stats.values() if st]
        common = sorted(set.intersection(*measured)) if measured else []
        previous = None
        for i in ids:
            rec, st = cell["steps"][str(i)], stats[i]
            row = dict(network=cell["network"], n_splits=cell["n_splits"], step=i, name=rec["name"],
                       device=rec.get("device", ""), error=rec.get("error") or "")
            if st:
                row.update(ms_per_pass=statistics.mean(v["ms"] for v in st.values()),
                           bound_mean=statistics.mean(v["bound"] for v in st.values()),
                           pairs=len(st), passes=len(rec["passes"]), common_pairs=len(common),
                           ms_common_pairs=statistics.mean(st[k]["ms"] for k in common) if common else None,
                           bound_common_pairs=statistics.mean(st[k]["bound"] for k in common) if common else None,
                           reduced=rec["plan"]["reduced"], first_pass_s=rec["first_pass_s"],
                           peak_vram_mb=max(v["peak_mb"] for v in st.values()))
                row["min_bound_over_sampled"] = min(
                    v["bound"] / cell["sampled"][f"{k[0]}|{k[1]}"] for k, v in st.items())
                if stats.get(0):
                    row["speedup_vs_original"], row["rel_diff_vs_original"] = compare(stats[0], st)
                if previous is not None:
                    row["previous_step"] = previous
                    row["speedup_vs_previous"], row["rel_diff_vs_previous"] = compare(stats[previous], st)
                # passes after which a compiled step has won its first pass back against a2_eager
                if i in (7, 8) and stats.get(6):
                    shared = sorted(set(stats[6]) & set(st))
                    if shared:
                        gain = statistics.mean(stats[6][k]["ms"] - st[k]["ms"] for k in shared)
                        row["break_even_passes_vs_a2_eager"] = rec["first_pass_s"] * 1e3 / gain if gain > 0 else float("inf")
                stages = rec.get("stages")
                if stages:
                    total = stages["pass_ms"] or 1.0
                    row["instrumented_pass_ms"] = stages["pass_ms"]
                    for key in ("affine", "grid", "fit", "error_check", "sigmoid_rest", "sigmoid", "other"):
                        if f"{key}_ms" in stages:
                            row[f"{key}_pct"] = 100.0 * stages[f"{key}_ms"] / total
                for key, value in (rec.get("counts") or {}).items():
                    if isinstance(value, (int, float)):
                        row[key] = value
                previous = i
            rows.append(row)
    return rows


def fmt(value, spec, empty="-"):
    if value is None or value == "":
        return empty
    return format(value, spec)


def markdown(rows, env):
    out = ["# Speedup ladder", "",
           f"Commit `{env['commit']}`; CPU {env['cpu']} ({env['torch_threads']} torch threads); "
           f"device {env.get('gpu', env['device'])}; torch {env['torch']}; float64; seed {env['seed']}; "
           f"{env['num_images']} images, eps indices {env['eps_indices']}; "
           f"median of up to {env['repeats']} timed passes per (image, epsilon), mean over the pairs; "
           "steps interleaved. Step 0 is the current code with all selectors off "
           "(upstream code + the soundness fixes). `*` = fewer repeats or pairs than planned "
           "(the pass was too long for the budget); speedups use the pairs both steps measured. "
           "The first pass of a step carries its compilation.", ""]
    cells = sorted({(r["network"], r["n_splits"]) for r in rows},
                   key=lambda c: (["3layer", "4layer", "5layer", "big"].index(c[0]), c[1]))
    for network, n_splits in cells:
        in_cell = [r for r in rows if (r["network"], r["n_splits"]) == (network, n_splits) and "pairs" in r]
        note = ""
        if in_cell:
            note = (f"Times and bounds over the {in_cell[0]['common_pairs']} (image, epsilon) pairs every step "
                    f"measured (of {max(r['pairs'] for r in in_cell)}).")
        out += [f"## {network}, {n_splits} split{'s' if n_splits != 1 else ''}", "", note, "",
                "| step | device | ms per pass | x vs original | x vs previous | bound | rel. diff vs original "
                "| bound / sampled max | fit % | affine % | checks % | peak VRAM MiB | first pass s |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if (r["network"], r["n_splits"]) != (network, n_splits):
                continue
            if r["error"]:
                out.append(f"| {r['step']} {r['name']} | {r['device']} | failed: {r['error']} |||||||||||")
                continue
            if "ms_per_pass" not in r:
                continue
            star = "*" if r.get("reduced") else ""
            out.append(
                f"| {r['step']} {r['name']} | {r['device']} | {fmt(r.get('ms_common_pairs'), '.2f')}{star} "
                f"| {fmt(r.get('speedup_vs_original'), '.2f')} | {fmt(r.get('speedup_vs_previous'), '.2f')} "
                f"| {fmt(r.get('bound_common_pairs'), '.4f')} | {fmt(r.get('rel_diff_vs_original'), '.1e')} "
                f"| {r['min_bound_over_sampled']:.4f} | {fmt(r.get('fit_pct'), '.1f')} "
                f"| {fmt(r.get('affine_pct'), '.1f')} | {fmt(r.get('error_check_pct'), '.1f')} "
                f"| {r['peak_vram_mb']:.0f} | {r['first_pass_s']:.2f} |")
        out.append("")
    out += ["Percentages come from one instrumented pass per step (device synchronized around every "
            "stage), not from the timed passes; inside the compiled sigmoid the stages cannot be "
            "separated. Full profiles: `profiles/`.", ""]
    return "\n".join(out)


def write_report(out_dir):
    cells_dir = os.path.join(out_dir, "cells")
    cells = [json.load(open(os.path.join(cells_dir, f))) for f in sorted(os.listdir(cells_dir))
             if f.endswith(".json")]
    env = json.load(open(os.path.join(out_dir, "environment.json")))
    passes = [dict(network=c["network"], n_splits=c["n_splits"], step=int(i), name=rec["name"],
                   device=rec.get("device", ""), **p)
              for c in cells for i, rec in c["steps"].items() for p in rec["passes"]]
    if passes:
        with open(os.path.join(out_dir, "passes.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(passes[0]))
            writer.writeheader()
            writer.writerows(passes)
    rows = summarize(cells)
    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    text = markdown(rows, env)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    print("\n" + text)
    bad = [r for r in rows if r.get("min_bound_over_sampled", 1.0) < 1.0]
    for r in bad:
        print(f"BELOW THE SAMPLED MAXIMUM: {r['network']} S={r['n_splits']} step {r['step']} {r['name']}: "
              f"{r['min_bound_over_sampled']:.6f}")
    return 1 if bad else 0


# ----------------------------------------------------------------------------- main

def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    if args.report_only:
        return write_report(out_dir)
    for sub in ("cells", "profiles"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")
    if device.type == "mps":
        raise SystemExit("MPS has no float64; use cuda or cpu")
    torch.set_default_dtype(DTYPE)
    if args.threads:
        torch.set_num_threads(args.threads)
    env = environment(args, device)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=1)
    print(json.dumps(env, indent=1), flush=True)

    # identifies the hardware of every cell; a resumed run must match it
    machine = dict(cpu=env["cpu"], gpu=env.get("gpu", "none"), torch=env["torch"])
    cpu = torch.device("cpu")
    big_chunk = args.big_chunk or default_big_chunk(device)
    for network in args.networks:
        chunk = big_chunk if network == "big" else None
        sides = dict(cpu=Side(network, cpu, args.num_images, args.seed, chunk))
        sides["dev"] = sides["cpu"] if device.type == "cpu" else Side(network, device, args.num_images, args.seed, chunk)
        for n_splits in args.splits:
            path = os.path.join(out_dir, "cells", f"{network}_S{n_splits}.json")
            if os.path.exists(path):
                old = json.load(open(path)).get("machine")
                if old != machine:
                    raise SystemExit(
                        f"{path} was measured on {old}, this run is on {machine}. A ladder must come "
                        "from one machine: use a new --out folder.")
                print(f"\n== {network}, n_splits {n_splits}: done on this machine, skipped", flush=True)
                continue
            print(f"\n== {network}, n_splits {n_splits}", flush=True)
            cell = run_cell(network, n_splits, sides, args, out_dir)
            cell["machine"] = machine
            with open(path + ".tmp", "w") as f:
                json.dump(cell, f, indent=1)
            os.replace(path + ".tmp", path)
        del sides
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return write_report(out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
