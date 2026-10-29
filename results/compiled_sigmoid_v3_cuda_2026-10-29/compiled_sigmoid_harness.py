"""Whole precise_sigmoid under TorchInductor: correctness gate, recompilation, cold and
warm compilation, steady state.

Harness of the notebook COLAB_COMPILED_PRECISE_SIGMOID.ipynb. Pasado is used as it is at
a57b560: nothing in the library changes, the modes below are selected from outside
(module flags, and batched_splits.precise_sigmoid swapped for a compiled function).

PRE-REGISTERED PROTOCOL, VERSION 3 (written 2026-10-29, before any measurement of v3;
the notebook stores the sha256 of this file with the results)

History. Each earlier version ran once on an A100-80GB and its gate stopped it before
any timing; both runs are archived unchanged in Pasado.
  v1 (workspace 788589d, first committed as 91e02e4): 868 of 876 checks; G4 failed 4
     times (the graph serving the first layer depended on the shapes seen before) and G5
     4 times (one normwise metric over the whole dg, dominated by D on narrow boxes).
     results/compiled_sigmoid_cuda_2026-10-28_gate_stop, Pasado 1ac62f9.
  v2 (workspace 7c9a8ef, first committed as e18278c): mark_dynamic and G5 by blocks;
     892 of 908 checks, every bound and soundness gate passed, but the D contract could
     not run: its (A, B, C) came from a second, differently fused compiled graph, which
     on the A100 was never bit-identical to the measured one.
     results/compiled_sigmoid_v2_cuda_2026-10-29_gate_stop, Pasado aff244b.
Version 3 changes one thing: the graph whose A, B, C and D are checked is the graph whose
correctness, speed, memory and break-even are measured. The formal candidate is
precise_sigmoid_instrumented below, a line-for-line copy of batched_splits.precise_sigmoid
at a57b560 that also returns (A, B, C, lx, ux, ly, uy, D) as tensor outputs of the
compiled function itself (return_diagnostics=True). The batched pass uses its main outputs
and drops the diagnostics; their cost is part of the candidate. A gate check proves the
copy faithful: run eagerly, it is bit-identical to batched_splits.precise_sigmoid. The
uninstrumented compiled graph (v2's candidate) is an ablation only: output difference,
speed and kernels are reported, outside the formal outcome. Everything else as in v2;
no tolerance is raised.

Modes, all float64, seed 7, every batched mode through batched_splits.split_bounds:
  sequential_G              the original loop over splits, gelsd on the CPU
  batched_G                 batched pass, gelsd on the CPU (production default)
  batched_A2_eager          batched pass, planar_fit.fit_planes eager (PASADO_CLOSED_FORM_FIT)
  batched_A2_compiled_fit   as above with fit_planes compiled (PASADO_COMPILED_FIT, through
                            the production wrapper planar_fit.compiled_fit_planes)
  batched_compiled_sigmoid  the formal candidate: torch.compile(precise_sigmoid_instrumented,
                            fullgraph=True), return_diagnostics=True, with the eager fit_planes
                            inside and the generator dimension (dimension 1 of z.rg and z.dg)
                            marked dynamic before every call (torch._dynamo.mark_dynamic).
                            Never nested: fit_planes is not compiled separately in this mode.
  batched_compiled_sigmoid_uninstrumented
                            ablation, not in the outcome: torch.compile of
                            batched_splits.precise_sigmoid itself, marked the same way.
Grid: networks 3layer, 4layer, 5layer, big; S = n_splits in 1, 8, 32, 64; 3 images; eps
indices 0 and 8. big runs in chunks of --big-chunk splits (from the VRAM, as in
COLAB_V3_COMPILED: 32 on 40 GB, 64 on 80 GB).

Correctness gate, checked on the whole grid with a new compiled callable per (network, S);
all must hold, otherwise the notebook stops before any timing:
  G1 fullgraph=True compiles every shape met (no exception).
  G2 no fallback: every sigmoid layer call goes through the compiled candidate, a
     profiled pass shows one "Torch-Compiled Region" per call and no lstsq, planar_fit.
     compile_fallback stays empty, and the plane fit selected inside is the eager
     fit_planes (no nested compile). Exactly one graph is compiled per (network, S), and
     every compiled region of the profiled pass carries the same compile id.
  G3 the NumPy generator state after a pass is identical to the eager A2 pass.
  G4 two compiled passes with the same seed give the same bits.
  G5 every call of the compiled candidate against eager batched_splits.precise_sigmoid on
     the same input, normwise (max |diff| / max |eager|, or max |diff| where eager is all
     zero), at most 1e-12 for rc, rg, dc, and for the first two blocks of dg: the
     propagated generators A rg + B dg (rows :G0) and the zeros (rows G0:, the diagonal D
     of the last N rows excluded); the same shapes and the same nonzero flag. Also, at
     every call:
       copy      the candidate run eagerly gives the four main outputs and the flag of
                 batched_splits.precise_sigmoid bit for bit;
       D output  the D the compiled candidate returns equals, bit for bit, the D it
                 propagates (the diagonal of the last N rows of its dg).
     The normwise difference of D against eager is recorded, not judged. D is judged at
     S = 64, eps index 8, as in G1 of compare_lstsq_solvers.py:
       contract  per box, shortfall = max(0, D_true - D) / W with D_true the maximum of
                 |sigmoid'(x) y - (A x + B y + C)| over the box (lstsq_oracle.true_max_error,
                 mpmath, 50 digits) and W = |A| r_x + |B| r_y + D. Compiled: A, B, C and
                 the box from the diagnostic outputs of the compiled candidate, D from its
                 dg. Eager: the same from the candidate run eagerly. Pass, per network over
                 its sample: worst compiled shortfall <= 10 * max(1e-12, worst eager
                 shortfall). Boxes whose D_true cannot be computed (no convergence) are
                 listed, not counted. Also recorded: the absolute shortfall and the
                 shortfall relative to D.
       sample    fixed before the run: networks 3layer, 4layer, 5layer, big; every sigmoid
                 layer; the 3 images; S = 64, eps index 8; the boxes of all chunks of the
                 pass. Per (network, image, layer): the 32 boxes with the largest
                 |D_compiled - D_eager|, then the 32 others with the largest
                 |D_compiled - D_eager| / W_compiled, then 64 others uniformly at random
                 from numpy.random.default_rng([20261028, network index, image id, layer]);
                 ties by box index. At most 128 boxes, 4992 in all.
     Recorded, not judged: the ablation's main outputs against the candidate's (normwise,
     and whether bit-identical).
  G6 every per-split bound within 1e-10 relative of eager A2.
  G7 the final bound (max over splits) within 1e-10 relative of sequential G.
  G8 the final bound is not below the sampled maximum of the true derivative (autograd,
     1001 samples over [0, eps]); also with unseeded noise, on 3layer, as in
     experiments/test_batched_splits.py.

Recompilation: per (network, S), from a new callable after torch._dynamo.reset(): the
graphs compiled, the input shape that triggered each, the dimensions that became dynamic
(TORCH_LOGS=dynamic) and whether the recompile limit was hit. Separately, one shared
callable over the whole grid (two rounds) with Dynamo's default recompile limit, which is
not raised, for the whole sigmoid and for the compiled fit.

Timing:
  start   per (network, S, compiled mode) two processes sharing one Inductor and Triton
          cache directory that starts empty: the first is cold, the second warm. Each
          process imports, initialises CUDA with one eager A2 pass, creates a new compiled
          callable, runs the first sweep over the 6 (image, eps) passes (e2e_s, includes
          compiling), then REPEATS more passes of each (its own steady state).
  steady  one process with a warm cache; per (network, S) a new compiled state; the five
          modes and the ablation interleaved round-robin with the order rotated every
          repeat, REPEATS = 5 passes per (image, eps) per mode, so 30 samples per
          (network, S, mode), under torch.compiler.set_stance("fail_on_recompile").
          Reported: median, min, max; kernels per sigmoid layer and per pass (profiler);
          peak VRAM of one pass. The checks of the ablation are recorded with
          formal=False and do not enter C1.

Definitions:
  overhead_s  e2e_s of a start process minus the sum of its 6 per-(image, eps) steady
              medians: the compilation a run pays in that cache state.
  break_even  overhead_s / (t_ref - t_mode) passes, t the steady-state medians (steady
              process, pooled over its 30 samples); infinite when the mode is not
              faster. References: batched_G, and the best path without compilation,
              min(batched_G, batched_A2_eager).

Criteria:
  C1 correctness: the gate (G1 to G8) passes, and so does every check of the steady
     process (formal modes only: the final bound within 1e-10 of sequential G, one compiled
     region per sigmoid call in the compiled modes and none in the others, lstsq only in
     the gelsd modes, no compiled-fit fallback, every configuration ran) and of every
     start process (no compiled-fit fallback, at least one graph compiled, every bound
     within 1e-10 of sequential G from the steady process). The gate stops the notebook
     before any timing; a later failure gives the outcome stop in criteria.json.
  C2 the compiled whole sigmoid is not slower than compiled-fit-only by more than 5%:
     median(batched_compiled_sigmoid) <= 1.05 * median(batched_A2_compiled_fit) in every
     (network, S).
  C3 a meaningful gain where about 9 ms of fixed work remains: median(batched_A2_compiled_fit)
     / median(batched_compiled_sigmoid) >= 1.2 on 3layer, 4layer and 5layer at S = 1.
  C4 cold start and break-even are reported separately for the cold and the warm cache,
     never folded into the warm steady-state numbers.
Outcome, applied mechanically by `summarize`:
  stop               C1 fails.
  opt-in             C1 holds, but C2 or C3 fails, or the warm break-even against the best
                     path without compilation exceeds 480 passes (one standard sweep of a
                     network: 30 images x 16 eps) for a small network at S = 1. A gain on
                     the warm path only lands here.
  auto candidate     C1, C2 and C3 hold, and that warm break-even is <= 480 passes for all
                     three small networks at S = 1.
  default candidate  auto candidate, and the cold break-even is <= 480 passes as well.
  incomplete         a configuration of the grid is missing; no outcome until rerun.
No outcome changes production code by itself.

Subcommands: gate, shared, start, steady, logs, summarize (see main()).
"""
T_START = __import__("time").perf_counter()  # before the heavy imports, for process_s

import argparse
import copy
import csv
import glob
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch._dynamo

REPO = os.environ.get("PASADO_REPO", "/content/Pasado")
for _p in ("forward_mode_tensorized_src", "Section_5_4", "experiments"):
    sys.path.insert(0, os.path.join(REPO, _p))

import batched_splits as bs  # noqa: E402
import planar_fit  # noqa: E402
import precise_transformer as pt  # noqa: E402
from check_sweep_soundness import sampled_max_derivative  # noqa: E402
from lstsq_oracle import true_max_error  # noqa: E402
from measure_split_sweep import build_forward_precise, load_network_and_images  # noqa: E402
from test_batched_splits import (EPS_SCHEDULE, _numpy_state, _same_state,  # noqa: E402
                                 sequential_split_bounds)

T_IMPORTED = time.perf_counter()

NETWORKS = ["3layer", "4layer", "5layer", "big"]
SMALL_NETWORKS = ["3layer", "4layer", "5layer"]
SPLITS = [1, 8, 32, 64]
EPS_INDICES = [0, 8]
NUM_IMAGES = 3
REPEATS = 5
SEED = 7
RTOL_BOUND = 1e-10
RTOL_TENSOR = 1e-12
AUTOGRAD_SAMPLES = 1001
C2_MAX_SLOWDOWN = 1.05
C3_MIN_SPEEDUP = 1.2
BREAK_EVEN_PASSES = 480  # 30 images x 16 eps
# G5 for D (since v2): where, the sample, and the G1 rule of compare_lstsq_solvers.py
D_CONTRACT_S = 64
D_CONTRACT_EPS = 8
D_SAMPLE_TOP = 32  # by |D_compiled - D_eager|, then as many by that over W_compiled
D_SAMPLE_RANDOM = 64
D_SAMPLE_SEED = 20261028
D_CONTRACT_FACTOR = 10
D_CONTRACT_FLOOR = 1e-12

SEQUENTIAL = "sequential_G"
COMPILED_FIT = "batched_A2_compiled_fit"
COMPILED_SIGMOID = "batched_compiled_sigmoid"
ABLATION = "batched_compiled_sigmoid_uninstrumented"
MODES = [SEQUENTIAL, "batched_G", "batched_A2_eager", COMPILED_FIT, COMPILED_SIGMOID]  # formal
STEADY_MODES = MODES + [ABLATION]
BATCHED_MODES = STEADY_MODES[1:]
COMPILED_MODES = [COMPILED_FIT, COMPILED_SIGMOID]
CLOSED_FORM_MODES = STEADY_MODES[2:]

EAGER_SIGMOID = bs.precise_sigmoid
OUTPUTS = ("rc", "rg", "dc", "dg")


# ---------------------------------------------------------------- shared pieces

def grid(args):
    """(networks, splits, images, eps indices); --smoke is a small CPU check of the harness."""
    if args.smoke:
        return ["3layer"], [1, 2], 1, [0, 8]
    return NETWORKS, SPLITS, NUM_IMAGES, EPS_INDICES


def contract_split(args):
    """The S at which G5 judges D (the largest S of the smoke grid under --smoke)."""
    return 2 if args.smoke else D_CONTRACT_S


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def unique_graphs():
    return torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)


def cache_entries(fn):
    try:
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(fn.__code__))
    except Exception:
        return None


def mark(text):
    """A marker in stderr, between the TORCH_LOGS lines, to split the log per configuration."""
    print(f"### CONFIG {text}", file=sys.stderr, flush=True)


def short(e):
    return f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"


def worst_of(values):
    """max, NaN if any value is NaN (so a comparison with a limit fails)."""
    values = list(values)
    if not values or any(math.isnan(v) for v in values):
        return math.nan
    return max(values)


def normwise(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / s if s else d


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def set_production_flags():
    pt.USE_VECTORIZED_PRECISE = True
    pt.USE_VECTORIZED_BOUNDARY = True
    pt.USE_BATCHED_LSTSQ = True
    pt.USE_BATCHED_GRID = True
    pt.USE_REAL_CUBIC = True


class Net:
    def __init__(self, name, device, num_images, big_chunk):
        self.name = name
        self.module, self.images = load_network_and_images(name, device, torch.float64, num_images)
        self.forward_fn = build_forward_precise(pt, self.module, name)
        self.layers = bs.extract_affine_layers(self.module, name)
        self.sigmoid_layers = len(self.layers) - 1
        self.chunk = big_chunk if name == "big" else None

    def chunks(self, S):
        c = S if self.chunk is None else max(1, min(self.chunk, S))
        return -(-S // c)

    def net64(self):
        return copy.deepcopy(self.module).to("cpu", torch.float64)


class Tracker:
    """The sigmoid of one mode: counts its calls and records every call that compiled a graph.

    For batched_compiled_sigmoid it wraps the compiled function (from outside, so nothing
    of it is traced); for the other batched modes the eager precise_sigmoid, so a graph
    compiled inside (compiled fit) is recorded too.
    """

    def __init__(self, fn, mode):
        self.fn, self.mode = fn, mode
        self.calls = 0
        self.events = []
        self.tag = {}

    def __call__(self, z, noise):
        before = unique_graphs()
        t0 = time.perf_counter()
        out = self.fn(z, noise)
        new = unique_graphs() - before
        if new:
            S_chunk, G0, N = z.rg.shape
            self.events.append(dict(self.tag, mode=self.mode, call=self.calls, S_chunk=S_chunk, G0=G0, N=N,
                                    new_graphs=new, host_s=time.perf_counter() - t0))
        self.calls += 1
        return out


def marked(fn):
    """fn with the generator dimension of z marked dynamic before every call, so one graph
    serves every layer whatever shapes came before (without it v1 failed G4)."""
    def run(z, noise):
        torch._dynamo.mark_dynamic(z.rg, 1)
        torch._dynamo.mark_dynamic(z.dg, 1)
        return fn(z, noise)
    return run


def precise_sigmoid_instrumented(z, noise, return_diagnostics=True):
    """The v3 candidate: batched_splits.precise_sigmoid of a57b560, line for line, that can
    also return the values of its error term as tensor outputs, each [S*N]:
    (A, B, C, lx, ux, ly, uy, D). The gate checks that, run eagerly, it gives the main
    outputs of batched_splits.precise_sigmoid bit for bit."""
    S, G0, N = z.rg.shape

    lx, ux = bs._bounds(z.rc, z.rg)
    ly, uy = bs._bounds(z.dc, z.dg)

    # SigmoidZonotope on the real part
    s_lb, s_ub = torch.sigmoid(lx), torch.sigmoid(ux)
    lam = torch.min(s_lb * (1 - s_lb), s_ub * (1 - s_ub))
    mu1 = 0.5 * (s_ub + s_lb - lam * (ux + lx))
    mu2 = 0.5 * (s_ub - s_lb - lam * (ux - lx))
    sig_c = lam * z.rc + mu1

    # Planar regression of sigmoid'(x) * y on the 5 x 5 grid of every (split, neuron)
    # box. The boxes are independent, so [S, N] flattens to one batch of S*N.
    lx_f, ux_f, ly_f, uy_f = (t.reshape(-1) for t in (lx, ux, ly, uy))
    grid = pt.get_linspace_batched(lx_f, ux_f, ly_f, uy_f)  # [S*N, 25, 2]
    gx, gy = grid[..., 0], grid[..., 1]
    targets = (1.0 - torch.sigmoid(gx)) * (torch.sigmoid(gx) * gy)
    coef = bs.plane_fit_for(grid)(grid, targets)  # [S*N, 3], intercept first
    As = coef[:, 1] + noise.reshape(-1)
    Bs = coef[:, 2]
    Cs = coef[:, 0]
    a_nonzero = ((As > 0) | (As < 0)).all()

    # worst-case approximation error, same three checks as compute_max_error
    abc = torch.stack((As, Bs, Cs), dim=1)
    corners = pt.check_corners_tensor(abc, lx_f, ux_f, ly_f, uy_f)
    err_ly, err_uy = pt.check_nonlinear_boundary_tensor(lx_f, ux_f, ly_f, uy_f, abc)
    errors = torch.maximum(torch.maximum(corners, err_ly), err_uy).reshape(S, N)
    diagnostics = (As, Bs, Cs, lx_f, ux_f, ly_f, uy_f, errors.reshape(-1))

    As, Bs, Cs = As.reshape(S, N), Bs.reshape(S, N), Cs.reshape(S, N)
    dual_c = As * z.rc + Bs * z.dc + Cs

    # Generator layout per split, G0 + 2N rows:
    #   real: [lam * rg | diag(mu2) | 0]      dual: [A * rg + B * dg | 0 | diag(errors)]
    new_rg = z.rg.new_zeros((S, G0 + 2 * N, N))
    new_rg[:, :G0, :] = lam.unsqueeze(1) * z.rg
    new_rg[:, G0:G0 + N, :].diagonal(dim1=1, dim2=2).copy_(mu2)

    new_dg = z.rg.new_zeros((S, G0 + 2 * N, N))
    new_dg[:, :G0, :] = As.unsqueeze(1) * z.rg + Bs.unsqueeze(1) * z.dg
    new_dg[:, G0 + N:, :].diagonal(dim1=1, dim2=2).copy_(errors)

    out = bs.BatchedDualZonotope(sig_c, new_rg, dual_c, new_dg)
    if return_diagnostics:
        return out, a_nonzero, diagnostics
    return out, a_nonzero


def drop_diagnostics(fn):
    """fn for the batched pass: its main outputs; the diagnostics are computed and dropped."""
    def run(z, noise):
        out, a_nonzero, _ = fn(z, noise)
        return out, a_nonzero
    return run


def make_trackers():
    """A tracker per batched mode, the compiled ones new torch.compile objects (they compile
    on their first call). The candidate's tracker keeps the compiled function with its
    diagnostics as .with_diagnostics."""
    trackers = {m: Tracker(EAGER_SIGMOID, m) for m in BATCHED_MODES}
    candidate = marked(torch.compile(precise_sigmoid_instrumented, fullgraph=True))
    trackers[COMPILED_SIGMOID] = Tracker(drop_diagnostics(candidate), COMPILED_SIGMOID)
    trackers[COMPILED_SIGMOID].with_diagnostics = candidate
    trackers[ABLATION] = Tracker(marked(torch.compile(EAGER_SIGMOID, fullgraph=True)), ABLATION)
    return trackers


def fresh_compiled_state():
    """Forget every compiled graph in this process: Dynamo's caches and planar_fit's
    compiled wrapper. The disk cache stays."""
    torch._dynamo.reset()
    planar_fit._compiled.clear()


def set_flags(mode):
    bs.USE_AUTO_PLANE_FIT = False
    bs.USE_CLOSED_FORM_FIT = mode in CLOSED_FORM_MODES
    bs.USE_COMPILED_FIT = mode == COMPILED_FIT


def set_mode(mode, trackers):
    set_flags(mode)
    bs.precise_sigmoid = trackers[mode] if mode in trackers else EAGER_SIGMOID


def restore():
    set_flags("batched_G")
    bs.precise_sigmoid = EAGER_SIGMOID


def plane_fit_inside(device):
    """The plane fit precise_sigmoid selects under the current flags (reads only the shape)."""
    return bs.plane_fit_for(torch.zeros(1, 25, 2, dtype=torch.float64, device=device))


def run_pass(mode, net, img, eps, S, seed=SEED):
    """Per-split bounds of one pass, [S]."""
    if mode == SEQUENTIAL:
        return sequential_split_bounds(net.forward_fn, img, eps, S, seed, img.device, img.dtype)
    return bs.split_bounds(net.layers, img, eps, S, seed=seed, chunk=net.chunk)


def timed_pass(mode, net, img, eps, S, device):
    sync(device)
    t0 = time.perf_counter()
    bounds = run_pass(mode, net, img, eps, S)
    sync(device)
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, bounds.max().item()


def profile_counts(fn, device):
    """CUDA kernels, memcpy, memset, compiled-region entries and lstsq calls of one call."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    sync(device)
    with profile(activities=acts) as prof:
        fn()
        sync(device)
    c = dict(kernels=0, memcpy=0, memset=0, compiled_regions=0, lstsq=0)
    ids = set()
    for e in prof.events():
        name = e.name
        if e.device_type == DeviceType.CUDA:
            low = name.lower()
            key = "memcpy" if low.startswith("memcpy") else "memset" if low.startswith("memset") else "kernels"
            c[key] += 1
        elif name.startswith("Torch-Compiled Region"):
            c["compiled_regions"] += 1
            ids.add(name.split(": ", 1)[-1])
        elif name == "aten::linalg_lstsq":
            c["lstsq"] += 1
    if device.type != "cuda":
        c["kernels"] = c["memcpy"] = c["memset"] = None
    c["region_ids"] = " ".join(sorted(ids))
    return c


def peak_vram(fn, device):
    """(peak MiB, peak above the memory allocated before the call) of one call."""
    if device.type != "cuda":
        return None, None
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return peak / 2**20, (peak - base) / 2**20


def setup(args):
    set_production_flags()
    torch.set_default_dtype(torch.float64)
    restore()
    os.makedirs(args.out, exist_ok=True)
    return torch.device(args.device)


# ---------------------------------------------------------------- gate

class Gate:
    def __init__(self):
        self.checks = []

    def check(self, name, ok, value=None, limit=None, detail="", **where):
        self.checks.append(dict(check=name, **where, value=value, limit=limit, ok=bool(ok), detail=detail))
        if not ok:
            print(f"  FAIL {name} {where} value={value} limit={limit} {detail}", flush=True)
        return ok


def _d_of(dg, G0, N):
    """The error term D of every (split, neuron): the diagonal of the last N rows of dg."""
    return dg[:, G0 + N:, :].diagonal(dim1=1, dim2=2)


def compare_layers(net, img, eps, S, candidate, ablation, keep_boxes=False):
    """Every precise_sigmoid call of an eager A2 pass: the compiled candidate (with its
    diagnostics) against eager batched_splits.precise_sigmoid on the same input, the
    candidate's copy run eagerly against the same (bit for bit), and the uninstrumented
    ablation against the candidate (recorded only). With keep_boxes the per-box values of
    the D contract are kept (on the CPU, all chunks)."""
    rows, boxes = [], []

    def comparer(z, noise):
        out_e, ok_e = EAGER_SIGMOID(z, noise)
        out_x, ok_x, diag_x = precise_sigmoid_instrumented(z, noise)  # the copy, eager
        k = len(rows)
        S_, G0, N = z.rg.shape
        row = dict(call=k, layer=k % net.sigmoid_layers + 1, S_chunk=S_, G0=G0, N=N, flag_eager=bool(ok_e))
        row["copy_identical"] = bool(all(torch.equal(getattr(out_x, n), getattr(out_e, n)) for n in OUTPUTS)
                                     and bool(ok_x) == bool(ok_e))
        del out_x
        out_c, ok_c, diag_c = candidate(z, noise)
        row["flag_compiled"] = bool(ok_c)
        shapes_ok = all(getattr(out_c, n).shape == getattr(out_e, n).shape
                        and getattr(out_c, n).dtype == getattr(out_e, n).dtype for n in OUTPUTS)
        row["shapes_ok"] = shapes_ok
        row["d_output_identical"] = False
        if shapes_ok:
            for name in ("rc", "rg", "dc"):
                row[f"rel_{name}"] = normwise(getattr(out_c, name), getattr(out_e, name))
            dg_c, dg_e = out_c.dg, out_e.dg
            eye = torch.eye(N, dtype=torch.bool, device=dg_c.device)
            row["rel_dg_generators"] = normwise(dg_c[:, :G0], dg_e[:, :G0])
            row["rel_dg_zeros"] = max(normwise(dg_c[:, G0:G0 + N], dg_e[:, G0:G0 + N]),
                                      normwise(dg_c[:, G0 + N:].masked_fill(eye, 0),
                                               dg_e[:, G0 + N:].masked_fill(eye, 0)))
            d_c, d_e = _d_of(dg_c, G0, N), _d_of(dg_e, G0, N)
            # recorded, not judged: D by itself, and the v1 metric over the whole dg
            row["rel_dg_D"] = normwise(d_c, d_e)
            row["rel_dg"] = normwise(dg_c, dg_e)
            row["d_max_shortfall_rel"] = ((d_e - d_c) / d_e.abs().clamp_min(1e-300)).max().item()
            row["d_max_shortfall_abs"] = (d_e - d_c).max().item()
            row["max_rel"] = worst_of(row[k_] for k_ in ("rel_rc", "rel_rg", "rel_dc", "rel_dg_generators",
                                                          "rel_dg_zeros"))
            # the D the candidate returns must be the D it propagates
            row["d_output_identical"] = bool(torch.equal(diag_c[7], d_c.reshape(-1)))
            if keep_boxes:
                boxes.append(dict(layer=row["layer"], D_c=d_c.reshape(-1).cpu(), D_e=d_e.reshape(-1).cpu(),
                                  abc_c=torch.stack(diag_c[:3], dim=1).cpu(), box_c=[t.cpu() for t in diag_c[3:7]],
                                  abc_e=torch.stack(diag_x[:3], dim=1).cpu(), box_e=[t.cpu() for t in diag_x[3:7]]))
        else:
            row["max_rel"] = math.inf
        out_p, ok_p = ablation(z, noise)
        same = all(getattr(out_p, n).shape == getattr(out_c, n).shape for n in OUTPUTS)
        for name in OUTPUTS:
            row[f"ablation_rel_{name}"] = normwise(getattr(out_p, name), getattr(out_c, name)) if same else math.inf
        row["ablation_identical"] = bool(same and all(torch.equal(getattr(out_p, n), getattr(out_c, n))
                                                      for n in OUTPUTS) and bool(ok_p) == bool(ok_c))
        rows.append(row)
        del out_p, out_c
        return out_e, ok_e

    bs.precise_sigmoid = comparer
    try:
        bs.split_bounds(net.layers, img, eps, S, seed=SEED, chunk=net.chunk)
    finally:
        bs.precise_sigmoid = EAGER_SIGMOID
    return rows, boxes


def sample_boxes(diff, diff_w, key):
    """The pre-registered sample of one (network, image, layer): indices and the reason each
    was chosen. diff = |D_compiled - D_eager|, diff_w = diff / W_compiled (numpy)."""
    n = diff.size
    idx = np.arange(n)
    taken = np.zeros(n, dtype=bool)
    chosen, reasons = [], []
    for values, reason in ((diff, "top_abs"), (diff_w, "top_over_W")):
        values = np.where(np.isnan(values), -np.inf, values)
        order = np.lexsort((idx, -values))  # largest first, ties by box index
        picked = order[~taken[order]][:D_SAMPLE_TOP]
        taken[picked] = True
        chosen += picked.tolist()
        reasons += [reason] * len(picked)
    rest = idx[~taken]
    rng = np.random.default_rng([D_SAMPLE_SEED] + list(key))
    picked = np.sort(rng.choice(rest, size=min(D_SAMPLE_RANDOM, rest.size), replace=False))
    chosen += picked.tolist()
    reasons += ["random"] * len(picked)
    return chosen, reasons


def _shortfall(abc, box, D, i):
    """D_true by mpmath and the shortfall of D below it for box i, as in G1 of
    compare_lstsq_solvers.py: absolute, over W = |A| r_x + |B| r_y + D, and over D_true."""
    A, B, C = abc[i].tolist()
    lx, ux, ly, uy = (t[i].item() for t in box)
    d = D[i].item()
    w = abs(A) * (ux - lx) / 2 + abs(B) * (uy - ly) / 2 + d
    dt = true_max_error(A, B, C, lx, ux, ly, uy)
    out = dict(A=A, B=B, C=C, lx=lx, ux=ux, ly=ly, uy=uy, D=d, W=w, D_true=dt)
    if math.isnan(dt):
        out.update(shortfall_abs=math.nan, shortfall_over_W=math.nan, shortfall_over_D=math.nan)
    else:
        gap = max(0.0, dt - d)
        out.update(shortfall_abs=gap, shortfall_over_W=gap / w if w > 0 else gap,
                   shortfall_over_D=gap / dt if dt > 0 else 0.0)
    return out


def d_contract_rows(net, network_index, image_id, boxes):
    """mpmath evaluation of the sample of one (network, image) at the contract configuration."""
    rows = []
    for layer in sorted({b["layer"] for b in boxes}):
        parts = [b for b in boxes if b["layer"] == layer]  # chunks in pass order
        cat = {k: torch.cat([b[k] for b in parts]) for k in ("D_c", "D_e", "abc_c", "abc_e")}
        for k in ("box_c", "box_e"):
            cat[k] = [torch.cat([b[k][j] for b in parts]) for j in range(4)]
        A, B = cat["abc_c"][:, 0], cat["abc_c"][:, 1]
        lx, ux, ly, uy = cat["box_c"]
        W = A.abs() * (ux - lx) / 2 + B.abs() * (uy - ly) / 2 + cat["D_c"]
        diff = (cat["D_c"] - cat["D_e"]).abs()
        diff_w = torch.where(W > 0, diff / W, diff)
        chosen, reasons = sample_boxes(diff.numpy(), diff_w.numpy(), (network_index, image_id, layer))
        for i, reason in zip(chosen, reasons):
            row = dict(network=net.name, image=image_id, layer=layer, box=i, reason=reason,
                       D_diff=diff[i].item())
            for side, tag in (("compiled", "c"), ("eager", "e")):
                for k, v in _shortfall(cat[f"abc_{tag}"], cat[f"box_{tag}"], cat[f"D_{tag}"], i).items():
                    row[f"{side}_{k}"] = v
            rows.append(row)
    return rows


def gate_config(gate, net, S, trackers, sampled, eps_indices, device, bound_rows, tensor_rows,
                contract_s, network_index, contract_rows):
    where = dict(network=net.name, S=S)
    comp = trackers[COMPILED_SIGMOID]
    calls_per_pass = net.sigmoid_layers * net.chunks(S)
    set_mode(COMPILED_SIGMOID, trackers)
    gate.check("G2 plane fit inside is the eager fit_planes (no nested compile)",
               plane_fit_inside(device) is planar_fit.fit_planes and not bs.USE_COMPILED_FIT, **where)
    contract_here = S == contract_s and D_CONTRACT_EPS in eps_indices
    candidate, ablation = comp.with_diagnostics, trackers[ABLATION].fn
    my_contract = []
    for image_id, img in net.images:
        for e in eps_indices:
            eps = EPS_SCHEDULE[e]
            w = dict(where, image=image_id, eps_idx=e)
            comp.tag = w
            seq = run_pass(SEQUENTIAL, net, img, eps, S).double().cpu()
            set_mode("batched_A2_eager", trackers)
            np.random.seed(0)
            eager = run_pass("batched_A2_eager", net, img, eps, S).double().cpu()
            st_eager = _numpy_state()
            set_mode(COMPILED_SIGMOID, trackers)
            np.random.seed(0)
            calls0 = comp.calls
            got = run_pass(COMPILED_SIGMOID, net, img, eps, S).double().cpu()
            st_comp = _numpy_state()
            calls = comp.calls - calls0
            again = run_pass(COMPILED_SIGMOID, net, img, eps, S).double().cpu()

            per_split = ((got - eager).abs() / eager.abs().clamp_min(1e-300)).max().item()
            per_split_seq = ((got - seq).abs() / seq.abs().clamp_min(1e-300)).max().item()
            final_seq = rel(got.max().item(), seq.max().item())
            gate.check("G2 every sigmoid call through the compiled callable", calls == calls_per_pass,
                       calls, calls_per_pass, **w)
            gate.check("G3 NumPy state after the pass as eager A2", _same_state(st_comp, st_eager), **w)
            gate.check("G4 same seed, same bits", torch.equal(got, again), **w)
            gate.check("G6 per-split bound vs eager A2", per_split <= RTOL_BOUND, per_split, RTOL_BOUND, **w)
            gate.check("G7 final bound vs sequential G", final_seq <= RTOL_BOUND, final_seq, RTOL_BOUND, **w)
            gate.check("G8 final bound >= sampled autograd maximum", got.max().item() >= sampled[(image_id, e)],
                       got.max().item(), sampled[(image_id, e)], **w)

            with_diag = contract_here and e == D_CONTRACT_EPS
            rows, boxes = compare_layers(net, img, eps, S, candidate, ablation, keep_boxes=with_diag)
            for r in rows:
                tensor_rows.append(dict(w, **r))
            worst = worst_of(r["max_rel"] for r in rows)
            gate.check("G5 rc, rg, dc and the dg generator and zero blocks vs eager (normwise)",
                       worst <= RTOL_TENSOR, worst, RTOL_TENSOR, **w)
            gate.check("G5 shapes and nonzero flag as eager",
                       all(r["shapes_ok"] and r["flag_eager"] == r["flag_compiled"] for r in rows), **w)
            copies = sum(r["copy_identical"] for r in rows)
            gate.check("G5 copy: the candidate run eagerly is bit-identical to batched_splits.precise_sigmoid",
                       copies == len(rows), copies, len(rows), **w)
            d_out = sum(r["d_output_identical"] for r in rows)
            gate.check("G5 D output: the D the compiled candidate returns equals the D it propagates",
                       d_out == len(rows), d_out, len(rows), **w)
            if with_diag:
                t0 = time.perf_counter()
                found = d_contract_rows(net, network_index, image_id, boxes)
                my_contract += found
                print(f"  D contract {net.name} image {image_id}: {len(found)} boxes, "
                      f"{time.perf_counter() - t0:.1f} s", flush=True)
            bound_rows.append(dict(w, final_sequential_G=seq.max().item(), final_eager_A2=eager.max().item(),
                                   final_compiled=got.max().item(), sampled_autograd=sampled[(image_id, e)],
                                   max_rel_per_split_vs_eager_A2=per_split,
                                   max_rel_per_split_vs_sequential_G=per_split_seq,
                                   rel_final_vs_sequential_G=final_seq,
                                   eager_A2_final_vs_sequential_G=rel(eager.max().item(), seq.max().item()),
                                   compiled_calls_per_pass=calls, bits_equal=bool(torch.equal(got, again)),
                                   numpy_state_equal=_same_state(st_comp, st_eager)))
            print(f"  {net.name} S={S} image {image_id} eps_idx {e}: per-split {per_split:.2e}, "
                  f"final vs sequential {final_seq:.2e}, tensors {worst:.2e}", flush=True)

    if contract_here:
        def finite(key):
            return [r[key] for r in my_contract if not math.isnan(r[key])]
        comp_sf, eager_sf = finite("compiled_shortfall_over_W"), finite("eager_shortfall_over_W")
        worst_c = max(comp_sf) if comp_sf else math.nan
        worst_e = max(eager_sf) if eager_sf else 0.0
        limit = D_CONTRACT_FACTOR * max(D_CONTRACT_FLOOR, worst_e)
        not_converged = len(my_contract) - len(comp_sf)
        gate.check("G5 D: worst compiled shortfall / W <= 10 * max(1e-12, worst eager shortfall / W)",
                   bool(comp_sf) and worst_c <= limit, worst_c, limit,
                   detail=f"eager {worst_e:.3e}; {len(my_contract)} boxes, {not_converged} not converged; "
                          f"worst absolute {max(finite('compiled_shortfall_abs'), default=math.nan):.3e}, "
                          f"over D {max(finite('compiled_shortfall_over_D'), default=math.nan):.3e}",
                   network=net.name, S=S, eps_idx=D_CONTRACT_EPS)
        contract_rows += my_contract

    img0, eps0 = net.images[0][1], EPS_SCHEDULE[eps_indices[0]]
    set_mode(COMPILED_SIGMOID, trackers)
    c = profile_counts(lambda: run_pass(COMPILED_SIGMOID, net, img0, eps0, S), device)
    gate.check("G2 profiled pass: one compiled region per sigmoid call", c["compiled_regions"] == calls_per_pass,
               c["compiled_regions"], calls_per_pass, **where)
    gate.check("G2 profiled pass: no lstsq", c["lstsq"] == 0, c["lstsq"], 0, **where)
    graphs = sum(ev["new_graphs"] for ev in comp.events if ev.get("network") == net.name and ev.get("S") == S)
    gate.check("G2 at least one graph compiled", graphs > 0, graphs, 1, **where)
    gate.check("G2 one graph serves every sigmoid layer", graphs == 1 and len(c["region_ids"].split()) == 1,
               graphs, 1, detail=f"compile ids in the profiled pass: {c['region_ids']}", **where)
    gate.check("G2 planar_fit.compile_fallback empty", not planar_fit.compile_fallback,
               detail=json.dumps(planar_fit.compile_fallback), **where)
    restore()


def cmd_gate(args):
    device = setup(args)
    networks, splits, n_img, eps_indices = grid(args)
    gate = Gate()
    bound_rows, tensor_rows, events, contract_rows = [], [], [], []
    for name in networks:
        net = Net(name, device, n_img, args.big_chunk)
        net64 = net.net64()
        sampled = {(image_id, e): sampled_max_derivative(net64, img.cpu().double(), EPS_SCHEDULE[e],
                                                         AUTOGRAD_SAMPLES)
                   for image_id, img in net.images for e in eps_indices}
        for S in splits:
            mark(f"network={name} S={S}")
            print(f"gate: {name} S={S}", flush=True)
            fresh_compiled_state()
            trackers = make_trackers()
            try:
                gate_config(gate, net, S, trackers, sampled, eps_indices, device, bound_rows, tensor_rows,
                            contract_split(args), NETWORKS.index(name), contract_rows)
                gate.check("G1 fullgraph compile and run", True, network=name, S=S)
            except Exception as e:
                traceback.print_exc()
                gate.check("G1 fullgraph compile and run", False, detail=short(e), network=name, S=S)
            finally:
                restore()
            events += [dict(ev, phase="grid") for ev in trackers[COMPILED_SIGMOID].events]

        if name == "3layer":
            # unseeded noise: no reference to match, so soundness and the NumPy draws
            for S in (1, 16):
                mark(f"network={name} S={S} unseeded")
                fresh_compiled_state()
                trackers = make_trackers()
                comp = trackers[COMPILED_SIGMOID]
                where = dict(network=name, S=S, phase="unseeded")
                try:
                    for image_id, img in net.images[:2]:
                        for e in eps_indices:
                            comp.tag = dict(where, image=image_id, eps_idx=e)
                            set_mode(COMPILED_SIGMOID, trackers)
                            lc = bs.lipschitz_bound(net.layers, img, EPS_SCHEDULE[e], S, seed=None)
                            gate.check("G8 unseeded bound >= sampled autograd maximum", lc >= sampled[(image_id, e)],
                                       lc, sampled[(image_id, e)], image=image_id, eps_idx=e, **where)
                    img0 = net.images[0][1]
                    set_mode("batched_A2_eager", trackers)
                    np.random.seed(11)
                    bs.split_bounds(net.layers, img0, EPS_SCHEDULE[3], S, seed=None)
                    st_eager = _numpy_state()
                    set_mode(COMPILED_SIGMOID, trackers)
                    np.random.seed(11)
                    bs.split_bounds(net.layers, img0, EPS_SCHEDULE[3], S, seed=None)
                    gate.check("G3 unseeded: NumPy state after the pass as eager A2",
                               _same_state(_numpy_state(), st_eager), **where)
                    gate.check("G1 fullgraph compile and run", True, **where)
                except Exception as e:
                    traceback.print_exc()
                    gate.check("G1 fullgraph compile and run", False, detail=short(e), **where)
                finally:
                    restore()
                events += [dict(ev, phase="unseeded") for ev in comp.events]
        del net

    write_csv(os.path.join(args.out, "gate_checks.csv"), gate.checks)
    write_csv(os.path.join(args.out, "gate_bounds.csv"), bound_rows)
    write_csv(os.path.join(args.out, "gate_tensors.csv"), tensor_rows)
    write_csv(os.path.join(args.out, "gate_compile_events.csv"), events)
    write_csv(os.path.join(args.out, "gate_d_contract.csv"), contract_rows)
    failed = [c for c in gate.checks if not c["ok"]]
    summary = dict(checks=len(gate.checks), failed=len(failed), failed_checks=failed[:50],
                   worst=dict(
                       per_split_vs_eager_A2=max((r["max_rel_per_split_vs_eager_A2"] for r in bound_rows), default=None),
                       final_vs_sequential_G=max((r["rel_final_vs_sequential_G"] for r in bound_rows), default=None),
                       tensors=worst_of(r["max_rel"] for r in tensor_rows),
                       dg_D_normwise_recorded=max((r.get("rel_dg_D", -math.inf) for r in tensor_rows), default=None),
                       d_contract_boxes=len(contract_rows),
                       d_contract_compiled_over_W=max((r["compiled_shortfall_over_W"] for r in contract_rows
                                                       if not math.isnan(r["compiled_shortfall_over_W"])),
                                                      default=None),
                       d_contract_eager_over_W=max((r["eager_shortfall_over_W"] for r in contract_rows
                                                    if not math.isnan(r["eager_shortfall_over_W"])),
                                                   default=None)))
    with open(os.path.join(args.out, "gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2, default=str))
    print("GATE PASSED" if not failed else f"GATE FAILED: {len(failed)} of {len(gate.checks)} checks")
    return 0 if not failed else 1


# ---------------------------------------------------------------- shared callable

def cmd_shared(args):
    """One compiled callable for the whole grid, default recompile limit, two rounds."""
    device = setup(args)
    networks, splits, n_img, eps_indices = grid(args)
    limit = getattr(torch._dynamo.config, "recompile_limit", None)
    if limit is None:
        limit = getattr(torch._dynamo.config, "cache_size_limit", None)
    nets = {name: Net(name, device, 1, args.big_chunk) for name in networks}
    rows, events = [], []
    for mode in (COMPILED_SIGMOID, COMPILED_FIT):
        fresh_compiled_state()
        trackers = make_trackers()
        tr = trackers[mode]
        code_fn = precise_sigmoid_instrumented if mode == COMPILED_SIGMOID else planar_fit.fit_planes
        stopped = False
        for round_ in (1, 2):
            for name in networks:
                net = nets[name]
                for S in splits:
                    mark(f"shared mode={mode} round={round_} network={name} S={S}")
                    tr.tag = dict(round=round_, network=name, S=S)
                    g0, fb0 = unique_graphs(), dict(planar_fit.compile_fallback)
                    set_mode(mode, trackers)
                    t0 = time.perf_counter()
                    try:
                        run_pass(mode, net, net.images[0][1], EPS_SCHEDULE[eps_indices[0]], S)
                        sync(device)
                        status = "ok"
                    except Exception as e:
                        traceback.print_exc()
                        status = short(e)
                        stopped = True
                    finally:
                        restore()
                    row = dict(mode=mode, round=round_, network=name, S=S, status=status,
                               seconds=time.perf_counter() - t0, new_graphs=unique_graphs() - g0,
                               graphs_total=unique_graphs(), cache_entries=cache_entries(code_fn),
                               recompile_limit=limit,
                               new_fallback=json.dumps({k: v for k, v in planar_fit.compile_fallback.items()
                                                        if k not in fb0}))
                    rows.append(row)
                    print(row, flush=True)
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break
        events += tr.events
    write_csv(os.path.join(args.out, "shared_callable.csv"), rows)
    write_csv(os.path.join(args.out, "shared_compile_events.csv"), events)
    return 0


# ---------------------------------------------------------------- start (cold / warm)

def cmd_start(args):
    device = setup(args)
    networks, splits, n_img, eps_indices = grid(args)
    repeats = 2 if args.smoke else REPEATS
    mode, S = args.mode, args.splits
    assert mode in COMPILED_MODES, mode
    net = Net(args.network, device, n_img, args.big_chunk)
    img0, eps0 = net.images[0][1], EPS_SCHEDULE[eps_indices[0]]
    trackers = make_trackers()
    # CUDA, cuBLAS and the allocator start with an eager A2 pass, outside the measurement
    set_mode("batched_A2_eager", trackers)
    warmup_ms, _ = timed_pass("batched_A2_eager", net, img0, eps0, S, device)
    restore()

    fresh_compiled_state()
    trackers = make_trackers()
    tr = trackers[mode]
    set_mode(mode, trackers)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    rows = []
    sync(device)
    t0 = time.perf_counter()
    for image_id, img in net.images:
        for e in eps_indices:
            tr.tag = dict(image=image_id, eps_idx=e)
            ms, b = timed_pass(mode, net, img, EPS_SCHEDULE[e], S, device)
            rows.append(dict(phase="first_sweep", image=image_id, eps_idx=e, rep=0, ms=ms, bound=b))
    sync(device)
    e2e_s = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None

    torch.compiler.set_stance("fail_on_recompile")
    try:
        for image_id, img in net.images:
            for e in eps_indices:
                for rep in range(1, repeats + 1):
                    ms, b = timed_pass(mode, net, img, EPS_SCHEDULE[e], S, device)
                    rows.append(dict(phase="steady", image=image_id, eps_idx=e, rep=rep, ms=ms, bound=b))
    finally:
        torch.compiler.set_stance("default")
        restore()

    medians = {}
    for r in rows:
        if r["phase"] == "steady":
            medians.setdefault((r["image"], r["eps_idx"]), []).append(r["ms"])
    steady_sum_s = sum(statistics.median(v) for v in medians.values()) / 1000.0
    result = dict(network=args.network, S=S, mode=mode, cache=args.cache, big_chunk=net.chunk,
                  cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                  import_s=T_IMPORTED - T_START, warmup_eager_ms=warmup_ms,
                  first_pass_ms=rows[0]["ms"], e2e_s=e2e_s, steady_sum_s=steady_sum_s,
                  overhead_s=e2e_s - steady_sum_s,
                  steady_median_ms=statistics.median(r["ms"] for r in rows if r["phase"] == "steady"),
                  graphs=sum(ev["new_graphs"] for ev in tr.events),
                  compile_host_s=sum(ev["host_s"] for ev in tr.events),
                  calls=tr.calls, fallback=dict(planar_fit.compile_fallback),
                  peak_vram_mb_first_sweep=peak, process_s=time.perf_counter() - T_START,
                  events=tr.events, rows=rows)
    os.makedirs(os.path.join(args.out, "start"), exist_ok=True)
    path = os.path.join(args.out, "start", f"{args.network}_S{S}_{mode}_{args.cache}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"RESULT {args.network} S={S} {mode} {args.cache}: e2e {e2e_s:.2f} s, overhead {result['overhead_s']:.2f} s, "
          f"first pass {result['first_pass_ms']:.1f} ms, steady {result['steady_median_ms']:.2f} ms, "
          f"graphs {result['graphs']}, fallback {result['fallback'] or 'none'}", flush=True)
    return 0 if not planar_fit.compile_fallback else 1


# ---------------------------------------------------------------- steady state

def layer_kernels(net, S, trackers, img, eps, device):
    """Kernels of one precise_sigmoid call per layer and batched mode, on the inputs of the
    first chunk of an eager A2 pass."""
    rows = []
    calls = [0]

    def recorder(z, noise):
        k = calls[0]
        calls[0] += 1
        if k < net.sigmoid_layers:
            for mode in BATCHED_MODES:
                set_flags(mode)
                fn = trackers[mode]
                c = profile_counts(lambda: fn(z, noise), device)
                rows.append(dict(network=net.name, S=S, layer=k + 1, mode=mode, S_chunk=z.rg.shape[0],
                                 G0=z.rg.shape[1], N=z.rg.shape[2], **c))
            set_flags("batched_A2_eager")
        return EAGER_SIGMOID(z, noise)

    set_flags("batched_A2_eager")
    bs.precise_sigmoid = recorder
    try:
        bs.split_bounds(net.layers, img, eps, S, seed=SEED, chunk=net.chunk)
    finally:
        restore()
    return rows


def cmd_steady(args):
    device = setup(args)
    networks, splits, n_img, eps_indices = grid(args)
    repeats = 2 if args.smoke else REPEATS
    times, firsts, kernels, memory, checks = [], [], [], [], []
    for name in networks:
        net = Net(name, device, n_img, args.big_chunk)
        img0, eps0 = net.images[0][1], EPS_SCHEDULE[eps_indices[0]]
        for S in splits:
            mark(f"steady network={name} S={S}")
            print(f"steady: {name} S={S}", flush=True)
            fresh_compiled_state()
            trackers = make_trackers()
            calls_per_pass = net.sigmoid_layers * net.chunks(S)
            try:
                for mode in STEADY_MODES:  # first pass: compiles the compiled modes (warm disk cache)
                    set_mode(mode, trackers)
                    ms, b = timed_pass(mode, net, img0, eps0, S, device)
                    firsts.append(dict(network=name, S=S, mode=mode, first_ms=ms, bound=b,
                                       graphs=sum(ev["new_graphs"] for ev in trackers[mode].events)
                                       if mode in trackers else 0))
                restore()
                ref = {}
                torch.compiler.set_stance("fail_on_recompile")
                try:
                    for image_id, img in net.images:
                        for e in eps_indices:
                            for rep in range(repeats):
                                k = rep % len(STEADY_MODES)
                                for pos, mode in enumerate(STEADY_MODES[k:] + STEADY_MODES[:k]):
                                    set_mode(mode, trackers)
                                    ms, b = timed_pass(mode, net, img, EPS_SCHEDULE[e], S, device)
                                    times.append(dict(network=name, S=S, mode=mode, image=image_id, eps_idx=e,
                                                      rep=rep, position=pos, ms=ms, bound=b))
                                    if mode == SEQUENTIAL and rep == 0:
                                        ref[(image_id, e)] = b
                finally:
                    torch.compiler.set_stance("default")
                    restore()
                for t in times:
                    if t["network"] == name and t["S"] == S:
                        t["rel_vs_sequential_G"] = rel(t["bound"], ref[(t["image"], t["eps_idx"])])
                worst = {m: max(t["rel_vs_sequential_G"] for t in times
                                if t["network"] == name and t["S"] == S and t["mode"] == m) for m in STEADY_MODES}
                for m in STEADY_MODES:
                    checks.append(dict(check="final bound vs sequential G", network=name, S=S, mode=m, formal=m in MODES,
                                       value=worst[m], limit=RTOL_BOUND, ok=worst[m] <= RTOL_BOUND))

                for mode in STEADY_MODES:
                    set_mode(mode, trackers)
                    peak, above = peak_vram(lambda: run_pass(mode, net, img0, eps0, S), device)
                    restore()
                    memory.append(dict(network=name, S=S, mode=mode, peak_mb=peak, peak_above_base_mb=above))

                for mode in STEADY_MODES:
                    if mode == SEQUENTIAL and S > 1:
                        continue  # S times the S = 1 count; a long trace for nothing
                    set_mode(mode, trackers)
                    c = profile_counts(lambda: run_pass(mode, net, img0, eps0, S), device)
                    restore()
                    kernels.append(dict(network=name, S=S, layer="pass", mode=mode, **c))
                    want_regions = calls_per_pass if mode in COMPILED_MODES + [ABLATION] else 0
                    checks.append(dict(check="compiled regions per pass", network=name, S=S, mode=mode,
                                       formal=mode in MODES,
                                       value=c["compiled_regions"], limit=want_regions,
                                       ok=c["compiled_regions"] == want_regions))
                    uses_gelsd = mode in (SEQUENTIAL, "batched_G")
                    checks.append(dict(check="lstsq used iff gelsd mode", network=name, S=S, mode=mode,
                                       formal=mode in MODES,
                                       value=c["lstsq"], limit=">0" if uses_gelsd else 0,
                                       ok=(c["lstsq"] > 0) == uses_gelsd))
                kernels += layer_kernels(net, S, trackers, img0, eps0, device)
                checks.append(dict(check="planar_fit.compile_fallback empty", network=name, S=S, mode=COMPILED_FIT,
                                   value=json.dumps(planar_fit.compile_fallback), limit="{}",
                                   ok=not planar_fit.compile_fallback))
            except Exception as e:
                traceback.print_exc()
                checks.append(dict(check="configuration ran", network=name, S=S, value=short(e), ok=False))
            finally:
                torch.compiler.set_stance("default")
                restore()
        del net
    write_csv(os.path.join(args.out, "steady_times.csv"), times)
    write_csv(os.path.join(args.out, "steady_first_pass.csv"), firsts)
    write_csv(os.path.join(args.out, "steady_kernels.csv"), kernels)
    write_csv(os.path.join(args.out, "steady_memory.csv"), memory)
    write_csv(os.path.join(args.out, "steady_checks.csv"), checks)
    failed = [c for c in checks if not c["ok"]]
    for c in failed:
        print("  FAIL", c, flush=True)
    print("STEADY CHECKS PASSED" if not failed else f"STEADY CHECKS FAILED: {len(failed)}")
    return 0 if not failed else 1


# ---------------------------------------------------------------- compiler logs

def cmd_logs(args):
    """One pass of one configuration; run with TORCH_LOGS set before the process starts and
    an empty cache directory, then copy the generated code out of the cache."""
    device = setup(args)
    networks, splits, n_img, eps_indices = grid(args)
    net = Net(args.network, device, 1, args.big_chunk)
    fresh_compiled_state()
    trackers = make_trackers()
    set_mode(args.mode, trackers)
    mark(f"logs network={args.network} S={args.splits} mode={args.mode}")
    try:
        run_pass(args.mode, net, net.images[0][1], EPS_SCHEDULE[eps_indices[0]], args.splits)
        sync(device)
    finally:
        restore()
    tag = f"{args.network}_S{args.splits}_{args.mode}"
    dest = os.path.join(args.out, "kernel_code", tag)
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    copied = 0
    for path in glob.glob(os.path.join(cache, "**", "*.py"), recursive=True):
        target = os.path.join(dest, os.path.relpath(path, cache))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    tr = trackers[args.mode]
    print(json.dumps(dict(config=tag, graphs=sum(ev["new_graphs"] for ev in tr.events), events=tr.events,
                          code_files=copied, fallback=planar_fit.compile_fallback), default=str), flush=True)
    return 0


# ---------------------------------------------------------------- summary and criteria

SYMBOL = re.compile(r"create_symbol (\w+) = (\d+) for L\['(\w+)'\]\.?(\w*)\.size\(\)\[(\d)\]")
DIM_NAMES = {("z", "rg"): ("S", "G0", "N"), ("z", "dg"): ("S", "G0", "N"), ("z", "rc"): ("S", "N"),
             ("z", "dc"): ("S", "N"), ("noise", ""): ("S", "N")}


def log_sections(path):
    """{config marker: [lines]} of a stderr log written with mark()."""
    sections, current = {}, None
    if not os.path.exists(path):
        return sections
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("### CONFIG "):
                current = line[len("### CONFIG "):].strip()
                sections[current] = []
            elif current is not None:
                sections[current].append(line.rstrip("\n"))
    return sections


def recompile_summary(out):
    events = read_csv(os.path.join(out, "gate_compile_events.csv"))
    sections = log_sections(os.path.join(out, "logs", "gate.stderr.log"))
    rows = []
    for key, lines in sections.items():
        fields = dict(kv.split("=", 1) for kv in key.split() if "=" in kv)
        unseeded = key.endswith("unseeded")
        evs = [e for e in events if e["network"] == fields.get("network") and str(e["S"]) == fields.get("S")
               and (e.get("phase") == "unseeded") == unseeded]
        dynamic = set()
        for line in lines:
            m = SYMBOL.search(line)
            if m:
                names = DIM_NAMES.get((m.group(3), m.group(4)))
                if names and int(m.group(5)) < len(names):
                    dynamic.add(names[int(m.group(5))])
        reasons = [line.split("[__recompiles]")[-1].strip() for line in lines if "[__recompiles]" in line]
        reasons = [r for r in reasons if r.startswith("- ")]
        text = "\n".join(lines)
        rows.append(dict(network=fields.get("network"), S=fields.get("S"), phase="unseeded" if unseeded else "grid",
                         graphs=sum(int(e["new_graphs"]) for e in evs),
                         compile_s=round(sum(float(e["host_s"]) for e in evs), 2),
                         triggering_shapes=" ".join(f"({e['S_chunk']},{e['G0']},{e['N']})" for e in evs),
                         dynamic_dims=",".join(sorted(dynamic)),
                         recompile_reasons=" | ".join(dict.fromkeys(reasons))[:600],
                         limit_hit=("hit config.recompile_limit" in text or "hit config.cache_size_limit" in text
                                    or "FailOnRecompileLimitHit" in text)))
    return rows


def pooled(times):
    """{(network, S, mode): [ms, ...]} of the steady process."""
    out = {}
    for t in times:
        out.setdefault((t["network"], int(t["S"]), t["mode"]), []).append(float(t["ms"]))
    return out


def cmd_summarize(args):
    out = args.out
    networks, splits, n_img, eps_indices = grid(args)
    report = dict(C1_correctness=None)  # filled below, first in criteria.json

    gate = json.load(open(os.path.join(out, "gate_summary.json"))) if os.path.exists(
        os.path.join(out, "gate_summary.json")) else None
    gate_ok = gate is not None and gate["failed"] == 0
    steady_checks = read_csv(os.path.join(out, "steady_checks.csv"))
    # the ablation's checks are recorded with formal=False and stay out of C1
    steady_failed = [c for c in steady_checks if c.get("ok") != "True" and c.get("formal") != "False"]

    rec = recompile_summary(out)
    write_csv(os.path.join(out, "recompiles.csv"), rec)

    times = read_csv(os.path.join(out, "steady_times.csv"))
    samples = pooled(times)
    steady = []
    for (name, S, mode), v in sorted(samples.items(), key=lambda kv: (networks.index(kv[0][0])
                                                                       if kv[0][0] in networks else 99,
                                                                       kv[0][1], STEADY_MODES.index(kv[0][2]))):
        steady.append(dict(network=name, S=S, mode=mode, n=len(v), median_ms=statistics.median(v),
                           min_ms=min(v), max_ms=max(v)))
    med = {(r["network"], r["S"], r["mode"]): r["median_ms"] for r in steady}
    for r in steady:
        key = (r["network"], r["S"])
        for ref in ("batched_G", "batched_A2_eager", COMPILED_FIT, COMPILED_SIGMOID):
            if (key + (ref,)) in med:
                r[f"speedup_vs_{ref}"] = med[key + (ref,)] / r["median_ms"]
    write_csv(os.path.join(out, "steady_summary.csv"), steady)

    missing = [] if steady_checks else ["steady checks"]
    for name in networks:
        for S in splits:
            for mode in MODES:
                if (name, S, mode) not in med:
                    missing.append(f"steady {name} S={S} {mode}")

    c2 = []
    for name in networks:
        for S in splits:
            a, b = med.get((name, S, COMPILED_SIGMOID)), med.get((name, S, COMPILED_FIT))
            if a is not None and b is not None:
                c2.append(dict(network=name, S=S, ratio=a / b, ok=a <= C2_MAX_SLOWDOWN * b))
    report["C2_not_slower_than_compiled_fit"] = dict(
        rule=f"median(compiled sigmoid) <= {C2_MAX_SLOWDOWN} * median(compiled fit) in every (network, S)",
        passed=bool(c2) and all(r["ok"] for r in c2), cells=c2)

    c3 = []
    for name in [n for n in SMALL_NETWORKS if n in networks]:
        a, b = med.get((name, splits[0], COMPILED_SIGMOID)), med.get((name, splits[0], COMPILED_FIT))
        if a is not None and b is not None:
            c3.append(dict(network=name, S=splits[0], speedup=b / a, ok=b / a >= C3_MIN_SPEEDUP))
    report["C3_gain_small_networks_S1"] = dict(
        rule=f"median(compiled fit) / median(compiled sigmoid) >= {C3_MIN_SPEEDUP} on the small networks at S = 1",
        passed=bool(c3) and len(c3) == len([n for n in SMALL_NETWORKS if n in networks]) and all(r["ok"] for r in c3),
        cells=c3)

    seq_ref = {}
    for t in times:
        if t["mode"] == SEQUENTIAL:
            seq_ref.setdefault((t["network"], int(t["S"]), int(t["image"]), int(t["eps_idx"])), float(t["bound"]))
    starts, start_checks = [], []
    for path in sorted(glob.glob(os.path.join(out, "start", "*.json"))):
        r = json.load(open(path))
        starts.append({k: v for k, v in r.items() if k not in ("rows", "events")})
        where = dict(network=r["network"], S=r["S"], mode=r["mode"], cache=r["cache"])
        start_checks.append(dict(check="no compiled-fit fallback", **where, value=json.dumps(r["fallback"]),
                                 limit="{}", ok=not r["fallback"]))
        start_checks.append(dict(check="at least one graph compiled", **where, value=r["graphs"], limit=1,
                                 ok=r["graphs"] > 0))
        diffs = [rel(row["bound"], seq_ref[key]) for row in r["rows"]
                 for key in [(r["network"], int(r["S"]), int(row["image"]), int(row["eps_idx"]))] if key in seq_ref]
        if diffs:
            start_checks.append(dict(check="bounds vs sequential G", **where, value=worst_of(diffs),
                                     limit=RTOL_BOUND, ok=worst_of(diffs) <= RTOL_BOUND))
    write_csv(os.path.join(out, "start_summary.csv"), starts)
    write_csv(os.path.join(out, "start_checks.csv"), start_checks)
    start_failed = [c for c in start_checks if not c["ok"]]
    report["C1_correctness"] = dict(
        passed=gate_ok and bool(steady_checks) and not steady_failed and not start_failed,
        gate=dict(ran=gate is not None, passed=gate_ok, failed=None if gate is None else gate["failed"]),
        steady=dict(ran=bool(steady_checks), checks=len(steady_checks), failed=len(steady_failed),
                    failed_checks=steady_failed[:50]),
        start=dict(checks=len(start_checks), failed=len(start_failed), failed_checks=start_failed[:50]))
    have = {(r["network"], int(r["S"]), r["mode"], r["cache"]) for r in starts}
    for name in networks:
        for S in splits:
            for mode in COMPILED_MODES:
                for cache in ("cold", "warm"):
                    if (name, S, mode, cache) not in have:
                        missing.append(f"start {name} S={S} {mode} {cache}")

    be = []
    for r in starts:
        key = (r["network"], int(r["S"]))
        t_mode = med.get(key + (r["mode"],))
        t_g = med.get(key + ("batched_G",))
        t_a2 = med.get(key + ("batched_A2_eager",))
        if None in (t_mode, t_g, t_a2):
            continue
        row = dict(network=r["network"], S=int(r["S"]), mode=r["mode"], cache=r["cache"],
                   overhead_s=r["overhead_s"], e2e_s=r["e2e_s"], first_pass_ms=r["first_pass_ms"],
                   steady_ms=t_mode)
        for ref_name, t_ref in (("batched_G", t_g), ("best_without_compile", min(t_g, t_a2))):
            save = t_ref - t_mode
            row[f"saves_ms_vs_{ref_name}"] = save
            row[f"break_even_vs_{ref_name}"] = (max(r["overhead_s"], 0.0) * 1000.0 / save) if save > 0 else math.inf
        be.append(row)
    write_csv(os.path.join(out, "break_even.csv"), be)

    def small_s1(cache):
        cells = [b for b in be if b["mode"] == COMPILED_SIGMOID and b["cache"] == cache
                 and b["network"] in SMALL_NETWORKS and b["S"] == splits[0]]
        return cells, len(cells) == len([n for n in SMALL_NETWORKS if n in networks]) and all(
            b["break_even_vs_best_without_compile"] <= BREAK_EVEN_PASSES for b in cells)

    warm_cells, warm_ok = small_s1("warm")
    cold_cells, cold_ok = small_s1("cold")
    report["C4_break_even_small_networks_S1"] = dict(
        rule=f"break-even of the compiled sigmoid against the best path without compilation "
             f"<= {BREAK_EVEN_PASSES} passes, small networks at S = 1",
        warm=dict(passed=warm_ok, cells=[{k: b[k] for k in ("network", "overhead_s",
                                                               "break_even_vs_best_without_compile")}
                                         for b in warm_cells]),
        cold=dict(passed=cold_ok, cells=[{k: b[k] for k in ("network", "overhead_s",
                                                               "break_even_vs_best_without_compile")}
                                         for b in cold_cells]))

    if not gate_ok or steady_failed or start_failed:
        outcome = "stop"
    elif missing:
        outcome = "incomplete"
    elif not (report["C2_not_slower_than_compiled_fit"]["passed"] and report["C3_gain_small_networks_S1"]["passed"]
              and warm_ok):
        outcome = "opt-in"
    elif not cold_ok:
        outcome = "auto candidate"
    else:
        outcome = "default candidate"
    report["missing"] = missing
    report["outcome"] = outcome
    # informative, outside the outcome: what the diagnostic outputs cost the candidate
    report["ablation_uninstrumented"] = dict(
        note="median(candidate) / median(uninstrumented compiled graph); not part of the outcome",
        cells=[dict(network=name, S=S, ratio=med[(name, S, COMPILED_SIGMOID)] / med[(name, S, ABLATION)])
               for name in networks for S in splits
               if (name, S, COMPILED_SIGMOID) in med and (name, S, ABLATION) in med])
    with open(os.path.join(out, "criteria.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["gate", "shared", "start", "steady", "logs", "summarize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--big-chunk", type=int, default=None)
    ap.add_argument("--network")
    ap.add_argument("--splits", type=int)
    ap.add_argument("--mode", choices=MODES)
    ap.add_argument("--cache", choices=["cold", "warm"])
    ap.add_argument("--smoke", action="store_true", help="tiny grid, for a CPU check of the harness")
    args = ap.parse_args()
    return dict(gate=cmd_gate, shared=cmd_shared, start=cmd_start, steady=cmd_steady, logs=cmd_logs,
                summarize=cmd_summarize)[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
