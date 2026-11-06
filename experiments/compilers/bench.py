"""One measurement of results/compilers_2026-11/PROTOCOL.md: one tool, mode, kernel and size
in this process. Appends one JSON line to --out. Every failure is recorded, not raised.

  python experiments/compilers/bench.py --tool torch --mode default --kernel fit --size 1024 \
      --device cuda --phase cold --out records.jsonl
  python experiments/compilers/bench.py --tool jax --kernel fit --size 1024 --device cuda ...

The cache directory for the cold/warm pair is TORCHINDUCTOR_CACHE_DIR (Inductor) or
--jax-cache-dir (JAX); run_matrix.py sets both and runs the cold process first.
"""
import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
import traceback
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def flat(out):
    return list(out) if isinstance(out, (tuple, list)) else [out]


def compare(got, ref):
    """Largest normwise relative difference over finite entries, and NaN-pattern equality."""
    import numpy as np
    worst, same_nan = 0.0, True
    for g, r in zip(got, ref):
        g, r = np.asarray(g, dtype=np.float64), np.asarray(r, dtype=np.float64)
        same_nan &= bool((np.isnan(g) == np.isnan(r)).all())
        fin = np.isfinite(r) & np.isfinite(g)
        if fin.any():
            scale = np.abs(r[fin]).max()
            worst = max(worst, float(np.abs(g[fin] - r[fin]).max() / scale) if scale else 0.0)
    return worst, same_nan


def scaled_size(kernel, size, factor):
    if factor == 1:
        return size
    if kernel in ("bounds", "sigmoid"):
        return f"{size}:{factor}"
    return str(int(size) * factor)


def count_code_kernels(codes):
    text = "\n".join(codes)
    generated = len(re.findall(r"async_compile\.(?:triton|cpp_pybinding|cpp)\(", text))
    extern = len(re.findall(r"extern_kernels\.\w+\(", text))
    return dict(generated_kernels=generated, extern_kernel_calls=extern)


def hlo_count(text):
    """Instructions of the ENTRY computation that launch work (fusions, library calls...)."""
    skip = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast"}
    entry, n, ops = False, 0, {}
    for line in text.splitlines():
        if line.startswith("ENTRY"):
            entry = True
            continue
        if entry:
            if line.startswith("}"):
                break
            m = re.search(r"=\s*[^\s]+\s+([a-z\-]+)\(", line) or re.search(r"=\s*\([^)]*\)\s+([a-z\-]+)\(", line)
            if m and m.group(1) not in skip:
                n += 1
                ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    return n, ops


def bench_torch(args, rec):
    import torch
    import kernels_torch as kt
    torch.set_default_dtype(torch.float64)
    dev = torch.device(args.device)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    fn, inputs = kt.make(args.kernel, args.size, args.device)
    ref = [t.detach().cpu() for t in flat(fn(*inputs))]
    rec["reference"] = "PyTorch eager, same device, same process"
    if args.mode == "eager":
        f = fn
    else:
        import torch._dynamo as dynamo
        try:
            ex = dynamo.explain(fn)(*inputs)
            rec["graph_count"] = ex.graph_count
            rec["graph_break_count"] = ex.graph_break_count
            rec["break_reasons"] = [str(getattr(b, "reason", b)).splitlines()[0][:300] for b in ex.break_reasons]
        except Exception as e:  # noqa: BLE001
            rec["explain_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        dynamo.reset()
        f = torch.compile(fn, mode=None if args.mode == "default" else args.mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sync(); t0 = time.perf_counter()
        if args.mode == "eager":
            out = f(*inputs)
            codes = []
        else:
            from torch._inductor.utils import run_and_get_code
            out, codes = run_and_get_code(f, *inputs)
        sync(); rec["first_call_s"] = time.perf_counter() - t0
    rec["warnings"] = sorted({str(w.message).splitlines()[0][:200] for w in caught})[:10]
    if args.mode != "eager":
        rec.update(count_code_kernels(codes))
        rec["output_code_chars"] = sum(len(c) for c in codes)
    got = [t.detach().cpu() for t in flat(out)]
    rec["max_rel_vs_eager"], rec["nan_pattern_equal"] = compare(got, ref)
    for _ in range(10):
        f(*inputs)
    sync()
    wall = []
    for _ in range(args.repeats):
        sync(); t0 = time.perf_counter(); f(*inputs); sync(); wall.append(time.perf_counter() - t0)
    rec["steady_wall_ms_median"], rec["steady_wall_ms_min"] = 1e3 * statistics.median(wall), 1e3 * min(wall)
    if dev.type == "cuda":
        ev = []
        for _ in range(args.repeats):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); f(*inputs); b.record(); torch.cuda.synchronize(); ev.append(a.elapsed_time(b))
        rec["steady_cuda_event_ms_median"], rec["steady_cuda_event_ms_min"] = statistics.median(ev), min(ev)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            f(*inputs); torch.cuda.synchronize()
        evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
        rec["cuda_kernels_per_call"] = sum(1 for e in evs if not e.name.startswith(("Memcpy", "Memset")))
        rec["cuda_memcpy_memset_per_call"] = sum(1 for e in evs if e.name.startswith(("Memcpy", "Memset")))
    # dynamic shapes: two more sizes
    if args.mode != "eager" and args.kernel != "sigmoid":
        import torch._dynamo.utils as du
        before = dict(du.counters["stats"])
        dyn = []
        for factor in (2, 3):
            _, inp2 = kt.make(args.kernel, scaled_size(args.kernel, args.size, factor), args.device)
            sync(); t0 = time.perf_counter(); f(*inp2); sync()
            dyn.append(dict(factor=factor, first_call_s=time.perf_counter() - t0))
        after = dict(du.counters["stats"])
        rec["dynamic_shapes"] = dyn
        rec["dynamo_unique_graphs_added"] = after.get("unique_graphs", 0) - before.get("unique_graphs", 0)
    rec["versions"] = dict(torch=torch.__version__, cuda=torch.version.cuda)
    if dev.type == "cuda":
        rec["gpu"] = torch.cuda.get_device_name(0)


def bench_jax(args, rec):
    if args.jax_cache_dir:
        import jax
        jax.config.update("jax_compilation_cache_dir", args.jax_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    import jax
    import kernels_jax as kj
    import kernels_torch as kt
    import torch
    torch.set_default_dtype(torch.float64)
    if args.kernel not in kj.PORTED:
        rec["error"] = "not ported to JAX (see PROTOCOL.md)"
        return
    platform_name = "gpu" if args.device == "cuda" else "cpu"
    jdev = jax.devices(platform_name)[0]
    fn, inputs = kt.make(args.kernel, args.size, args.device)
    ref = [t.detach().cpu() for t in flat(fn(*inputs))]
    rec["reference"] = "PyTorch eager, same device, same process"
    jargs = [jax.device_put(t.detach().cpu().numpy(), jdev) for t in inputs]
    f = jax.jit(kj.FUNCS[args.kernel])
    t0 = time.perf_counter()
    out = f(*jargs).block_until_ready()
    rec["first_call_s"] = time.perf_counter() - t0
    import numpy as np
    assert len(ref) == 1 and tuple(ref[0].shape) == tuple(out.shape), (len(ref), out.shape)
    rec["max_rel_vs_eager"], rec["nan_pattern_equal"] = compare([np.asarray(jax.device_get(out))],
                                                                [ref[0].numpy()])
    try:
        text = f.lower(*jargs).compile().as_text()
        rec["hlo_launch_ops"], rec["hlo_ops"] = hlo_count(text)
        rec["hlo_fusions"] = text.count(" fusion(")
    except Exception as e:  # noqa: BLE001
        rec["hlo_error"] = f"{type(e).__name__}: {e}"[:300]
    for _ in range(10):
        f(*jargs).block_until_ready()
    wall = []
    for _ in range(args.repeats):
        t0 = time.perf_counter(); f(*jargs).block_until_ready(); wall.append(time.perf_counter() - t0)
    rec["steady_wall_ms_median"], rec["steady_wall_ms_min"] = 1e3 * statistics.median(wall), 1e3 * min(wall)
    dyn = []
    for factor in (2, 3):
        _, inp2 = kt.make(args.kernel, scaled_size(args.kernel, args.size, factor), args.device)
        j2 = [jax.device_put(t.detach().cpu().numpy(), jdev) for t in inp2]
        t0 = time.perf_counter(); f(*j2).block_until_ready()
        dyn.append(dict(factor=factor, first_call_s=time.perf_counter() - t0))
    rec["dynamic_shapes"] = dyn
    rec["jax_compiled_variants"] = f._cache_size()
    rec["versions"] = dict(jax=jax.__version__, backend=jax.default_backend(), device=str(jdev))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tool", choices=["torch", "jax"], required=True)
    ap.add_argument("--mode", default="eager", choices=["eager", "default", "max-autotune", "reduce-overhead"])
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--size", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--phase", default="cold", choices=["cold", "warm", "none"])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--jax-cache-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = dict(tool=args.tool, mode=args.mode if args.tool == "torch" else "jit", kernel=args.kernel,
               size=args.size, device=args.device, phase=args.phase, dtype="float64",
               repeats=args.repeats, host=platform.node(), machine=platform.machine(),
               cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR") if args.tool == "torch" else args.jax_cache_dir,
               commit=os.environ.get("PASADO_COMMIT", ""), time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    try:
        (bench_torch if args.tool == "torch" else bench_jax)(args, rec)
    except Exception as e:  # noqa: BLE001 - a failure is a result
        rec["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:500] if str(e) else ''}"
        rec["traceback_tail"] = traceback.format_exc().splitlines()[-6:]
    with open(args.out, "a") as handle:
        handle.write(json.dumps(rec, default=str) + "\n")
    short = {k: rec.get(k) for k in ("tool", "mode", "kernel", "size", "phase", "first_call_s",
                                      "steady_wall_ms_median", "max_rel_vs_eager", "graph_break_count",
                                      "generated_kernels", "cuda_kernels_per_call", "hlo_launch_ops", "error")}
    print(json.dumps(short, default=str))


if __name__ == "__main__":
    main()
