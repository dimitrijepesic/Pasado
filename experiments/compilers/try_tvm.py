"""TVM on the isolated kernels, through torch.export and the Relax PyTorch frontend
(results/compilers_2026-11/PROTOCOL.md). Every step is recorded with its error; the first
failing step ends the kernel. Appends one JSON line per kernel to --out.

  python experiments/compilers/try_tvm.py --device cuda --kernels fit viete bounds --out tvm.jsonl
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kernels", nargs="+", default=["fit", "viete", "bounds", "complex_cubic", "lstsq"])
    ap.add_argument("--sizes", nargs="*", default=None, help="e.g. fit=1024 bounds=small")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sizes = dict(fit="1024", viete="65536", bounds="small", complex_cubic="65536", lstsq="1024")
    for item in args.sizes or []:
        k, v = item.split("=")
        sizes[k] = v
    for kernel in args.kernels:
        rec = dict(tool="tvm", kernel=kernel, size=sizes[kernel], device=args.device, steps=[],
                   host=platform.node(), commit=os.environ.get("PASADO_COMMIT", ""))

        def step(name, fn):
            t0 = time.perf_counter()
            try:
                out = fn()
            except Exception as e:  # noqa: BLE001 - the failing step is the result
                rec["steps"].append(dict(step=name, ok=False, seconds=time.perf_counter() - t0,
                                         error=f"{type(e).__name__}: {str(e).strip().splitlines()[0][:400] if str(e).strip() else ''}",
                                         traceback_tail=traceback.format_exc().splitlines()[-4:]))
                raise Stop
            rec["steps"].append(dict(step=name, ok=True, seconds=time.perf_counter() - t0))
            return out

        try:
            import numpy as np
            import torch
            torch.set_default_dtype(torch.float64)
            tvm = step("import tvm", lambda: __import__("tvm"))
            from tvm import relax
            rec["tvm_version"] = tvm.__version__
            target_name = "cuda" if args.device == "cuda" else "llvm"
            dev = tvm.cuda(0) if target_name == "cuda" else tvm.cpu(0)
            step("device available", lambda: (_ for _ in ()).throw(RuntimeError("tvm.cuda(0) does not exist"))
                 if not dev.exist else True)
            import kernels_torch as kt
            fn, inputs = kt.make(kernel, sizes[kernel], "cpu")
            ref = fn(*inputs)
            ref = torch.stack(ref) if isinstance(ref, tuple) else ref

            class M(torch.nn.Module):
                def forward(self, *xs):
                    return fn(*xs)

            ep = step("torch.export", lambda: torch.export.export(M(), tuple(inputs)))
            from tvm.relax.frontend.torch import from_exported_program
            mod = step("Relax frontend (from_exported_program)", lambda: from_exported_program(ep))
            target = tvm.target.Target(target_name)

            def build():
                m = relax.get_pipeline("zero")(mod)
                if target_name == "cuda":
                    from tvm import dlight as dl
                    with target:
                        m = dl.ApplyDefaultSchedule(dl.gpu.Reduction(), dl.gpu.GeneralReduction(),
                                                    dl.gpu.Fallback())(m)
                rec["tir_kernels"] = sum(1 for _, f in m.functions.items() if type(f).__name__ == "PrimFunc")
                t0 = time.perf_counter()
                ex = tvm.compile(m, target=target) if hasattr(tvm, "compile") else relax.build(m, target)
                rec["compile_s"] = time.perf_counter() - t0
                return ex
            ex = step("build (zero pipeline, dlight default schedules on CUDA)", build)
            vm = step("VirtualMachine", lambda: relax.VirtualMachine(ex, dev))
            to_dev = (lambda a: tvm.runtime.tensor(a, dev)) if hasattr(tvm.runtime, "tensor") else \
                (lambda a: tvm.nd.array(a, dev))
            targs = [to_dev(t.numpy()) for t in inputs]
            out = step("run", lambda: vm["main"](*targs))
            got = out[0] if isinstance(out, (list, tuple)) or type(out).__name__ == "Array" else out
            got = got.numpy()
            r = ref.numpy()
            fin = np.isfinite(r) & np.isfinite(got)
            rec["max_rel_vs_eager"] = float(np.abs(got[fin] - r[fin]).max() / np.abs(r[fin]).max())
            rec["nan_pattern_equal"] = bool((np.isnan(got) == np.isnan(r)).all())
            rec["out_dtype"] = str(got.dtype)
            for _ in range(10):
                vm["main"](*targs)
            dev.sync()
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter(); vm["main"](*targs); dev.sync(); times.append(time.perf_counter() - t0)
            rec["steady_wall_ms_median"] = 1e3 * statistics.median(times)
        except Stop:
            pass
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
        failed = [s for s in rec["steps"] if not s["ok"]]
        rec["stopped_at"] = failed[0]["step"] if failed else None
        with open(args.out, "a") as handle:
            handle.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("kernel", "stopped_at", "max_rel_vs_eager", "tir_kernels",
                                                   "compile_s", "steady_wall_ms_median")}),
              (failed[0]["error"] if failed else ""), flush=True)


if __name__ == "__main__":
    main()
