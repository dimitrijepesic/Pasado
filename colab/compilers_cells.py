"""Cells of colab/02_compilers.ipynb (imported by make_notebooks.py)."""

JAX_INSTALL = r'''# 0.4 JAX with CUDA (pinned), without touching torch; recorded in the environment JSON of every run
JAX_PIN = "0.12.0.1"
if not T:
    import importlib.util
    have = subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__, jax.default_backend())"],
                          capture_output=True, text=True).stdout.split()
    if have[:2] != [JAX_PIN, "gpu"]:
        for extra in ("cuda13", "cuda12"):
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"jax[{extra}]=={JAX_PIN}"],
                               capture_output=True, text=True)
            ok = subprocess.run([sys.executable, "-c", "import jax; assert jax.default_backend() == 'gpu'"],
                                capture_output=True, text=True).returncode == 0
            print(f"jax[{extra}]=={JAX_PIN}: install exit {r.returncode}, GPU backend {'yes' if ok else 'no'}")
            if ok:
                break
print(subprocess.run([sys.executable, "-c", "import jax; print(jax.__version__, jax.devices())"],
                     capture_output=True, text=True).stdout)
'''

SMOKE = r'''# 0.5 Smoke test (~1 min): one eager and one compiled measurement of the plane fit
out = R.path("smoke", "records.jsonl")
for mode in ("eager", "default"):
    R.run(f"00_smoke_torch_{mode}", ["experiments/compilers/bench.py", "--tool", "torch", "--mode", mode,
                                     "--kernel", "fit", "--size", "1024", "--device", DEV, "--repeats", 5,
                                     "--out", out], estimate_min=1)
print(open(out).read()[-1500:])
'''

TORCH = r'''# A. torch.compile: default, max-autotune, reduce-overhead, every kernel; cold and warm processes
# (~35 min; max-autotune dominates). One run per kernel, so it resumes per kernel.
KERNELS = ["fit"] if T else ["fit", "viete", "bounds", "sigmoid", "complex_cubic", "lstsq"]   # priority order
for k in KERNELS:
    R.run(f"A_torch_{k}", ["experiments/compilers/run_matrix.py", "--device", DEV, "--tools", "torch",
                           "--kernels", k, "--repeats", 5 if T else 50, "--out", R.path("matrix", "x")[:-2]]
          + (["--sizes", "fit=1024", "--torch-modes", "eager", "default"] if T else []),
          estimate_min={"fit": 8, "viete": 5, "bounds": 8, "sigmoid": 8, "complex_cubic": 4, "lstsq": 3}[k])
'''

JAX = r'''# B. JAX/XLA (jit, x64) on the ported kernels; cold and warm processes (~6 min)
for k in (["fit"] if T else ["fit", "viete", "bounds", "complex_cubic"]):
    R.run(f"B_jax_{k}", ["experiments/compilers/run_matrix.py", "--device", DEV, "--tools", "jax",
                         "--kernels", k, "--repeats", 5 if T else 50, "--out", R.path("matrix", "x")[:-2]]
          + (["--sizes", "fit=1024"] if T else []), estimate_min=2)
'''

TVM = r'''# C. TVM: at most 60 minutes for an installation with CUDA support, then the kernels through
# torch.export + the Relax PyTorch frontend. Every attempt and every failing step is recorded.
import time
TVM_BUDGET_S = 60 * 60
TVM_ATTEMPTS = [   # in order; the first one whose tvm sees a CUDA device is used
    ["apache-tvm==0.27.0.post1"],
    ["--pre", "mlc-ai-nightly-cu130", "-f", "https://mlc.ai/wheels"],
    ["--pre", "mlc-ai-nightly-cu128", "-f", "https://mlc.ai/wheels"],
    ["--pre", "mlc-ai-nightly-cu123", "-f", "https://mlc.ai/wheels"],
]
log_path = R.path("C_tvm_install.json")
attempts = json.load(open(log_path)) if os.path.exists(log_path) else []
venv = os.path.join("/content" if not T else OUT, "tvm_venv")
py = os.path.join(venv, "bin", "python")
probe = "import tvm; print(tvm.__version__, tvm.cuda(0).exist if hasattr(tvm, 'cuda') else 'no cuda attr')"
ok = any(a.get("cuda") for a in attempts)
start = time.time()
if not ok and not T:
    if not os.path.exists(py):   # separate environment: TVM must not change the torch of the other sections
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv], check=True)
    for spec in TVM_ATTEMPTS:
        if time.time() - start > TVM_BUDGET_S:
            attempts.append(dict(spec=spec, skipped="time budget of 60 min used up")); break
        if any(a.get("spec") == spec for a in attempts):
            continue
        t0 = time.time()
        try:
            r = subprocess.run([py, "-m", "pip", "install", "-q", *spec], capture_output=True, text=True,
                               timeout=max(60, TVM_BUDGET_S - (time.time() - start)))
            p = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=300)
            rec = dict(spec=spec, install_exit=r.returncode, install_tail=r.stderr[-800:], seconds=time.time() - t0,
                       probe=p.stdout.strip(), probe_err=p.stderr[-800:], cuda=p.stdout.strip().endswith("True"))
        except subprocess.TimeoutExpired as e:
            rec = dict(spec=spec, error=f"timeout: {e}", seconds=time.time() - t0, cuda=False)
        attempts.append(rec)
        json.dump(attempts, open(log_path, "w"), indent=1)
        print(rec.get("spec"), "->", rec.get("probe") or rec.get("error"), "| exit", rec.get("install_exit"))
        if rec.get("cuda"):
            break
        subprocess.run([py, "-m", "pip", "uninstall", "-q", "-y", "apache-tvm", "mlc-ai-nightly-cu130",
                        "mlc-ai-nightly-cu128", "mlc-ai-nightly-cu123"], capture_output=True)
json.dump(attempts, open(log_path, "w"), indent=1)
if T or any(a.get("cuda") for a in attempts):
    R.run("C_tvm_kernels", ["experiments/compilers/try_tvm.py", "--device", DEV, "--kernels", "fit", "viete",
                            "bounds", "complex_cubic", "lstsq", "--out", R.path("C_tvm.jsonl")],
          estimate_min=5) if T else subprocess.run(
        [py, "experiments/compilers/try_tvm.py", "--device", DEV, "--out", R.path("C_tvm.jsonl")], cwd=REPO)
else:
    print("TVM with CUDA did not install within the budget; see", log_path)
'''

SUMMARY = r'''# Z. Table tool x kernel
subprocess.run([sys.executable, "experiments/compilers/summarize.py", R.path("matrix", "x")[:-2],
                "--out-md", R.path("table.md")], cwd=REPO)
if os.path.exists(R.path("C_tvm.jsonl")):
    for line in open(R.path("C_tvm.jsonl")):
        r = json.loads(line)
        print("TVM", r["kernel"], "stopped at:", r.get("stopped_at"), "| max rel vs eager:", r.get("max_rel_vs_eager"),
              "| out dtype:", r.get("out_dtype"), "| TIR kernels:", r.get("tir_kernels"))
'''


def cells(sub, md, code, setup, env):
    return [
        md("""# 02 - Default compilation by different tools on Pasado kernels
Protocol: `results/compilers_2026-11/PROTOCOL.md` (written before the run). Sections in priority
order; every run resumes. Cold and warm compilation are measured in separate processes; the
Inductor cache lives on Drive per GPU model. Runtime: GPU, A100."""),
        code(sub(setup, "02_compilers")), code(sub(env, "02_compilers")), code(JAX_INSTALL), code(SMOKE),
        code(TORCH), code(JAX), code(TVM), code(SUMMARY)]
