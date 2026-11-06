"""Builds the checkpoint Colab notebooks from the cell sources below.

  python colab/make_notebooks.py --commit <hash of the code under test>

The notebooks pin that commit; build them after the commit that contains the code.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = r'''# 0.1 Setup: Google Drive, the code at the pinned commit, pinned packages (~3-5 min)
import json, os, subprocess, sys
COMMIT = "__COMMIT__"          # code under test (branch checkpoint)
BRANCH = "checkpoint"
REPO_URL = "https://github.com/dimitrijepesic/Pasado.git"   # default source of the code
DRIVE_ROOT = "/content/drive/MyDrive/pasado_checkpoint_2026_11"
BUNDLE = DRIVE_ROOT + "/checkpoint.bundle"   # optional: used instead of GitHub if present
LOCAL_TEST = os.environ.get("PASADO_NB_LOCAL_TEST", "0") == "1"
PIN_TORCH = ["torch==2.11.0", "torchvision==0.26.0"]        # the stack of the October A100 runs
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
PINS = ["numpy==2.5.1", "scipy==1.18.0", "scikit-learn==1.9.0", "mpmath==1.4.1", "affapy==0.1"]
if LOCAL_TEST:
    REPO, DRIVE_ROOT = os.environ["PASADO_NB_REPO"], os.environ["PASADO_NB_OUT"]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    REPO = "/content/Pasado"
    if not os.path.isdir(REPO + "/.git"):
        src = BUNDLE if os.path.exists(BUNDLE) else REPO_URL
        extra = [] if src == BUNDLE else ["--branch", BRANCH]
        subprocess.run(["git", "clone", *extra, src, REPO], check=True)
    subprocess.run(["git", "-C", REPO, "checkout", "--quiet", "--detach", COMMIT], check=True)
    import importlib.metadata as md
    if md.version("torch").split("+")[0] != PIN_TORCH[0].split("==")[1]:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIN_TORCH,
                        "--index-url", TORCH_INDEX], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
print("code at", subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip())
'''

ENV = r'''# 0.2 Environment record and result folder on Drive (one folder per GPU model)
sys.path.insert(0, os.path.join(REPO, "colab"))
import nbutil
GPU = nbutil.gpu_tag()
OUT = os.path.join(DRIVE_ROOT, "__NAME__", GPU)
CACHE = os.path.join(DRIVE_ROOT, "inductor_cache", GPU)      # compilation is paid once per GPU model
os.makedirs(CACHE, exist_ok=True)
R = nbutil.Runner(REPO, OUT, extra_env={"TORCHINDUCTOR_CACHE_DIR": CACHE})
T = nbutil.LOCAL_TEST
DEV = "cpu" if T else "cuda"
NETS = ["3layer"] if T else ["3layer", "4layer", "5layer", "big"]
if T:
    BIG_CHUNK = 8
else:
    free_mib = int(nbutil.sh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").split()[0])
    BIG_CHUNK = 64 if free_mib >= 70 * 1024 else 32
print(json.dumps(R.env_record, indent=1))
print("results ->", OUT, "| big chunk", BIG_CHUNK)
if not T:
    print(nbutil.sh("nvidia-smi"))
'''

SMOKE = r'''# 0.3 Smoke test (~2 min): the batched path against the sequential loop on this device
ok = R.run("00_smoke_batched_splits", ["experiments/test_batched_splits.py", "--quick", "--device", DEV],
           estimate_min=2)
assert ok, "smoke test failed - stop here and look at the log"
'''

A = r'''# A. Split sweep after the fix: bound, time, peak VRAM (~15 min). Mentor's figure.
SPLITS = [1, 2] if T else [1, 2, 4, 8, 16, 32, 64]
IMGS = 1 if T else 3
EST = {"sequential": {"3layer": 1, "4layer": 1.5, "5layer": 2, "big": 5},
       "batched_G": {"3layer": 0.3, "4layer": 0.3, "5layer": 0.3, "big": 3},
       "batched_auto": {"3layer": 0.3, "4layer": 0.3, "5layer": 0.3, "big": 0.7}}
seq_csvs = []
for net in NETS:
    for mode in ("sequential", "batched_G", "batched_auto"):
        out = R.path("A_sweep", f"{net}_{mode}.csv")
        args = ["experiments/measure_split_sweep.py", "--device", DEV, "--dtype", "float64",
                "--networks", net, "--splits", *SPLITS, "--num-images", IMGS, "--eps-indices", 0, 8,
                "--warmup", 1 if T else 2, "--repeats", 1 if T else 5, "--seed", 7, "--out", out,
                "--split-mode", "sequential" if mode == "sequential" else "batched"]
        if mode == "batched_auto":
            args += ["--plane-fit", "auto"]
        if net == "big" and mode != "sequential":
            args += ["--split-chunk", BIG_CHUNK]
        R.run(f"A_{net}_{mode}", args, outputs=[out], estimate_min=EST[mode][net])
        if mode == "sequential":
            seq_csvs.append(out)
R.run("A_soundness", ["experiments/check_sweep_soundness.py", *seq_csvs, "--out", R.path("A_soundness.csv")],
      estimate_min=1)
R.run("A_report", ["experiments/make_checkpoint_report.py", "--postfix", OUT, "--label", "postfix_" + GPU,
                   "--fig-dir", R.path("figures", "x")[:-2]])
try:
    from IPython.display import Image, display
    for f in sorted(os.listdir(os.path.join(OUT, "figures"))):
        display(Image(os.path.join(OUT, "figures", f)))
except ImportError:
    pass
'''

B = r'''# B. Share of the CPU part (gelsd and copies) in a pass (~6 min)
for net in NETS:
    out = R.path("B_cpu_share", f"{net}.csv")
    R.run(f"B_cpu_share_{net}", ["experiments/measure_cpu_share.py", "--device", DEV, "--networks", net,
                                 "--splits", *([1, 2] if T else [1, 8, 64]), "--num-images", 1 if T else 3,
                                 "--warmup", 1 if T else 3, "--repeats", 1 if T else 5,
                                 "--big-chunk", BIG_CHUNK, "--out", out],
          outputs=[out], estimate_min={"big": 3}.get(net, 1))
'''

C = r'''# C. Equal time on the same instance: the CPU of this VM at n = 1 (~2 min); CUDA side from A
out = R.path("C_cpu_n1.csv")
R.run("C_cpu_n1", ["experiments/measure_split_sweep.py", "--device", "cpu", "--dtype", "float64",
                   "--networks", *NETS, "--splits", 1, "--num-images", 1 if T else 3, "--eps-indices", 0, 8,
                   "--warmup", 1, "--repeats", 1 if T else 3, "--seed", 7, "--out", out],
      outputs=[out], estimate_min=2)
'''

D = r'''# D. Device matrix: CUDA against the CPU reference, 30 images x 16 eps, per image (~8 min)
dev_dir = R.path("D_matrix", "cuda", "x")[:-2]
for net in NETS:
    args = ["experiments/postfix_device_matrix.py", "--device", DEV, "--networks", net,
            "--repeat-precise", "--out-dir", dev_dir]
    if T:
        args += ["--num-images", 2, "--eps-indices", 0, 8]
    R.run(f"D_cuda_{net}", args, estimate_min={"big": 4}.get(net, 1))
ref = os.path.join(REPO, "results", "postfix_rerun_2026-11", "cpu_reference")
R.run("D_compare", ["experiments/compare_device_matrix.py", "--cpu", ref, "--device", dev_dir,
                    "--out", R.path("D_summary.csv")] + (["--allow-subset"] if T else []))
# Same-instance check: the Colab CPU against the local reference and against CUDA (subset).
spot = R.path("D_matrix", "colab_cpu", "x")[:-2]
R.run("D_colab_cpu_spot", ["experiments/postfix_device_matrix.py", "--device", "cpu",
                           "--networks", *(["3layer"] if T else ["3layer", "big"]), "--configs", "precise_f64_real",
                           "--num-images", 2 if T else 5, "--eps-indices", 0, 8, 15, "--out-dir", spot],
      estimate_min=1)
R.run("D_spot_vs_reference", ["experiments/compare_device_matrix.py", "--cpu", ref, "--device", spot,
                              "--allow-subset", "--out", R.path("D_spotcheck_summary.csv")])
'''

E = r'''# E. Layerwise CPU/CUDA difference, real and complex solver (~5 min)
out = R.path("E_layerwise.csv")
R.run("E_layerwise", ["experiments/measure_layerwise_device_diff.py", "--device", DEV, "--dtype", "float64",
                      "--networks", *NETS, "--num-images", 1 if T else 5,
                      "--eps-indices", *([0] if T else [0, 4, 8, 12, 15]), "--solvers", "real", "complex",
                      "--seed", 7, "--summary-only", "--out", out], outputs=[out], estimate_min=5)
'''

F = r'''# F. How far the splits go below one minute per pass (~30-60 min; stop any time, it resumes)
LIMIT = 0.2 if T else 60
for mode in ("batched_auto", "sequential", "batched_G"):       # most useful first
    for net in NETS:
        out = R.path("F_ramp", f"{net}_{mode}.csv")
        R.run(f"F_ramp_{net}_{mode}", ["experiments/measure_split_ramp.py", "--device", DEV, "--networks", net,
                                       "--modes", mode, "--start", 8 if T else 64, "--limit-s", LIMIT,
                                       "--max-splits", 32 if T else 65536, "--out", out],
              outputs=[out], estimate_min=4)
# F2: one full standard run (30 images x 16 eps) per network at the largest n of section A whose
# batched_auto median pass time is at most 125 ms (so 480 passes could fit in a minute).
import csv, statistics
full = []
for net in NETS:
    rows = list(csv.DictReader(open(R.path("A_sweep", f"{net}_batched_auto.csv"))))
    fit = [int(n) for n in sorted({r["n_splits"] for r in rows}, key=int)
           if statistics.mean(float(r["time_median_ms"]) for r in rows if r["n_splits"] == n) <= 125]
    if fit:
        full.append(f"{net}={fit[-1]}")
print("full runs:", full)
if full:
    R.run("F_full_run", ["experiments/measure_split_ramp.py", "--device", DEV, "--full-run", *full,
                         "--big-chunk", BIG_CHUNK, "--out", R.path("F_full_run.csv")], estimate_min=5)
'''

REPORT = r'''# Z. Summary tables (re-run after any section). Copy the whole Drive folder OUT into the repo as
# results/postfix_rerun_2026-11/colab_<GPU>/ and run make_checkpoint_report.py there to update CHECKPOINT.md.
subprocess.run([sys.executable, "experiments/make_checkpoint_report.py", "--postfix", OUT,
                "--label", "postfix_" + GPU, "--fig-dir", os.path.join(OUT, "figures")], cwd=REPO)
'''

LIRPA_INSTALL = r'''# 0.4 auto_LiRPA at a pinned commit (no release tag after v0.3 exists), dependencies without torch
LIRPA_COMMIT = "e3930353a3589f7d10879f1e71223d1eefb57196"
if not T:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                    f"git+https://github.com/Verified-Intelligence/auto_LiRPA.git@{LIRPA_COMMIT}"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "appdirs", "sortedcontainers", "packaging",
                    "graphviz", "pyyaml"], check=True)
print(subprocess.run([sys.executable, "-c", "import auto_LiRPA, torch; print(auto_LiRPA.__version__, torch.__version__)"],
                     capture_output=True, text=True).stdout)
'''

LIRPA_TIERS = r'''# T1-T4 in priority order; each line resumes per (network, eps, image). Planning estimates from the
# August A100 sweep (old per-image times), not results.
TIERS = [
    ("T1_small_10img", ["3layer", "4layer", "5layer"], [0, 2, 4], "0-9", 9, 40),
    ("T2_big_10img_K1", ["big"], [0, 2, 4], "0-9", 1, 8),
    ("T3_small_img10-29", ["3layer", "4layer", "5layer"], [0, 2, 4], "10-29", 9, 80),
    ("T4_all_30img_16eps_K1", ["3layer", "4layer", "5layer", "big"], list(range(16)), "0-29", 1, 110),
]
if T:
    TIERS = [("T1_small_10img", ["3layer"], [2], "0", 3, 1)]
for name, nets, eps, imgs, budget, est in TIERS:
    for net in nets:
        R.run(f"{name}_{net}", ["experiments/lirpa_same_gpu.py", "--device", DEV, "--networks", net,
                                "--eps-indices", *eps, "--images", imgs, "--budget", budget,
                                "--pasado-ks", *([1, 3] if T else [1, 3, 5, 7, 9, 16, 32, 64, 128, 256]),
                                "--big-chunk", BIG_CHUNK, "--out-dir", R.path("records", "x")[:-2]],
              estimate_min=round(est / len(nets), 1))
    subprocess.run([sys.executable, "experiments/analyze_lirpa_postfix.py", R.path("records", "x")[:-2],
                    "--out-md", R.path("tables.md")], cwd=REPO)
'''

LIRPA_ARCHIVE = r'''# Optional: auto_LiRPA K=1..9 against the archived August A100 sweep (put bab_sweep_gpu.csv on Drive).
# Meaningful only once all 30 images of an (network, eps) are present (T1 + T3).
import csv, statistics
arch = os.path.join(DRIVE_ROOT, "bab_sweep_gpu.csv")
if os.path.exists(arch):
    recs = [json.loads(l) for l in open(R.path("records", "lirpa.jsonl"))]
    for r in csv.DictReader(open(arch)):
        mine = [x for x in recs if x["network"] == r["network"] and x["eps_index"] == int(r["eps_index"])
                and r["solves"] in x["curve"]]
        if len(mine) == int(r["images_averaged"]):
            new = statistics.mean(x["curve"][r["solves"]]["bound"] for x in mine)
            print(r["network"], r["eps_index"], "K=" + r["solves"], "archived", r["lirpa_bab"], "now", new,
                  "rel diff", abs(new - float(r["lirpa_bab"])) / float(r["lirpa_bab"]))
else:
    print("no archived CSV on Drive - skipped")
'''

DEEP = r'''# Train deep8 (~10-15 min on an A100; resumes per epoch), then the bounds of big and deep8 (~10 min)
deep_dir = R.path("deep8", "x")[:-2]
R.run("A_train_deep8", ["experiments/train_deep_network.py", "--out-dir", deep_dir, "--device", DEV]
      + (["--epochs", 1, "--limit-batches", 3] if T else []), estimate_min=15)
print(open(os.path.join(deep_dir, "training.json")).read()[:300])
R.run("B_bounds_deep8", ["experiments/measure_deep_network.py", "--deep-dir", deep_dir, "--device", DEV,
                         "--out", R.path("bounds.csv")]
      + (["--num-images", 1, "--eps-indices", 0, "--splits", 1, "--smoke"] if T else []), estimate_min=10)
'''


def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.strip("\n").splitlines(True))


def code(text):
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                source=text.strip("\n").splitlines(True))


def notebook(cells):
    return dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python"),
                              language_info=dict(name="python"), accelerator="GPU",
                              colab=dict(provenance=[], gpuType="A100", machine_shape="hm")))


def build(commit):
    sub = lambda text, name: text.replace("__COMMIT__", commit).replace("__NAME__", name)  # noqa: E731
    nb01 = [
        md("""# 01 - Post-fix re-run on CUDA
Protocol: `results/postfix_rerun_2026-11/PROTOCOL.md` (written before the run). Sections are in
priority order - stop when the A100 time runs out; re-running the notebook from the top skips
every run that already finished (one JSON per run on Drive). Planning estimates per section are
in the cell titles; they come from the October A100 runs and are not results.

The code comes from branch `checkpoint` of `github.com/dimitrijepesic/Pasado`, checked out at the
pinned commit (alternatively a `checkpoint.bundle` on Drive, see `colab/README.md`).
Runtime: GPU, A100, high RAM."""),
        code(sub(SETUP, "01_postfix_rerun")), code(sub(ENV, "01_postfix_rerun")), code(SMOKE),
        code(A), code(B), code(C), code(D), code(E), code(F), code(REPORT)]
    nb01c = [
        md("""# 01c - auto_LiRPA against Pasado after the fix, both on the same GPU
Protocol: `results/lirpa_postfix_2026-11/PROTOCOL.md`. Run in a fresh runtime (auto_LiRPA is
installed into it). Tiers in priority order; every record is per (tool, network, eps, image)
and resumes after a disconnect."""),
        code(sub(SETUP, "01c_lirpa_postfix")), code(sub(ENV, "01c_lirpa_postfix")), code(LIRPA_INSTALL),
        code(LIRPA_TIERS), code(LIRPA_ARCHIVE)]
    nb03 = [
        md("""# 03 - A network twice as deep as big (P2, lowest priority)
Protocol: `results/deep_network_2026-11/PROTOCOL.md`. Training writes a checkpoint per epoch to Drive."""),
        code(sub(SETUP, "03_deep_network")), code(sub(ENV, "03_deep_network")), code(DEEP)]
    out = {"01_postfix_rerun.ipynb": nb01, "01c_lirpa_postfix.ipynb": nb01c, "03_deep_network.ipynb": nb03}
    try:
        import compilers_cells
        out["02_compilers.ipynb"] = compilers_cells.cells(sub, md, code, SETUP, ENV)
    except ImportError:
        pass
    for name, cells in out.items():
        with open(os.path.join(HERE, name), "w") as handle:
            json.dump(notebook(cells), handle, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    build(ap.parse_args().commit)
