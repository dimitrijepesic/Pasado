# Checkpoint Colab notebooks (A100)

Built by `python colab/make_notebooks.py --commit <hash>`; every notebook checks out that commit.
Every measurement is one "run" with a JSON record on Drive; re-running a notebook from the top
skips finished runs, so a session can stop at any point. Results land in
`MyDrive/pasado_checkpoint_2026_11/<notebook>/<GPU model>/`; the TorchInductor cache in
`MyDrive/pasado_checkpoint_2026_11/inductor_cache/<GPU model>/`.

## Getting the code to Colab

The notebooks clone branch `checkpoint` from `github.com/dimitrijepesic/Pasado` and check out the
pinned commit, so the branch has to be pushed first. Without pushing, a bundle works too:

    git bundle create checkpoint.bundle HEAD checkpoint

uploaded to `MyDrive/pasado_checkpoint_2026_11/checkpoint.bundle`; the notebooks use it when it
is there.

## Order (priority) and A100 time

Planning estimates from the October A100 runs, not measurements.

| step | notebook / sections | A100 minutes | answers |
|---:|---|---:|---|
| 1 | `01_postfix_rerun` setup + A, B, C, D, E | ~45 | figure, CPU share, memory, equal time, CPU/CUDA matrix, layerwise |
| 2 | `01c_lirpa_postfix` setup + T1, T2 | ~55 | LiRPA on the same GPU (10 images, eps 0.632 / 0.2 / 0.063) |
| 3 | `02_compilers` setup + A (torch.compile), B (JAX) | ~50 | compiler table on CUDA |
| 4 | `01_postfix_rerun` F | 30-60 | splits below one minute |
| 5 | `02_compilers` C (TVM, capped at 60 min install) | <= 65 | TVM on CUDA, or where it fails |
| 6 | `01c_lirpa_postfix` T3 | ~80 | the same with 30 images |
| 7 | `03_deep_network` | ~30 | the 8-layer network |
| 8 | `01c_lirpa_postfix` T4 | ~110 | single pass, all 16 eps, 30 images |

Steps 1-3 (~2.5 h) cover everything the checkpoint needs. Use a fresh runtime for `01c` (it
installs auto_LiRPA) and for `02` (it installs JAX with CUDA).

## Bringing results back

Download the Drive folders into the repository, for example
`results/postfix_rerun_2026-11/colab_<GPU>/`, `results/lirpa_postfix_2026-11/colab_<GPU>/`,
`results/compilers_2026-11/colab_<GPU>/`, then:

    python experiments/make_checkpoint_report.py --postfix results/postfix_rerun_2026-11/colab_<GPU> \
        --label postfix_a100 --update-checkpoint
    python experiments/analyze_lirpa_postfix.py results/lirpa_postfix_2026-11/colab_<GPU>/records --out-md /tmp/l.md
    python experiments/update_checkpoint_block.py lirpa_a100 /tmp/l.md
    python experiments/compilers/summarize.py results/compilers_2026-11/colab_<GPU>/matrix --out-md /tmp/c.md
    python experiments/update_checkpoint_block.py compilers_a100 /tmp/c.md

## Local test

`PASADO_NB_LOCAL_TEST=1` runs every cell on the CPU with tiny grids and without Drive/clone/pip:

    python colab/run_local_test.py colab/01_postfix_rerun.ipynb /tmp/nbtest
