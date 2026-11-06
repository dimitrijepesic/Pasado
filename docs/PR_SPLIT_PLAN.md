# Plan: splitting `gpu-hybrid` into reviewable pull requests for uiuc-arc/Pasado

Plan only (2026-11-06). Nothing has been pushed, no PR opened. Sizes are `git diff --numstat`
against upstream `main` (`e857413`) on `checkpoint`.

## Principles

- Every PR leaves the published numbers of the artifact unchanged by default: new behaviour sits
  behind a selector that is off, or is bit-identical (checked by a test in the same PR).
- No PR touches `Section_5_4/results/*.pth`. The branch modifies all 24 of them (regenerated
  30-image averages from Sept 15); every PR branch starts from upstream `main` and the files are
  restored from it (`git checkout e857413 -- Section_5_4/results/`).
- Tests travel with the code they test; raw results, logs and write-ups stay on the fork and are
  linked from the PR description (`results/` + `logs/` are ~124k added lines).
- One topic per PR, in dependency order, each small enough to review in one sitting.

## Proposed sequence

| # | PR | Files (lines) | Source commits | Default behaviour | Evidence in the PR |
|---|---|---|---|---|---|
| 0 | **Soundness fixes** (already open as uiuc-arc/Pasado#3): `Zonotope.__neg__` / `DualZonotope.__neg__` negate the generators; boundary checks bound `|E|` | `SimpleZono.py` (+2/-2), `precise_transformer.py` (+4/-4) | `b526762` (fork PR branch `a99d70b`) | **changes results** (that is the point); bound lower by 17-37% at n_splits = 1, slightly higher at fine splits | add `experiments/test_soundness.py` and `verify_upstream_bugs.py` to the PR; table of `results/soundness_fix_2026-10-22/` in the description. Ask the authors how to present changed numbers (re-generate the `.pth` files in a separate commit, or a note in the README) |
| 1 | **Device and dtype plumbing**: new tensors on the device and dtype of their inputs; `lin_reg_tensor_batched` solves with gelsd on the CPU for GPU inputs | `SimpleZono.py` (+5/-5), `Duals.py` (+4/-4), `HyperDuals.py` (+15/-15), `HyperDualZono.py` (+3/-3), part of `precise_transformer.py` | `ca122eb`, `f51f977`, `4ba249e` | bit-identical on CPU | `experiments/check_device_plumbing.py` (CPU/MPS/CUDA), `test_batched_lin_reg.py` device cases |
| 2 | **`get_lipschitz.py` options**: `--seed`, `--device`, `--dtype`, `--num-images`, `--precise-only`, `--n-splits`, `--eps-indices`, `--no-save`, `--results-dir`, `--per-image-csv`; suffixed file names for non-default runs so reference files can never be overwritten | `Section_5_4/get_lipschitz.py` (+227/-60) | `708c5a8`, `f8476dc` (CLI part), this checkpoint | default run unchanged | `experiments/test_get_lipschitz_cli.py`; drop the `time` prefixes in `lipschitz.sh` (`f8476dc`) unless wanted |
| 3 | **Vectorized precise transformer behind selectors**: corner and boundary checks (A, B; on), batched lstsq (C), batched grid (D), real Viete cubic (E; changes bounds by <= 5.7e-5, so off) | `precise_transformer.py` (rest of +295/-40) | `f8476dc`, `e3916af` | A+B bit-identical; C, D, E off | the 5 unit tests in `forward_mode_tensorized_src/test_*.py`, `test_batched_lin_reg.py`, `experiments/mutation_check.py` (M1-M14); speedup tables (pre-fix; re-measure before quoting) |
| 4 | **Batched interval splitting**: all sub-intervals in one forward pass, `--split-mode` in the sweep | `batched_splits.py` (core of +288), `experiments/measure_split_sweep.py` | `b144542`, `b0eb02d`, `8e3d0df` | opt-in (new API; `get_lipschitz.py` unchanged) | `experiments/test_batched_splits.py` (per-split equality with the sequential loop) |
| 5 | **Closed-form plane fit A2 for GPUs**: `planar_fit.py`, `PASADO_CLOSED_FORM_FIT`, `PASADO_AUTO_PLANE_FIT` (A100 thresholds), optional `PASADO_COMPILED_FIT` | `planar_fit.py` (+257), `batched_splits.py` (switches) | `7c19c57`, `500ae16`, `832323a`, `3243aef` | off | `test_planar_fit.py`, `test_auto_plane_fit.py`, mutants M15-M29; the pre-registered comparison (`results/lstsq_comparison_2026-10-27_v3_zpivot/`) linked |
| 6 | (optional) **Experiment scripts** used by the write-ups | `experiments/*.py` (~10k lines) | many | none | only if the authors want them upstream; otherwise they stay on the fork |

Not proposed for upstream: `logs/`, `results/` (except what a PR description links),
`requirements-win-freeze.txt`, `KNOWN_ISSUES.md`, `PR_TODO_environment.md`, the outward-rounding
prototype (`USE_OUTWARD_D`, preliminary) and the checkpoint notebooks.

## Mechanics (when approved)

For each PR: a branch from upstream `main`, `git cherry-pick` of the listed commits (commits that
mix topics, e.g. `f8476dc`, are split with `git checkout -p` into the right PR), the `.pth` files
restored from `main`, `experiments/run_all_checks.sh` run on the branch, then a PR whose text links
the fork's result folders. PR 0 should be merged (or its numbers decided) before 3-5, because every
measured number in those PRs is post-fix.

## Open questions for the authors

1. Do they want performance code (3-5) upstream at all, or only the fixes (0-2)?
2. How should the changed numbers after PR 0 be presented in the artifact?
3. Is a new dependency-free module (`batched_splits.py`, `planar_fit.py`) acceptable in
   `forward_mode_tensorized_src/`?
