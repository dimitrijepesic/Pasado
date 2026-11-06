# Post-fix re-run of the CUDA results (protocol, written before the run)

Written 2026-11-06, before any of these measurements, for the checkpoint with the mentor.
Notebook: `colab/01_postfix_rerun.ipynb`. Local part (Apple M1 Pro CPU): section D-local.

## Why

Every CUDA result before 2026-10-22 ran on the code before the two soundness fixes
(`b526762`: `Zonotope.__neg__` / `DualZonotope.__neg__` negate the generators; the boundary
checks bound `|E|`). That covers the A100 matrix of 2026-10-08, the fallback timing of
2026-10-16, the layerwise divergence and the split sweep of 2026-10-17. This run repeats
them on the fixed code and answers the mentor's open questions with measured numbers.

## Fixed for every section

- Code: the commit the notebook pins (printed and stored in every result JSON). It must
  contain `b526762`.
- GPU: whatever Colab assigns; the model, VRAM, driver, torch, CUDA and the host CPU are
  stored with every result. Results from different GPU models are not pooled.
- float64 unless a section says otherwise; real (Viete) cubic solver
  (`PASADO_REAL_CUBIC=1`), the legacy complex64 solver only as a negative control; selectors
  `PASADO_VECTORIZED_PRECISE=PASADO_VEC_BOUNDARY=PASADO_BATCHED_LSTSQ=PASADO_BATCHED_GRID=1`;
  seed 7 (np.random reseeded as in the existing scripts).
- Timing: `torch.cuda.synchronize()` at every timer boundary, warm-up passes excluded,
  median of the timed repeats. Times are per forward pass (one image, one epsilon).
- Modes of the split computation:
  - `sequential`: one forward pass per sub-interval, plane fit by `gelsd` on the CPU (the
    original loop);
  - `batched_G`: all sub-intervals of a chunk in one pass, `gelsd` on the CPU (the current
    production default of the batched path);
  - `batched_auto`: as `batched_G` with `PASADO_AUTO_PLANE_FIT=1` (gelsd below 512 boxes per
    call in float64, the eager closed form A2 from 512 on; opt-in).
- `big` chunk: 64 splits per pass when the GPU has at least 70 GiB free, otherwise 32.

## Sections, in the order of the notebook (priority order)

**A. Split sweep.** 3layer, 4layer, 5layer, big; 3 images; eps indices 0 and 8;
n_splits 1, 2, 4, 8, 16, 32, 64; all three modes; 2 warm-up passes, 5 timed repeats.
Recorded: bound, time, peak VRAM per (network, image, eps, n, mode).
Checks: (1) every batched bound within 1e-10 relative of the sequential bound for the same
(network, image, eps, n); (2) every sequential row passes `check_sweep_soundness.py`
(bound >= sampled autograd maximum of |d logit/dt|, 4001 points). Output figure, per network:
left axis bound normalised to n = 1, right axis time per pass (ms), eps index 0.

**B. Share of the CPU part.** All 4 networks; S = 1, 8, 64; modes `sequential`, `batched_G`
and `batched_auto`; 3 images, eps index 0; 3 warm-up passes, 5 timed repeats (instrumented and
uninstrumented passes alternate). Every call of
`lin_reg_tensor_batched` is timed in its steps (device-to-host copies, design matrix, CPU
`gelsd`, host-to-device copy). Metric: CPU share = (copies + design + gelsd) / pass time.
The instrumentation adds synchronizations; the uninstrumented pass time of the same
configuration is recorded next to it, and the share is reported against both.

**C. Equal time on the same instance.** The CPU of the same Colab VM (threads recorded) runs
`sequential` n = 1 for the configurations of A (1 warm-up, 3 repeats). For each (network,
eps, mode) the largest n of A whose CUDA median time is at most the CPU n = 1 median time is
k*, and the result is bound(n = 1, CPU) / bound(k*, CUDA). Descriptive, no decision rule.

**D. Device matrix (CPU against CUDA).** 30 images x 16 eps for every network, written per
image (`get_lipschitz.py --per-image-csv`), with `--no-save`:
(i) float64 full analysis (precise, zonotope, interval); (ii) float64 precise-only, run twice
on CUDA; (iii) float32 precise-only; (iv) complex64 solver, float64 precise-only (negative
control). The CPU side of (i), (iii) and (iv) is computed locally on the Apple M1 Pro
(section D-local, committed to `results/postfix_rerun_2026-11/cpu_reference/`). To show that
the host CPU does not matter at this level, the notebook also runs (i) precise-only on the
Colab CPU for 3layer and big, 5 images, eps indices 0, 8, 15.
Pass criteria (as on 2026-10-08): largest relative difference per image and of the 30-image
average <= 1e-9 in float64 and <= 1e-5 in float32; the two CUDA runs identical; full and
precise-only CUDA identical. Complex control: the divergence is reported, no criterion.

**E. Layerwise CPU/CUDA difference.** `measure_layerwise_device_diff.py`, all 4 networks,
5 images, eps indices 0, 4, 8, 12, 15, solvers real and complex, float64. Metric: relative
difference of every layer bound and of the final bound. Descriptive.

**F. How far the splits go below one minute.** Per network and mode: n = 64, 128, 256, ...
doubling while the median pass time stays below 60 s, up to n = 65536; 1 image, eps index 0;
1 warm-up pass (none once a pass takes more than 10 s), 2 repeats (1 above 10 s). Batched
chunk: the largest that keeps the measured peak VRAM below half the free memory. Result:
the largest n measured below 60 s per pass. F2: for each network, one full standard run
(30 images x 16 eps) at the largest n of A whose `batched_auto` median pass time is at most
125 ms, timed end to end; only measured numbers are reported, nothing is extrapolated.

## What this run does not do

No change of the analysis or of the library defaults; no `.pth` reference file is written
(every call uses `--no-save` or the batched API). LiRPA comparisons have their own protocol
(`results/lirpa_postfix_2026-11/PROTOCOL.md`).
