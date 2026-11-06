# auto_LiRPA against Pasado after the soundness fix, both on the same GPU (protocol)

Written 2026-11-06, before the run. Notebook: `colab/01c_lirpa_postfix.ipynb`.
Script: `experiments/lirpa_same_gpu.py`, analysis: `experiments/analyze_lirpa_postfix.py`.

## Why

The comparisons in `results/optimization_section_5_4/LIRPA_COMPARISON.md` and
`EQUAL_TIME_COMPARISON.md` used Pasado values from the code before the soundness fixes of
2026-10-21/22, Pasado times from a laptop CPU and auto_LiRPA times from an A100. The mentor
asked what happens on the same hardware. Two further defects of the old protocol are fixed:

1. The old A100 sweep (`experiments/colab_bab_sweep.py`) builds the whole best-first curve
   K = 1, 3, 5, 7, 9 in one incremental pass, i.e. 9 solves per image and epsilon, but the
   old documents divided the total time by 25 (= 1 + 3 + 5 + 7 + 9). If the archived sweep
   came from that script (the only version in the repository), the auto_LiRPA time per
   solve and every auto_LiRPA time at budget K were understated by about 25/9 = 2.8x. Here
   the time of every solve is recorded, and the time at budget K is the measured
   cumulative time of the first K solves.
2. Pasado and auto_LiRPA values are recorded per image, so any image subset gives the same
   images on both sides and the per-image ratios can be reported.

## Setup

- One Colab GPU session for both tools; GPU model, host CPU, versions and the Pasado commit
  are stored with every record. auto_LiRPA is installed from commit
  `e3930353a3589f7d10879f1e71223d1eefb57196` (master on 2026-11-06; the repository has no
  release tag after v0.3), in its default `optimize=True` (CROWN-Optimized) mode, exactly as
  `experiments/colab_bab_sweep.py` calls it (fresh `BoundedModule` per solve).
- Quantity: max_j |dF_j/dt| over t in [0, eps] for x(t) = img + t (1 - img), float64.
- Images: the first N correctly classified MNIST test images of each network (the same
  order as every other script). eps indices as listed per tier.
- auto_LiRPA: best-first bisection of [0, eps] up to a budget of 9 solves (K = 1, 3, 5, 7, 9).
- Pasado: uniform splits, n_splits = k for k in 1, 3, 5, 7, 9, 16, 32, 64, 128, 256, batched
  path with the automatic plane-fit choice (`PASADO_AUTO_PLANE_FIT=1`, gelsd below 512 boxes
  per call, closed form A2 from 512), real cubic, seed 7; 1 warm-up pass per (network, k),
  time = median of 3 passes. `big` chunk as in `results/postfix_rerun_2026-11/PROTOCOL.md`.
- The Pasado bound for every image is also checked against a sampled autograd maximum
  (4001 points) - it must be at least that maximum. auto_LiRPA bounds get the same check.

## Tiers (in notebook order; stop when the GPU time runs out)

| tier | networks | eps indices | images | auto_LiRPA solves per (image, eps) |
|---|---|---|---:|---:|
| T1 | 3layer, 4layer, 5layer | 0, 2, 4 | 10 | 9 |
| T2 | big | 0, 2, 4 | 10 | 1 |
| T3 | 3layer, 4layer, 5layer | 0, 2, 4 | images 11-30 | 9 |
| T4 | all four | 0-15 | 30 | 1 |

## Reported (no decision rule; descriptive)

1. Single pass (K = 1): mean over images of lirpa / pasado, per network and eps.
2. Equal budget: mean ratio at K = 1, 3, 5, 7, 9 and the interpolated crossover K.
3. Equal time on the same GPU: for each (network, eps), auto_LiRPA at K = 9 (value, measured
   time) against (a) the smallest Pasado k that is at least as tight, with its time, and (b)
   the tightest Pasado k whose measured time fits in auto_LiRPA's K = 9 time.
4. Agreement of auto_LiRPA K = 1 with the archived A100 sweep for the same images, if the
   archived CSV (`bab_sweep_gpu.csv`, outside the repository) is put on Drive.

Every table is labelled post-fix, with the GPU model, the commit and the number of images.
