# Batched interval splitting - CUDA, 2026-10-24

Outputs of `COLAB_GPU_HYBRID_BATCHED_SPLITS.ipynb`: the `--n-splits` sweep run once
with the sequential loop (`--split-mode sequential`) and once with all sub-intervals in a
single forward pass (`--split-mode batched`, `forward_mode_tensorized_src/batched_splits.py`).

Code state: `gpu-hybrid` at `69a7163` (includes the negation and `|E|` soundness fixes)
plus `batched_splits.py`, `experiments/test_batched_splits.py` and the `--split-mode`
option of `experiments/measure_split_sweep.py`. The GPU model is not recorded in the logs;
`--split-chunk 64` was chosen automatically from the free memory, which implies at least
64 GiB of VRAM.

## Protocol

- Networks `3layer`, `4layer`, `5layer`, `big`; 3 correctly classified MNIST test images,
  epsilon indices `0` and `8`, `n_splits` in `1, 2, 4, 8, 16, 32, 64`.
- `float64`, real (Viete) cubic solver, `seed = 7`, 2 warm-up runs, 3 timed repeats
  (median reported); `torch.cuda.synchronize()` at every timer boundary.
- Seeded batched runs share one noise stream across the splits, which reproduces the
  sequential protocol (`np.random` is reseeded before every split).
- `big` batched: `--split-chunk 64` (all splits in one pass). Small networks: unchunked.

## Files

| File | Content |
|---|---|
| `test_batched_splits_cuda.log` | correctness test on CUDA (`ALL CHECKS PASSED`) |
| `sweep_sequential.{csv,log}` | all four networks, sequential loop |
| `sweep_batched_small.{csv,log}` | `3layer`, `4layer`, `5layer`, batched |
| `sweep_batched_big.{csv,log}` | `big`, batched, `--split-chunk 64` |
| `batched_vs_sequential_summary.csv` | per (network, eps index, n_splits): bound, relative difference, both times, both peak VRAM, speedup |
| `batched_gelsd_share_big.csv` | share of CPU `gelsd` in the batched pass on `big` |
| `split_batched_vs_sequential.png` | time and peak VRAM against `n_splits`, eps index 0 |

## 1. Correctness

- Every per-split bound of the batched pass equals the sequential loop: `3layer`
  `9.8e-15`, `4layer` `2.3e-14`, `5layer` `1.5e-14`, `big` `2.1e-15` (maximum relative
  difference; splits 1, 2, 3, 8, three epsilons, three images; `big`: splits 1 and 4).
- Chunking (`chunk = 1, 3, 8, 100`) and the independence of a split from the others in its
  batch hold to `<= 2.5e-15`.
- Without a seed the bound stays above a sampled autograd maximum of the true derivative.
- In the sweep itself the largest relative difference of `lc_precise` between the two
  modes is `1.2e-14` over all 56 (network, eps index, n_splits) cells.

## 2. Speed, eps index 0 (median ms, mean over 3 images)

| Network | n_splits | sequential | batched | speedup |
|---|---:|---:|---:|---:|
| `3layer` | 8 | 100.7 | 21.7 | 4.6x |
| `3layer` | 32 | 399.8 | 58.4 | 6.8x |
| `3layer` | 64 | 807.5 | 107.6 | 7.5x |
| `4layer` | 64 | 1152.0 | 165.5 | 7.0x |
| `5layer` | 64 | 1511.5 | 224.2 | 6.7x |
| `big` | 8 | 497.7 | 310.0 | 1.6x |
| `big` | 32 | 2025.9 | 1287.9 | 1.6x |
| `big` | 64 | 4066.7 | 2536.3 | 1.6x |

Eps index 8 gives the same picture (`3layer` 7.8x and `big` 1.6x at `n_splits = 64`);
see `batched_vs_sequential_summary.csv`.

## 3. Why `big` stops at 1.6x

`lin_reg_tensor_batched` still solves the least squares problems with CPU `gelsd`
(transfer to the CPU, solve, transfer back). In the batched pass the solve is one call
over `S * N` small problems, which LAPACK works through one by one, so it costs about
`34 ms` per split whatever the batch size:

| n_splits | total ms | CPU `gelsd` ms | share |
|---:|---:|---:|---:|
| 1 | 54.5 | 35.3 | 64.9% |
| 8 | 337.2 | 282.7 | 83.8% |
| 32 | 1251.7 | 1082.3 | 86.5% |
| 64 | 2487.8 | 2200.7 | 88.5% |

The rest of the pass (everything but `gelsd`) is `287 ms` at 64 splits, so with the
regression removed `big` would be about `4067 / 287 = 14x` faster than sequential. This
is an estimate from the measurement, not a measured result.

## 4. Memory

Peak VRAM of the batched pass grows linearly with the number of splits in one pass:

| Network | n_splits = 1 | n_splits = 64 |
|---|---:|---:|
| `3layer` | 10.6 MB | 123 MB |
| `4layer` | 12.1 MB | 214 MB |
| `5layer` | 13.5 MB | 304 MB |
| `big` | 503 MB | 29 804 MB |

The sequential loop stays constant (`480 MB` on `big`). `--split-chunk` limits the splits
per pass and with it the memory.

## 5. Caveats

- 3 images, 2 epsilon indices, 3 repeats: a timing study, not a statistical one. The
  speedup column is a ratio of medians per cell, averaged over images.
- The first sigmoid layer has `ly == uy` for every neuron, so the regression there is
  rank deficient and relies on the minimum-norm behaviour of `gelsd`.
