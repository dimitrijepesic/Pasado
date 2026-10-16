# GPU hybrid fallback timing - CUDA, 2026-10-16

This directory contains the raw CSV and log outputs from
`COLAB_GPU_HYBRID_FALLBACK.ipynb` for commit
`a4dd04509b4b8c84b526fce11239446ed70a368c` (`torch 2.11.0+cu130`, 6 CPU threads).

## Protocol

- 3 correctly classified MNIST test images (`image_ids = [0, 1, 2]`),
- 3 epsilon indices (`0, 8, 15`), `n_splits = 1`, `seed = 7`,
- all five optimization/cubic selectors enabled (`vec=1, bnd=1, lstsq=1, grid=1, real_cubic=1`),
- 5 warm-up passes + 10 measured repeats = 90 measured forward passes per configuration,
- explicit `torch.cuda.synchronize()` at every timer boundary.

## Summary: `big` network (`float64`)

| Metric (median per pass, ms) | CPU (`float64`) | CUDA Hybrid (`baseline`) | CUDA Hybrid (`sync_free`) |
|---|---:|---:|---:|
| **Total `precise` forward** | **532.309** | **63.344** | **58.620** |
| `Affine` + non-sigmoid ops | 196.792 | 5.667 | 5.648 |
| Sigmoid layers total (4 layers) | 333.408 | 57.627 | 52.985 |
| — GPU part of Sigmoid (non-fallback) | 300.221 | 23.425 | 17.280 |
| **CPU fallback total (all 4 layers)** | **33.187** | **34.569** | **35.768** |
| — D2H `x_batch` (`[1024, 25, 2]`) | 0.000 | 0.576 | 0.549 |
| — D2H `zs_batch` (`[1024, 25]`) | 0.000 | 0.298 | 0.292 |
| — CPU design matrix (`[1024, 25, 3]`) | 0.924 | 1.064 | 0.998 |
| — CPU `lstsq(gelsd)` (+perturb) | 32.237 | 32.108 | 33.616 |
| — H2D solution (`[1024, 3]`) | 0.000 | 0.529 | 0.351 |
| **Fallback share of `precise` (%)** | **6.18%** | **54.54%** | **60.91%** |

## Summary across all networks on CUDA (`float64`, `baseline`)

| Network | Hidden width `n` | Sigmoid layers | `precise_total_ms` | `fallback_total_ms` | `gelsd_ms` | D2H + H2D (ms) | Fallback share (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `3layer` | 100 | 2 | 12.460 | 2.245 | 1.785 | 0.343 | 18.00% |
| `4layer` | 100 | 3 | 18.453 | 3.466 | 2.755 | 0.522 | 18.75% |
| `5layer` | 100 | 4 | 24.239 | 4.644 | 3.692 | 0.702 | 19.11% |
| `big` | 1024 | 4 | 63.344 | 34.569 | 32.108 | 1.403 | 54.54% |
