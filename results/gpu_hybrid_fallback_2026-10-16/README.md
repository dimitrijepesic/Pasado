# GPU hybrid fallback timing - CUDA, 2026-10-16

> **Pre-fix code.** Timed before the negation and `|E|` soundness fixes (see
> section 5 of `../gpu_hybrid_split_sweep_2026-10-17/README.md`). The fixes add
> one elementwise `abs` per boundary check and change no tensor shapes or
> transfers; the timings have not been re-measured.

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

## Interpretation and caveats

- **Decision rule outcome**: On `3layer`, `4layer`, and `5layer`, the CPU fallback accounts for `18.0%–19.1%` of `precise` time (`2.2–4.6 ms` total per pass), whereas on `big` (`n = 1024`) it accounts for **`54.54%`** (`34.57 ms` out of `63.34 ms`, with `32.11 ms` spent inside CPU `gelsd` and only `1.40 ms` in D2H/H2D transfers). Because this exceeds the `15%` threshold on wide networks, further acceleration on `big` requires either batching independent subproblems (e.g., across `n_splits`) or replacing CPU `gelsd` with a GPU SVD/minimum-norm solver.
- **Note on `sync_free` naming and timer boundaries**: `sync_free` removes avoidable host-sync assertions inside per-layer operations (`assert bool(...)` in `precise_transformer.py` and `torch.all(lb == ub)` in `SimpleZono.py`), but necessarily retains the `.cpu()` transfer inside `lin_reg_tensor_batched` and the final `.item()` readout. In addition, the per-segment breakdown uses `torch.cuda.synchronize()` at segment boundaries for accurate attribution, which adds synchronization points not present in an uninstrumented end-to-end call.

