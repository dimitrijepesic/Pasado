# Layerwise CPU vs. CUDA divergence - 2026-10-17

> **Pre-fix results.** These runs predate the negation and `|E|` soundness fixes
> (see section 5 of `../gpu_hybrid_split_sweep_2026-10-17/README.md`). Both CPU
> and CUDA ran the same pre-fix code, so the CPU/CUDA comparison is internally
> consistent, but the absolute bounds, root counts and divergence percentages
> have not been re-measured with the fixed code.

This directory contains the raw CSV and log outputs from
`COLAB_GPU_HYBRID_LAYERWISE.ipynb` (`experiments/measure_layerwise_device_diff.py`)
comparing CPU and CUDA layer-by-layer across `3layer`, `4layer`, `5layer`, and
`big` for both the legacy `complex` Cardano solver (`inverse_poly_tensor`) and
the `real` Viète solver (`inverse_poly_real_tensor`).

## Files

- `layerwise_pilot_cuda_fp64.{csv,log}`: 1 image (`image_id = 0`), `eps_indices = [0, 8, 15]`, `float64`, `seed = 7`.
- `layerwise_pilot_cuda_fp32.{csv,log}`: 1 image (`image_id = 0`), `eps_indices = [0, 8, 15]`, `float32`, `seed = 7`.
- `layerwise_multi_cuda_fp64.{csv,log}`: 5 correctly classified images $\times$ 5 epsilon indices (`[0, 4, 8, 12, 15]` = 25 pairs per network), `float64`, `seed = 7`.

## Key Findings (`float64`, `image_id = 0`, `eps_idx = 0`, $\varepsilon = 0.632456$)

### 1. `big` network (4 sigmoid layers, 6144 cubic root candidates per layer)

| Layer | Solver | Candidates | Killed by `isreal` (`CPU \| CUDA`) | Survived `inv_sigmoid` (`CPU \| CUDA`) | Kept in `[lx, ux]` (`CPU \| CUDA`) | `dual.generators` max rel diff | Layer bound (`CPU \| CUDA`) | Layer bound rel diff |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `0` | `complex` | 6144 | `0 \| 2348` | `4096 \| 2520` | `2804 \| 1778` | `5.761e-01` (57.6%) | `8.42607 \| 8.36109` | `7.712e-03` (0.77%) |
| `1` | `complex` | 6144 | `0 \| 2786` | `3994 \| 2188` | `2067 \| 1110` | `6.463e-01` (64.6%) | `19.82993 \| 15.14217` | `2.364e-01` (23.6%) |
| `2` | `complex` | 6144 | `0 \| 2812` | `2049 \| 1095` | `2048 \| 1094` | `8.704e-01` (87.0%) | `54.65717 \| 43.85133` | `1.977e-01` (19.8%) |
| `3` | `complex` | 6144 | `0 \| 2711` | `2048 \| 1126` | `2048 \| 1126` | `9.940e-01` (99.4%) | `439.61578 \| 244.67434` | `4.434e-01` (44.3%) |
| `final_affine` | `complex` | — | — | — | — | `9.940e-01` (99.4%) | `14466.40314 \| 5035.15142` | **`6.519e-01` (65.2%)** |
| `0` | `real` | 6144 | `0 \| 0` | `4096 \| 4096` | `2804 \| 2804` | `4.121e-12` | `8.42607 \| 8.42607` | `1.265e-15` |
| `1` | `real` | 6144 | `0 \| 0` | `4096 \| 4096` | `2067 \| 2067` | `4.868e-10` | `19.82993 \| 19.82993` | `0.000e+00` |
| `2` | `real` | 6144 | `0 \| 0` | `4096 \| 4096` | `2348 \| 2348` | `9.323e-10` | `54.65717 \| 54.65717` | `0.000e+00` |
| `3` | `real` | 6144 | `0 \| 0` | `4096 \| 4096` | `2670 \| 2670` | `2.539e-07` | `439.61578 \| 439.61578` | `1.293e-16` |
| `final_affine` | `real` | — | — | — | — | `6.355e-09` | `14466.40326 \| 14466.40326` | **`0.000e+00`** |

### 2. Layerwise bound relative difference progression (`float64`, `image_id = 0`, `eps_idx = 0`)

| Network | Solver | Layer 0 | Layer 1 | Layer 2 | Layer 3 | Final Affine (`lc_precise`) |
|---|---|---:|---:|---:|---:|---:|
| `3layer` | `complex` | `3.18e-16` | `2.63e-02` (2.63%) | — | — | **`1.41e-01` (14.14%)** |
| `3layer` | `real` | `3.18e-16` | `0.00e+00` | — | — | **`0.00e+00`** |
| `4layer` | `complex` | `4.79e-03` (0.48%) | `4.46e-02` (4.46%) | `8.03e-02` (8.03%) | — | **`3.85e-01` (38.47%)** |
| `4layer` | `real` | `0.00e+00` | `4.48e-16` | `0.00e+00` | — | **`1.37e-16`** |
| `5layer` | `complex` | `1.56e-15` | `3.22e-02` (3.22%) | `4.49e-02` (4.49%) | `2.34e-01` (23.40%) | **`5.10e-01` (50.95%)** |
| `5layer` | `real` | `1.39e-15` | `0.00e+00` | `0.00e+00` | `6.73e-16` | **`4.08e-16`** |
| `big` | `complex` | `7.71e-03` (0.77%) | `2.36e-01` (23.64%) | `1.98e-01` (19.77%) | `4.43e-01` (44.34%) | **`6.52e-01` (65.19%)** |
| `big` | `real` | `1.26e-15` | `0.00e+00` | `0.00e+00` | `1.29e-16` | **`0.00e+00`** |

### 3. Multi-image sweep (`float64`, 5 images $\times$ 5 epsilons = 25 pairs per network)

| Network | Solver | Max `nonreal` roots killed on CUDA per layer | Final `lc_precise` rel diff (`median / p95 / max`) |
|---|---|---:|---:|
| `3layer` | `complex` | 273 / 600 | `9.88e-06 / 1.64e-01 / 2.22e-01` (up to **22.2%**) |
| `3layer` | `real` | 0 / 600 | `9.89e-16 / 2.77e-15 / 2.82e-15` |
| `4layer` | `complex` | 309 / 600 | `9.37e-06 / 3.74e-01 / 4.18e-01` (up to **41.8%**) |
| `4layer` | `real` | 0 / 600 | `7.96e-16 / 2.01e-15 / 3.49e-15` |
| `5layer` | `complex` | 311 / 600 | `3.75e-05 / 5.09e-01 / 5.13e-01` (up to **51.3%**) |
| `5layer` | `real` | 0 / 600 | `1.25e-15 / 6.96e-15 / 8.99e-15` |
| `big` | `complex` | 2843 / 6144 | `3.49e-05 / 6.50e-01 / 6.52e-01` (up to **65.2%**) |
| `big` | `real` | 0 / 6144 | `4.06e-16 / 1.08e-15 / 1.41e-15` |

### 4. Additional note: CPU-side root loss in the legacy `complex64` path

Even on CPU (where `torch.isreal` kills `0` roots because imaginary parts happen to evaluate to `+0.0`), the legacy `complex64` Cardano path still loses valid roots near $s \approx 0$ or $s \approx 1$ due to single-precision (`float32`/`complex64`) rounding before `inv_sigmoid(s) = ln(s / (1 - s))`. For example, on `big` (`image_id = 0`, `eps_idx = 0`), in layer 2 the `complex` path on CPU retains only `2049` roots after `inv_sigmoid` (`2048` in `[lx, ux]`) and in layer 3 `2048` roots, whereas the `float64` `real` Viète solver retains all `4096` real roots after `inv_sigmoid` (`2348` in `[lx, ux]` in layer 2 and `2670` in layer 3). Thus, the shortcoming of the legacy complex Cardano implementation is not limited to `torch.isreal` on CUDA; its `complex64` intermediate precision also drops boundary roots on CPU.

