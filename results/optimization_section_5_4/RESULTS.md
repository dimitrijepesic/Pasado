# RESULTS - Section 5.4 Lipschitz benchmark, profiling & optimization sprint

> **Measured before the soundness fixes of 2026-10-21/22 (`b526762`); marked 2026-11-06.**
> Every number here comes from the code before `Zonotope.__neg__` / `DualZonotope.__neg__`
> negated the generators and before the boundary checks bounded `|E|` instead of `E`.
> Speedups, call counts and profiles compare two implementations of that same pre-fix
> code; the fixes add one elementwise `abs` per boundary check and a sign flip, so the
> ratios are not expected to move, but they have **not been re-measured**. Lipschitz bound
> values quoted here are **not valid** after the fix. Hardware measurements (GFLOP/s,
> `gels` against `gelsd`) do not depend on Pasado's code. Current state: `CHECKPOINT.md`.


## 0. Status on 2026-11-06 - read this first

This file was the first write-up of the July 2026 sprint. `FORMAL_SUMMARY.md` was kept up to
date afterwards and this file was not, so several statements below were superseded. The
superseded statements are kept for the record and marked **[superseded]** where they occur.

| Statement in this file | Current state | Source |
|---|---|---|
| Accepted optimizations are A, B, C | Two more selectors exist: D `PASADO_BATCHED_GRID` (batched sampling grid, **bit-identical**, default off) and E `PASADO_REAL_CUBIC` (real Viete cubic, **not** bit-identical: bounds move by <= 5.7e-5 relative, default off) | `FORMAL_SUMMARY.md`, `REAL_CUBIC.md` |
| Headline speedups 3.135x / 3.145x / 3.334x (A+B+C) | With D added, measured directly on 2026-08-12 (interleaved, pinned to P-cores, n = 3 pairs): 3layer **3.24x** (3.21-3.30), 4layer **3.64x** (3.63-4.22), 5layer **3.37x** (2.79-3.63) | `FORMAL_SUMMARY.md`, `logs/benchmark_results.csv` |
| `big`: no speedup claimed (section 4.4) | Measured on 2026-08-06, interleaved, pinned, n = 3, health 197-226 GFLOP/s: 1070.57 s -> 648.78 s = **1.650x** (A+B+C, without D; `big` with D not measured) | `FORMAL_SUMMARY.md`, `logs/benchmark_results.csv` |
| batched `linspace` deferred, ~1 ULP in 14 % of cases (section 9) | Done and **bit-identical**: the ~1 ULP was an artifact of the naive affine formula; reproducing `torch.linspace`'s symmetric-halves scheme is exact | `FORMAL_SUMMARY.md` |
| closed form instead of `lstsq` rejected (section 9) | Still rejected on the CPU. On CUDA, where `gelsd` does not exist, the pre-registered comparison of 2026-10-25/27 chose the closed form **A2** (`planar_fit.py`), behind `PASADO_CLOSED_FORM_FIT` / `PASADO_AUTO_PLANE_FIT` (both off by default) | `results/lstsq_comparison_2026-10-27_v3_zpivot/` |
| `torch.compile` rejected (section 9) | Rejected on the Windows CPU (no MSVC). On CUDA it works: compiled `fit_planes` (opt-in `PASADO_COMPILED_FIT`) and a compiled whole `precise_sigmoid` (experiment only, outcome opt-in) | `results/v3_compiled_cuda_2026-10-28/`, `results/compiled_sigmoid_v3_cuda_2026-10-29/` |
| GPU untested (section 10) | Tested on Colab T4 and A100 (2026-08 to 2026-10) | `GPU_READINESS.md`, `results/gpu_hybrid_*` |

Every number in this file, and in the table above, was measured on the code **before** the
soundness fixes of 2026-10-21/22; see the note at the top.

## 1. Workload

`Section_5_4/get_lipschitz.py` computes local Lipschitz bounds for four MNIST
fully connected networks via three abstract transformers (interval, ordinary
zonotope, **precise** zonotope). The precise transformer is much tighter and
much slower; it is the subject of this sprint.

| network | architecture | sigmoid layers |
|---|---|---|
| 3layer | 784 - 100 - 100 - 10 | 2 |
| 4layer | 784 - 100 - 100 - 100 - 10 | 3 |
| 5layer | 784 - 100 - 100 - 100 - 100 - 10 | 4 |
| big | 784 - 1024 - 1024 - 1024 - 1024 - 10 | 4 |

Fixed workload per network: 30 images x 16 epsilon values = **480 precise
forward passes**; `torch.set_default_dtype(torch.float64)`. Sampling density
(`STEPS=5`, 5x5 grid per neuron), epsilon range, image count and bound
computation were **not** modified.

## 2. Environment

Windows 11 (10.0.26200), Python 3.13.3, torch 2.12.1+cpu, numpy 2.5.1,
scikit-learn 1.9.0, torchvision 0.27.1+cpu, Intel Family 6 Model 154 (12th-gen
mobile), 8 torch threads, **CUDA unavailable**. All measurements CPU-only.

## 3. Optimizations accepted

| # | change | selector | default | numerical effect |
|---|---|---|---|---|
| A | `check_corners_tensor` - batched corner evaluation replacing a per-neuron Python loop | `PASADO_VECTORIZED_PRECISE` | **on** | bit-identical |
| B | `check_nonlinear_boundary_tensor` - batched boundary/root objective | `PASADO_VECTORIZED_PRECISE`, `PASADO_VEC_BOUNDARY` | **on** | bit-identical |
| C | `lin_reg_tensor_batched` - one batched `lstsq` per layer instead of one per neuron | `PASADO_BATCHED_LSTSQ` | **off** | bit-identical |
| D | `get_linspace_batched` - batched 5x5 sampling grid (added after this file; see section 0) | `PASADO_BATCHED_GRID` | **off** | bit-identical |
| E | real Viete cubic root solve instead of `complex64` Cardano (added after this file) | `PASADO_REAL_CUBIC` | **off** | not bit-identical (<= 5.7e-5 relative) |

All original implementations remain in the file and are reachable through the
selectors. No mathematics, sampling, driver (`gelsd`) or bound computation was
changed.

## 4. Wall-clock results (the headline numbers)

**Measurement caveat first - and the root cause found.** The benchmark machine
is an **Intel i5-12450H: a hybrid CPU with 4 performance cores (logical 0-7)
and 4 efficiency cores (logical 8-11)**. The Windows scheduler migrates the
benchmark process between them, and the E-cores are 2-3x slower for this
Python-heavy workload. This produced run-to-run swings of up to 2.7x that were
initially misattributed to background system load.

Decisive evidence: 4layer `vec_both_batched` took **35.29 s at a health-probe
reading of 93 GFLOP/s**, then **93.85 s at 125 GFLOP/s** - slower while the
machine measured *faster*. The health probe is multi-threaded and saturates
every core, so it reports high throughput regardless of which core type the
benchmark landed on. It cannot detect this effect.

Three mitigations, added to the harness in this order as the problem was
understood:

* a **machine-health probe** (fixed matmul, GFLOP/s) recorded per run - catches
  ordinary background load, but *not* P/E migration;
* **`--interleave`**, alternating A,B,A,B... so both variants see the same
  conditions, with per-pair ratios reported;
* **`--pin-pcores`**, restricting the process to the 8 P-core logical
  processors (matching torch's 8 threads) - removes the migration variance at
  the source;
* **minimum-of-N**, not median, as the reported statistic: on a machine with
  this failure mode the fastest run is the one that suffered least
  interference. Medians are reported alongside for transparency.

### 4.1 Interleaved comparison - original vs final optimized (A+B+C)

All measurements interleaved (A,B,A,B...). `min` is the headline statistic;
`health` is the per-run probe reading, showing how differently loaded the
machine was between networks.

| network | original min | optimized min | **speedup** | **runtime reduction** | runs | health range |
|---|---|---|---|---|---|---|
| 3layer | 156.25 s | 49.84 s | **3.135x** | **68.1 %** | 3 pairs | 38-48 GFLOP/s |
| 4layer | 110.98 s | 35.29 s | **3.145x** | **68.2 %** | 2 pairs | 93-125 GFLOP/s |
| 5layer | 325.91 s | 97.76 s | **3.334x** | **70.0 %** | 2 pairs | 35-52 GFLOP/s |

Per-pair ratios: 3layer 3.23x / 3.06x / 3.14x; 4layer 3.20x / 1.18x (second
pair hit by E-core migration - the 93.85 s outlier); 5layer 3.23x / 3.33x.

**The strongest evidence in this table is not any single number - it is that
3layer and 4layer give 3.135x and 3.145x while their machines differed by
almost 3x in measured throughput.** A speedup that is invariant to a 3x change
in machine conditions is a property of the code, not of the measurement.

The mild upward trend with depth (3.14 -> 3.33) is expected: more sigmoid
layers means a larger share of runtime in the precise transformer that was
optimized, against the fixed cost of data loading and the interval/ordinary
zonotope passes.

### 4.2 Incremental step: vectorized checks -> + batched regression

3layer, back-to-back on an **idle** machine (night), 3 runs each, spread < 1.5 %:

| variant | median | min | max |
|---|---|---|---|
| vec_both | 28.01 s | 27.81 | 29.16 |
| vec_both_batched | **19.36 s** | 19.16 | 19.41 |

* **speedup_factor = 1.447x**, **runtime_reduction = 30.9 %**

Cross-check: batched at 19.36 s (idle) vs 50.26 s (loaded) implies the machine
was 2.6x slower during the interleaved block; dividing the loaded original
(156.74 s) by 2.6 gives ~60 s, and 60 / 19.36 = 3.1x - consistent with the
measured 3.135x.

### 4.3 Raw data

All rows in `logs/benchmark_results.csv`; interleaved runs are those carrying a
`health_gflops` value. Earlier rows (no health value) were taken before the
P/E-core problem was understood and are superseded.

### 4.4 big - NOT claimed **[superseded: 1.650x measured on 2026-08-06, see section 0]**

| variant | time | status |
|---|---|---|
| original | 1744.9 s (29.1 min) | single run, quiet machine, plausible |
| vec_both | 2514.5 s (41.9 min) | **CONTAMINATED - discarded** |
| vec_both_batched | not measured | machine unavailable (see below) |

The `vec_both` run came out *slower than the original variant*, which is
physically impossible from the code: the entire footprint of the changed
functions in the big profile is under 60 s, and the per-neuron checks they
replace cost ~170-220 s. The run spanned local midnight (a Windows maintenance
window) and is discarded. A second corruption of the same kind was observed on
a 3layer cProfile run (2998 s versus an expected ~22 s).

A fresh `big` pair was not run because each variant needs 30-90 min under the
current load and could not be compared against the earlier quiet-machine
number anyway. The exact command is in `REPRODUCE.md`; with the new health
probe the result will be self-validating. **No `big` speedup is claimed.**

## 5. cProfile results (profiler time - NOT comparable to wall-clock)

cProfile adds per-Python-call overhead; on `big` it inflated a 18m15s run to
20m37s. Use for relative attribution only.

### 3layer, whole benchmark

| | baseline | + vectorized checks | + batched lstsq |
|---|---|---|---|
| profiled total | 45.55 s | 24.97 s | **22.46 s** |
| `forward_zono_precise` cum | 33.15 s | 14.75 s | 8.82 s |
| `sigmoid_prime_product_tensor` cum | 32.35 s | 14.03 s | 8.16 s |
| `compute_max_error` cum | **19.74 s** | 1.97 s | 2.41 s |
| `check_corners(_tensor)` | 8.40 s | 0.16 s | 0.17 s |
| `check_nonlinear_boundary(_tensor)` | 10.80 s | 1.07 s | 1.29 s |
| `linalg_lstsq` self | 3.79 s | 3.57 s | **0.92 s** |
| total function calls | - | 9 603 903 | **8 297 488** |

Baseline -> vectorized: 45.55 -> 24.97 s = **1.82x speedup, 45.2 % reduction
(profiler time)**.

### big, after vectorization (1249 s profiled, 61.2 M calls)

| rank | function | self s | share |
|---|---|---|---|
| 1 | `SimpleZono.AffineZonotope` (dense `generators @ layer`) | **416.3** | 33.3 % |
| 2 | `linalg_lstsq` (1 966 080 calls) | 119.6 | 9.6 % |
| 3 | `torch.cat` (`xplusone`, 1 989 120 calls) | 89.4 | 7.2 % |
| 4 | `torch.abs` + `torch.sum` (`get_coeff_abs`) | 133.2 | 10.7 % |
| 5 | grid generation (`cartesian_prod`+`linspace`+`get_linspace`) | 93.5 | 7.5 % |

`compute_max_error` fell to 40.8 s cumulative (3.3 %). Corner/boundary checking
is now 0.4 %/0.4 % - negligible.

## 6. Operator-level results (torch.profiler, reduced workload)

One image x one epsilon, full 3layer precise forward:

| | old | vectorized | + batched |
|---|---|---|---|
| operator calls | 64 738 | 36 262 | **15 056** (-76.7 %) |
| summed self CPU | 604.0 ms | 364.4 ms | **165.7 ms** (3.6x) |
| `aten::linalg_lstsq` | 400 | 400 | **4** |
| `aten::ones` | 200 | 200 | **2** |
| `aten::sigmoid` | 1 616 | 422 | **26** |
| `aten::linspace` / `cartesian_prod` | 800 / 200 | 800 / 200 | 800 / 200 (untouched) |

In the `old` variant `aten::linalg_lstsq` alone was **47.9 % of total CPU**.

## 7. Microbenchmark (isolated, idle machine, float64, gelsd, median of 15)

| n | per-neuron loop | batched | speedup |
|---|---|---|---|
| 1 | 0.269 ms | 0.263 ms | 1.0x |
| 10 | 2.27 ms | 0.406 ms | 5.6x |
| 100 (3layer layer) | 28.78 ms | 1.97 ms | **14.6x** |
| 1024 (big layer) | 163.9 ms | 26.0 ms | **6.3x** |

## 8. Correctness

| check | result |
|---|---|
| `check_corners_tensor` vs original | max diff 0.000e+00 |
| `check_nonlinear_boundary_tensor` vs original | max diff 0.000e+00 |
| `compute_max_error` tensor path vs original | max diff 0.000e+00 |
| selector routing (call-counted, 3 modes, incl. degenerate) | max diff 2.2e-16 (1 ULP) |
| `lin_reg_tensor_batched`, 26 configs (n=1..1024, float32+float64, rank-deficient, saturated, ill-conditioned) | **bit-identical** (`torch.equal`) |
| end-to-end old vs vectorized (3 images x 3 epsilons) | final bound <= 5.7e-14, generators <= 4.5e-14 |
| end-to-end old vs vectorized+batched | identical to the above - batching adds **zero** further difference |

Shapes, dtype (`float64`), device (`cpu`) and NaN/Inf *positions* verified
equal in every end-to-end case. Tolerances: `torch.testing.assert_close` with
explicit `rtol=1e-9, atol=1e-9` end-to-end and `rtol=1e-12, atol=1e-14`
(float64) in unit tests.

### A real bug was found and fixed in our own vectorization

The first version of `check_nonlinear_boundary_tensor` produced differences of
~1e-6 - too large for float64 reordering. Cause: `inverse_poly_tensor` computes
cube roots through `complex64`, so the roots come back **float32**. The
original code stacked each root with the float64 `ly`/`uy`, which promoted to
float64 before evaluating `sigmoid`; our version stacked the three float32
roots together and evaluated the objective in float32. Fix: promote the stacked
roots to `ly.dtype` (exactly what the original stack did implicitly) and match
the `(1-s)*(s*y)` association. After the fix the difference dropped from ~1e-6
to <= 5.7e-14. Earlier "<= 4.8e-7 is just reordering" readings were this bug,
masked by float32 defaults in the unit tests.

## 9. Rejected optimizations (documented negative results)

| candidate | verdict | reason |
|---|---|---|
| caching grids / design matrices | rejected | measured **0 % exact-repeat rate** (400/400 distinct SHA-1); value-keyed cache can never hit |
| closed-form / normal equations / pinv | rejected on CPU **[on CUDA: A2 chosen, see section 0]** | unnecessary (batched gelsd already native and bit-exact), worse conditioning |
| `torch.compile` | rejected on Windows CPU **[works on CUDA, see section 0]** | Inductor CPU backend unavailable (MSVC `cl.exe` missing); `linalg_lstsq` graph-breaks; complex ops unsupported; target kernels now < 1 ms |
| batched `linspace` | deferred **[superseded: done, bit-identical]** | **not** bit-exact (~1 ULP in 14 % of cases) - needs an explicit precision decision |

## 10. Limitations

1. **[superseded]** **No clean `big` comparison.** The only optimized `big` measurement was
   corrupted; no `big` speedup is claimed.
2. **Wall-clock numbers were taken on a loaded machine.** Mitigated by
   interleaving and per-run health probes, but absolute seconds are ~2.6x
   inflated versus an idle machine. Ratios, not absolutes, are the result.
3. **Single-run big figures** carry no statistics.
4. cProfile columns from different days/machines are compared only where noted;
   call counts (machine-independent) carry the structural argument.
5. **[superseded]** **GPU untested** - no CUDA device available. `GPU_READINESS.md` is a static
   audit, and `experiments/colab_gpu_profile.py` deliberately refuses to
   fabricate GPU numbers.
6. The benchmark is **not deterministic across processes** (`np.random.normal`
   perturbation of A is unseeded in production); all correctness comparisons
   seed it explicitly.

## 11. Exact commands

See `REPRODUCE.md`.
