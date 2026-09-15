# FINAL_PROFILE_COMPARISON (Phase 12)

Comparison of the profile structure before and after the sprint's two accepted
optimizations. Two independent instruments agree: **cProfile** (Python-level
attribution) and **torch.profiler** (ATen operator level).

> Times from different runs/machines are not directly comparable (see the
> contamination note at the end). **Call counts are machine-independent and
> exact** — they carry most of the argument here.

## A. cProfile, whole 3layer benchmark (480 precise forwards)

| | baseline (original) | + vectorized checks | + batched lstsq |
|---|---|---|---|
| profiled total | 45.55 s | 24.97 s | **22.46 s** |
| `forward_zono_precise` cum | 33.15 s | 14.75 s | **8.82 s** |
| `sigmoid_prime_product_tensor` cum | 32.35 s | 14.03 s | **8.16 s** |
| `compute_max_error` cum | 19.74 s | 1.97 s | 2.41 s |
| `check_corners(_tensor)` | 8.40 s | 0.16 s | 0.17 s |
| `check_nonlinear_boundary(_tensor)` | 10.80 s | 1.07 s | 1.29 s |
| `lin_reg_tensor(_batched)` cum | 5.00 s | ~4.7 s | **0.99 s** |
| `linalg_lstsq` self | 3.79 s | 3.57 s | **0.92 s** |
| `get_linspace` cum | ~3.1 s | 3.13 s | 3.87 s |
| total function calls | — | 9 603 903 | **8 297 488** |

## B. torch.profiler, one image x one epsilon, 3layer precise forward

| | old | vectorized | + batched |
|---|---|---|---|
| operator calls | 64 738 | 36 262 | **15 056** |
| summed self CPU | 604.0 ms | 364.4 ms | **165.7 ms** |
| `aten::linalg_lstsq` calls | 400 | 400 | **4** |
| `aten::ones` calls | 200 | 200 | **2** |
| `aten::sigmoid` calls | 1 616 | 422 | **26** |
| `aten::item` calls | 4 238 | 2 238 | 844 |
| `aten::linspace` calls | 800 | 800 | **800** |
| `aten::cartesian_prod` calls | 200 | 200 | **200** |

## 1. Which bottlenecks disappeared?

* **Corner + nonlinear-boundary error evaluation.** 19.2 s of the 45.55 s
  baseline (42 %) collapsed to ~1.5 s. In operator terms, `aten::sigmoid`
  dropped 1 616 -> 422 -> 26 calls per forward.
* **Per-neuron least-squares dispatch.** 96 000 `lstsq` calls per benchmark ->
  960 (one per layer call); `torch.ones` 96 960 -> 1 920; `torch.cat`
  108 480 -> 13 440. In the reduced profiler run, `linalg_lstsq` went from
  **47.9 % of total CPU (289 ms / 400 calls)** to 4 calls of genuine work.

## 2. Which bottlenecks merely moved?

* `compute_max_error` went 19.74 -> ~2 s, but its *self* time is now the
  dominant part of what remains: the Python combine loop plus the `unbind(0)`
  that converts our `[n]` tensors back into lists. On `big` that residue is
  ~20 s. It moved from "the checks are slow" to "the glue around the checks is
  slow" — cheap to finish (Tier 1.2 in NEXT_RESEARCH_DIRECTIONS).
* Grid generation did not move at all: `linspace` / `cartesian_prod` /`.item()`
  call counts are **identical in all three variants**. It became the top
  precise-path cost purely because everything around it shrank.

## 3. What is now dominant?

**`get_linspace`** — per-neuron grid construction. In the batched cProfile it
is 3.87 s of `sigmoid_prime_product_tensor`'s 8.16 s (~47 %); in the operator
profile the top entries are `aten::select` (1 848 calls), `aten::linspace`
(800), `aten::meshgrid` (200) and `aten::item` (844), all inside it.

Beyond the precise path, on the **`big`** network the true dominant cost is
different and untouched: `AffineZonotope`'s dense `generators @ layer` matmul
at **416 s self (33 % of the whole profile)**. That is genuine BLAS scaling as
width² x #generators, not dispatch overhead — no vectorization addresses it.

## 4. How did function call count change?

Python-level: 9 603 903 -> 8 297 488 per 3layer benchmark (**-13.6 %**).
Operator-level, per forward: 64 738 -> 15 056 (**-76.7 %**, 4.3x fewer).
The Python-level reduction is smaller because `get_linspace`'s per-neuron loop
(the largest remaining source of Python frames) was deliberately left alone.

## 5. How did tiny-op count change?

The three ops that defined the "many tiny PyTorch operations" pattern fell by
100x (`linalg_lstsq`), 100x (`ones`) and 62x (`sigmoid`) per forward. Summed
operator self-CPU fell 3.6x on the identical computation. This is the sprint's
central empirical claim, and it is confirmed by both instruments.

## 6. Which costs are unavoidable numerical work?

* `AffineZonotope` dense matmul (416 s on `big`) — one BLAS call already.
* `get_coeff_abs` (`abs` + `sum`, 133 s on `big`) — one reduction already.
* The batched `lstsq` itself — 4 calls x 8.4 ms in the reduced run, each
  solving 100 systems. This is now real arithmetic, not overhead.

Everything above is *inherent* to the analysis as currently formulated; making
it cheaper requires changing the algorithm (fewer noise symbols) or the
hardware/precision, not the Python.

## 7. Remaining targets, classified

| target | kind |
|---|---|
| Batch `get_linspace` (1-ULP caveat) | batching candidate — **top priority** |
| `compute_max_error` unbind/loop residue | batching candidate — trivial, bit-exact |
| `AffineZonotope` matmul (416 s on big) | GPU candidate (fp64 throughput caveat) / algorithmic |
| `get_coeff_abs` abs+sum (133 s on big) | GPU candidate |
| Noise-symbol growth per sigmoid layer | **algorithmic research question** |
| Fused grid+objective kernel | compiler/Triton candidate — only after grid batching |

## Contamination note (why some times above are not compared directly)

Three measurements in this sprint were corrupted by background system activity
on the benchmark machine: a `big vec_both` run (2514 s, versus 1745 s for the
*slower* original variant), one 3layer cProfile run (2998 s versus the expected
~22 s), and one interleaved 3layer wall-clock series (100.7 -> 55.9 -> 37.2 s
monotone decay while the machine settled). None of them are used for any claim.
Where a before/after pair could not be re-measured under identical conditions,
this document relies on **call counts**, which are unaffected.
