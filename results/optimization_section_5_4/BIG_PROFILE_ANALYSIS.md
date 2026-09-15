# BIG_PROFILE_ANALYSIS

Analysis of `profiles/lipschitz_big_nosave_both_vectorized.prof`
(the `big` network, `--no-save`, with the corner + nonlinear-boundary
vectorizations already applied).

Extracted report: `logs/lipschitz_big_nosave_both_vectorized_pstats.txt`.

> All numbers below are **cProfile** times, not ordinary wall-clock. cProfile
> adds per-Python-call overhead (~61 M calls here), so absolute seconds are
> inflated relative to the plain wall-clock run (18m15s no-cProfile vs 20m37s
> under cProfile). Use these for *relative* attribution only.

Profiled total: **1249.4 s**, 61.16 M calls.
The benchmark runs three analyses per image; the precise path we care about is
`forward_zono_precise` = **693.98 s cumulative** (480 calls, 1.45 s/call).
The ordinary zonotope path is 434.85 s and the interval path 64.5 s.

---

## 1. Top compute paths after both vectorizations (by self / `tottime`)

| Rank | Function | self s | ncalls | Bucket |
|---|---|---|---|---|
| 1 | `SimpleZono.AffineZonotope` (`generators @ layer`) | **416.3** | 9 600 | dense BLAS (shared) |
| 2 | `torch._C._linalg.linalg_lstsq` | **119.6** | 1 966 080 | per-neuron regression (precise) |
| 3 | `torch.cat` | 89.4 | 1 989 120 | per-neuron regression — `xplusone` (precise) |
| 4 | `sigmoid_prime_product_tensor` (self) | 75.9 | 1 920 | precise orchestration |
| 5 | `torch.abs` | 73.1 | 56 640 | interval bounds `get_coeff_abs` (shared) |
| 6 | `torch.sum` | 60.1 | 24 960 | interval bounds `get_coeff_abs` (shared) |
| 7 | `SimpleZono.__mul__` (self) | 51.3 | 3 840 | haze / zono mult (shared) |
| 8 | `torch.cartesian_prod` | 39.2 | 1 966 080 | grid gen `get_linspace` (precise) |
| 9 | `sigmoid_prime_times_y` (self) | 33.6 | 1 966 080 | grid eval (precise) |
| 10 | `get_linspace` (self) | 33.2 | 1 920 | grid gen (precise) |
| 11 | `HyperDuals.imatmul` (self) | 23.7 | 7 200 | interval matmul (shared) |
| 12 | `torch.linspace` | 21.1 | 3 932 160 | grid gen (precise) |
| 13 | `lin_reg_tensor` (self) | 18.4 | 1 966 080 | per-neuron regression (precise) |
| 14 | `compute_max_error` (self) | 14.9 | 1 920 | error combine loop (precise) |
| 15 | `torch.matmul` | 13.0 | 50 400 | interval/dual matmul (shared) |

Grouped by concern:

* **Dense affine matmul — `AffineZonotope`: 416 s self.** This is the single
  biggest cost on `big`. It is genuine BLAS (`generators @ layer`,
  `centers @ layer`), scaling with `width² × #noise_symbols`. Called 9 600×,
  split ~50/50 between the ordinary and precise forward passes
  (`AffineDualZonotope` ← 2 322 from `forward_zono`, 2 329 from
  `forward_zono_precise`), so ≈ 200 s is attributable to the precise path.
  This is **not** Python overhead and **not** a "many tiny ops" problem.

* **Per-neuron regression machinery — ≈ 300–390 s, precise-only, all tiny ops.**
  One instance per (neuron × layer × image × epsilon) ⇒ **1.97 M** repetitions:
  `linalg_lstsq` 119.6 + `xplusone` `torch.cat` 89.4 + `cartesian_prod` 39.2 +
  `linspace` 21.1 + `get_linspace` self 33.2 + `sigmoid_prime_times_y` 60.7 cum
  + `lin_reg_tensor` self 18.4 + `.item()` 8.3. This is the textbook
  "Python loop orchestrating many small PyTorch ops" pattern and is the
  **largest addressable / batchable block**.

* **Interval-bound computation — `get_coeff_abs` (`abs`+`sum`): 133 s, shared.**
  `torch.abs` 73.1 + `torch.sum` 60.1, 24 960 calls, computing `lb`/`ub` of
  large zonotopes. Grows with `#noise_symbols × width`.

* **Already-vectorized error computation — small now.** `compute_max_error`
  40.8 s cum (14.9 self), `check_nonlinear_boundary_tensor` 5.5 s,
  `check_corners_tensor` 1.1 s. Combined ~3.7 % of total. The earlier
  vectorization did its job (see §2).

---

## 2. Is `compute_max_error` still important on `big`?

**No — it is no longer a bottleneck.** `compute_max_error` is 40.8 s cumulative
(3.3 % of total), of which the vectorized children are tiny
(`check_corners_tensor` 1.1 s, `check_nonlinear_boundary_tensor` 5.5 s). The
remaining 14.9 s of self-time is the *Python* per-neuron combine loop
`for i in range(len(errors)): max(max(errors[i], ly[i]), uy[i])` plus the
`unbind(0)` (5.6 s) we introduced to hand tensor results back as lists. On
3layer this whole block was the dominant cost; the vectorization moved the
bottleneck off it. There is a small residual cleanup here (§6, item 2):
keep everything as `[n]` tensors and replace the loop + unbind with two
`torch.maximum` calls.

---

## 3. Time remaining, by category (self-time, cProfile seconds)

| Category | self s | % of 1249 s | Path |
|---|---|---|---|
| Dense affine matmul (`AffineZonotope`) | 416.3 | 33.3 % | shared (~½ precise) |
| Per-neuron least-squares (`lstsq`) | 119.6 | 9.6 % | precise |
| Per-neuron `xplusone` `torch.cat` | 89.4 | 7.2 % | precise |
| Interval bounds (`abs`+`sum`) | 133.2 | 10.7 % | shared |
| Grid generation (`cartesian_prod`+`linspace`+`get_linspace`) | 93.5 | 7.5 % | precise |
| Grid eval (`sigmoid_prime_times_y` self) | 33.6 | 2.7 % | precise |
| Zono/haze mult (`__mul__` self) | 51.3 | 4.1 % | shared |
| Interval/dual matmul (`imatmul`+`matmul`) | 36.7 | 2.9 % | shared |
| Corner checking (`check_corners_tensor`) | 0.4 | 0.03 % | precise |
| Nonlinear-boundary checking | 0.2 self / 5.5 cum | 0.4 % | precise |
| Error combine loop (`compute_max_error` self) | 14.9 | 1.2 % | precise |
| `lin_reg_tensor` self | 18.4 | 1.5 % | precise |
| Python loops (unbind, list comps, `.item()`) | ~20 | 1.6 % | precise |

Corner/boundary checking is now negligible. Regression + grid generation is the
dominant *precise-specific* cost; dense affine matmul is the dominant *overall*
cost.

---

## 4. Is the 3layer bottleneck structure representative of `big`?

**Only partly.** On 3layer (width 100, few generators) the precise per-neuron
machinery — `sigmoid_prime_product_tensor` and its children — dominated almost
everything (33 s of 45 s), and `AffineZonotope` was negligible. On `big`
(width 1024, thousands of generators) the dense affine matmul rises to become
the single largest self-time cost (416 s), because it scales with `width²` and
with the growing number of noise symbols, while the per-neuron regression count
scales only linearly with width. So:

* 3layer **over-represents** per-neuron Python orchestration.
* 3layer **under-represents** dense BLAS (`AffineZonotope`) and interval-bound
  `abs`/`sum` on large generator matrices.

Nonetheless, the per-neuron regression block is still the largest *batchable*
chunk on `big` (~300 s), so the optimization the 3layer profile points us to
(batching the regressions) remains the right next step for `big` too — it just
will not, by itself, close the gap with the dense-matmul floor.

---

## 5. What scales with what

| Cost driver | depth | width | #neurons | #generators | #epsilon | #images |
|---|---|---|---|---|---|---|
| `AffineZonotope` matmul | ∝ #layers | ∝ width² | — | ∝ #gens | ×16 | ×30 |
| `lstsq` / `cartesian_prod` / `linspace` call **count** | ∝ #sigmoid layers | ∝ width | ∝ neurons | — | ×16 | ×30 |
| `get_coeff_abs` (`abs`+`sum`) | ∝ #layers | ∝ width | — | ∝ #gens | ×16 | ×30 |
| `__mul__` / `expand` `cat` sizes | — | ∝ width | — | ∝ #gens | ×16 | ×30 |
| error checking (already vectorized) | ∝ #sigmoid layers | ∝ width (one batched op) | ∝ neurons | — | ×16 | ×30 |

Key point: `#generators` (noise symbols) **grows layer by layer** — the precise
transformer adds fresh error symbols at every sigmoid, so deeper/wider networks
inflate both the matmul and the `abs`/`sum` bound costs super-linearly across
the forward pass.

---

## 6. What to optimize next (evidence-ranked)

1. **Batch the per-neuron regression (Phase 5–6).** Replace the
   `[lin_reg_tensor(xys[i], zs[i]) for i in range(n)]` list comprehension —
   1.97 M tiny `lstsq` + `xplusone` `cat` calls — with **one** batched solve:
   design matrices stacked to `[n, 25, 3]`, targets `[n, 25, 1]`, one
   `torch.linalg.lstsq` (or batched pseudo-inverse) call per layer. Also batch
   `get_linspace` so `linspace`/`cartesian_prod` run once per layer instead of
   per neuron. Evidence: ~300 s of `big`'s profile, ~9.5 s on 3layer, all tiny
   ops. **Highest-value addressable target.**

2. **Vectorize `compute_max_error`'s final combine + drop the `unbind`.**
   Keep the corner/boundary results as `[n]` tensors and combine with two
   `torch.maximum` calls instead of unbinding to lists and looping. Cheap,
   removes ~20 s on `big` (14.9 self + 5.6 unbind), semantics-preserving.

3. **`AffineZonotope` dense matmul (416 s) — GPU / compile candidate, not a
   Python-overhead target.** It is already a single BLAS call; the only real
   levers are (a) running on GPU, (b) reducing the number of noise symbols
   (algorithmic — would change semantics, out of scope), or (c) float32 vs the
   current float64 (changes precision, out of scope). Flag for Phase 8/10, not
   for vectorization.

4. **`get_coeff_abs` (`abs`+`sum`, 133 s) — inherent, but GPU-friendly.**
   Already a single reduction; nothing to batch. Scales with generator count.
   GPU candidate.

The next code change (Phase 5) is the batched-regression analysis. No source
change is made before that analysis is written.
