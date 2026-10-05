# BATCHED_LSTSQ_RESULTS - batched per-neuron regression, integration + validation

Selector: `PASADO_BATCHED_LSTSQ=1` enables `lin_reg_tensor_batched` inside
`sigmoid_prime_product_tensor` (default **OFF**; the loop path is untouched and
remains the default until this document's validation is accepted). Driver stays
`gelsd`; grids, STEPS, sampling and the np.random stream are unchanged.

## Correctness (gate 1 - PASS)

* Unit: 26 configurations (n in {1,10,100,1024}, float64+float32, degenerate /
  rank-deficient / saturated / ill-conditioned / mixed batches)  -
  **bit-identical** (`torch.equal`, max diff 0.000e+00) against the per-neuron
  loop, through the production function.
* End-to-end (3 images x 3 epsilons, 3layer, seeded): old-loop-everything vs
  vectorized-checks+batched-regression - identical diffs to the
  vectorized-only comparison (final Lipschitz bound <= 5.7e-14, generators
  <= 4.5e-14, shapes/dtype/device/NaN-Inf positions all equal).
  The batched path adds **zero** additional difference.
  (`logs/correctness_results_batched.csv`, all PASS.)

## Microbenchmark (idle machine, float64, gelsd, warm-up 3, 15 reps, median)

| n | loop | batched | speedup |
|---|---|---|---|
| 1 | 0.269 ms | 0.263 ms | 1.0x |
| 10 | 2.27 ms | 0.406 ms | 5.6x |
| 100 (3layer layer) | 28.78 ms | 1.97 ms | **14.6x** |
| 1024 (big layer) | 163.9 ms | 26.0 ms | **6.3x** |

`logs/microbenchmark_results.csv`.

## 3layer wall-clock (gate 2 - PASS; same-condition back-to-back pair, n=3)

| variant | median | min | max |
|---|---|---|---|
| vec_both (loop lstsq) | 28.01 s | 27.81 | 29.16 |
| vec_both_batched | **19.36 s** | 19.16 | 19.41 |

**speedup_factor = 1.447x, runtime_reduction = 30.9 %** - far above the 5 %
gate. (An earlier interleaved attempt showed 100.7->37.2 s monotone decay on
the vec_both side - machine still settling after the 90-minute big chain; both
variants were therefore re-run back-to-back on the settled machine. Only the
settled pair is quoted.)

## cProfile structure (gate 3 - PASS)

Profiles: `profiles/lipschitz_3layer_batched_lstsq.prof` (healthy run:
22.46 s profiled vs 19.4 s wall) and, for the loop side, the Jul-15
both-vectorized profile (24.97 s) - the same-day vec_both re-profile landed in
a degraded-machine window (158 s profiled vs 28 s wall) so its *times* are
discarded; its *call counts* are exact and are used below.

**Call counts (machine-independent, whole 3layer benchmark):**

| counter | loop path | batched path |
|---|---|---|
| total function calls | 9,603,903 | 8,297,488 (-13.6 %) |
| `torch.linalg.lstsq` | 96,000 | **960** (/100, one per layer-call) |
| `lin_reg_tensor(_batched)` | 96,000 | 960 |
| `torch.cat` | 108,480 | 13,440 |
| `torch.ones` | 96,960 | 1,920 |
| `sigmoid_prime_times_y` | 96,000 | 0 (batched inline) |
| `torch.linspace` / `cartesian_prod` / `.item()` | 192,000 / 96,000 / 385,456 | **unchanged** (get_linspace untouched) |

**Times (cProfile seconds; caveat: loop column = Jul-15 machine, batched =
today's healthy machine - comparable magnitude, not identical):**

| function | loop (Jul 15) | batched (today) |
|---|---|---|
| profiled total | 24.97 | 22.46 |
| `forward_zono_precise` cum | 14.75 | 8.82 |
| `sigmoid_prime_product_tensor` cum | 14.03 | 8.16 |
| regression block (lin_reg + cat + ones / batched + stack) | ~ 5.3 | ~ **1.0** |
| `linalg_lstsq` self | 3.57 | 0.92 |
| `get_linspace` cum | 3.13 | 3.87 |
| `compute_max_error` cum | 1.97 | 2.41 |

## Answers

1. **Regression-block speedup:** ~5x under cProfile (5.3 s -> 1.0 s), 14.6x in
   the isolated microbenchmark at n=100 (cProfile's per-call overhead masks
   part of the win); lstsq self time 3.9x down.
2. **3layer end-to-end:** 1.447x / -30.9 % wall-clock (28.01 -> 19.36 s,
   same-condition medians of 3).
3. **lstsq call count:** yes - 96,000 per-neuron calls -> 960 per-layer batched
   calls (exactly one per `sigmoid_prime_product_tensor` invocation).
4. **New bottleneck:** `get_linspace` - the still-per-neuron grid generation
   (192k `linspace` + 96k `cartesian_prod` + 385k `.item()`), now ~44 % of
   `sigmoid_prime_product_tensor`'s cumulative time; beyond the precise path,
   the shared dense zonotope ops. Batching the grid construction is the next
   candidate but is NOT bit-exact (1-ULP linspace difference, see
   LSTSQ_ANALYSIS) - left for future work.
5. **Does this justify a big run?** Yes: all three gates pass, and the big
   profile attributes ~300 cProfile-s to this block (~265 s projected saving
   from the n=1024 microbenchmark). A single big `vec_both_batched` wall-clock
   run follows.

## Decision

`PASADO_BATCHED_LSTSQ=1` is validated. Default remains OFF in code; the
benchmark harness selects it explicitly per variant.
