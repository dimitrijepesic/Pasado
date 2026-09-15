# TORCH_PROFILER_ANALYSIS (Phase 9) — reduced-workload operator comparison

Script: `experiments/torch_profiler_compare.py`.
Workload: **one image, one epsilon (2e-3), full 3layer precise forward** (both
sigmoid layers) — deliberately not the whole benchmark. CPU activities,
`record_shapes=True`. Warm-up outside the profiler; seeded identically per
variant.

Variants: `old` (per-neuron checks + per-neuron lstsq), `vec` (vectorized
corner/boundary checks + per-neuron lstsq), `batched` (vectorized checks +
batched regression).

Artifacts: `logs/torch_profiler_{old,vec,batched}.txt` (operator tables with
name / call count / self CPU / total CPU) and Chrome traces
`profiles/torch_profiler_{old,vec,batched}.json` (small — one forward each).

## Aggregate

| variant | operator calls | summed self CPU |
|---|---|---|
| old | 64 738 | 604.0 ms |
| vec | 36 262 | 364.4 ms |
| batched | **15 056** | **165.7 ms** |

Operator dispatch count fell **4.3x** old -> batched; summed self CPU fell
**3.6x**. This is the direct confirmation that the sprint's wins came from
removing tiny-op dispatch, not from changing the math.

## Per-operator call counts (the tiny-op dispatch check)

| op | old | vec | batched | comment |
|---|---|---|---|---|
| `aten::linalg_lstsq` | 400 | 400 | **4** | per-neuron -> per-layer (2 layers x 2 = 4 in this reduced run) |
| `aten::ones` | 200 | 200 | **2** | the `xplusone` intercept column, now built once per layer |
| `aten::sigmoid` | 1 616 | 422 | **26** | per-corner/per-root evaluation -> batched |
| `aten::item` | 4 238 | 2 238 | 844 | remaining ones are `get_linspace`'s per-neuron `.item()` |
| `aten::cat` | 630 | 424 | 228 | |
| `aten::max` | 1 200 | 408 | 408 | vectorized in Phase 3, unchanged by batching |
| `aten::linspace` | 800 | 800 | **800** | **untouched** — the remaining per-neuron work |
| `aten::cartesian_prod` | 200 | 200 | **200** | **untouched** |
| `aten::stack` | 418 | 212 | 214 | |

## What is now dominant (batched variant, top self-CPU)

| op | self CPU | calls | note |
|---|---|---|---|
| `aten::select` | 21.5 ms (13.0 %) | 1 848 | indexing inside the still-per-neuron `get_linspace` loop |
| `aten::max` | 19.6 ms (11.8 %) | 408 | batched reductions in the error checks |
| `aten::copy_` | 7.3 ms | 541 | |
| `aten::linalg_lstsq` | 7.3 ms self / 33.6 ms total | **4** | now genuine numerical work, 8.4 ms per batched call |
| `aten::mm` | 6.8 ms | 12 | dense affine zonotope matmul |
| `aten::linspace` | 6.0 ms | 800 | per-neuron grid generation |
| `aten::meshgrid` | 5.0 ms | 200 | inside `cartesian_prod` |
| `aten::item` | 5.0 ms | 844 | host syncs in `get_linspace` |

Compare with `old`, where `aten::linalg_lstsq` alone was 47.9 % of total CPU
(289 ms across 400 calls) — that entire block is gone.

## Conclusions

1. **Tiny-op dispatch was reduced exactly as intended**: 64.7k -> 15.1k
   operator calls for the same computation, with lstsq/ones/sigmoid counts
   dropping by 100x, 100x and 62x respectively.
2. **What remains dominant is `get_linspace`**: `select` + `linspace` +
   `meshgrid`/`cartesian_prod` + `item` together are the largest remaining
   precise-path cost, and their call counts are *identical* across all three
   variants — we never touched them. This is the same conclusion the cProfile
   comparison reached, from an independent measurement.
3. **`linalg_lstsq` is now real work, not overhead**: 4 calls, 8.4 ms each,
   solving 100 systems apiece.
4. Batching the grid construction is the natural next step, but it is **not
   bit-exact** (1-ULP `linspace` difference, LSTSQ_ANALYSIS §grid), so it
   needs an explicit precision decision rather than a silent swap.
