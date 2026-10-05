# CACHING_ANALYSIS - is any repeated work worth caching? (Phase 7)

Question: do `get_linspace` / `torch.linspace` / `torch.cartesian_prod` /
`xplusone` construction / the constants in `inverse_poly_tensor` ever see
*identical* inputs again, and would a cache pay for itself?

Evidence: `experiments/probe_lstsq.py` (instrumented reduced real run:
1 image x 2 epsilons x both 3layer sigmoid layers = 400 regression instances),
plus the big-network pstats.

## What each candidate depends on

| candidate | depends on | repeats exactly? |
|---|---|---|
| `torch.linspace(lx_i, ux_i, 5)` | per-neuron bound values | no - bounds are continuous outputs of the affine+sigmoid pipeline; they change with neuron, layer, image and epsilon |
| `torch.cartesian_prod(xs_i, ys_i)` | the two linspaces above | no - same reason |
| `xplusone` = `[ones \| grid]` | the grid | **measured: 400/400 SHA-1-distinct, 0 duplicates** in the instrumented run |
| `torch.ones(25, 1)` | nothing (constant) | yes - trivially, 1.97M identical allocations on big |
| `inverse_poly_tensor` constants (`-1/12`, `1j`, `sqrt(3)/2`) | nothing | yes - 2 rebuilds per layer call, 3 840 on big; ~us each, < 0.05 s total |
| cartesian-product *index pattern* | STEPS=5 (fixed) | yes - structurally |

## Why exact repetition does not happen for the data-dependent parts

The bounds (lx, ux, ly, uy) are real-valued functions of the input box: every
epsilon changes the input zonotope, every image changes the centers, every
layer transforms them through weights and sigmoids. Two float64 bound tuples
colliding bit-for-bit across instances is measure-zero; the probe confirms 0
hits even within a single image (400 distinct matrices out of 400). A cache on
values would therefore have a **0% hit rate** while paying hashing cost on a
[25,3] float64 tensor per lookup - strictly slower than recomputation.

A shape/dtype/device-keyed cache (ignoring values) would be *incorrect* here,
because the downstream regression consumes the values, not the shape.

## The parts that DO repeat - and why a cache is still the wrong tool

The only exactly-repeating objects are value-independent constants:

1. `torch.ones(25,1)` per `lin_reg_tensor` call - eliminated wholesale by the
   Phase-6 batched path (one `ones(n,25,1)` per layer). A cache would save the
   same allocations while keeping 1.97M Python-loop iterations; batching
   removes the iterations themselves. Batching strictly dominates caching.
2. `inverse_poly_tensor` constants - 3 840 rebuilds x ~1 us on big ~
   milliseconds. Caching (module-level constants) would be *correct* (they are
   value-constant; key = dtype/device) but the measurable benefit is ~0.005 %
   of runtime - below benchmark noise, unjustifiable by rule "benchmark must
   show real benefit". Worth folding into a future GPU DEVFIX of the same
   lines (GPU_READINESS #2), not shipping alone.
3. The 5x5 cartesian index pattern - subsumed by the batched grid assembly
   (LSTSQ_ANALYSIS: `repeat_interleave`/`repeat` is bit-identical), again a
   batching change, not a cache.

## Verdict

**Caching is NOT justified.**

- The data-dependent candidates (grids, design matrices) have a measured 0 %
  exact-repeat rate - a value-keyed cache can never hit.
- The value-independent candidates are either eliminated by batching (ones
  column, index pattern) or too small to measure (poly constants).
- No cache is introduced. The memory-bound/keying design questions are moot.
