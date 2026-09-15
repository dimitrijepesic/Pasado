# LSTSQ_ANALYSIS — per-neuron regression in the precise transformer (Phase 5)

Evidence sources: big pstats (`logs/lipschitz_big_nosave_both_vectorized_pstats.txt`),
optimized 3layer cProfile (`logs/cprofile_3layer_both_vectorized.txt`),
and `experiments/probe_lstsq.py` (run on torch 2.12.1+cpu, Windows, float64).
No source change was made in this phase.

## The code under analysis

`precise_transformer.py`:

```python
def lin_reg_tensor(x, zs):                     # one call per neuron
    xplusone = torch.cat((torch.ones(x.size(0), 1), x), 1)
    R = torch.linalg.lstsq(xplusone, zs, driver='gelsd').solution
    return R[0:xplusone.size(1)]
```

Called from `sigmoid_prime_product_tensor` via a Python list comprehension:

```python
xys = get_linspace(lx, ux, ly, uy)             # list of n [25,2] grids
zs = [sigmoid_prime_times_y(xy) for xy in xys]
lineqs = [lin_reg_tensor(xys[i], zs[i]) for i in range(len(zs))]
```

`get_linspace` itself is a per-neuron Python loop: two `torch.linspace(...,steps=5)`
calls + one `torch.cartesian_prod` per neuron.

## Answers to the ten questions

**1. How many calls?**
Exactly `(#sigmoid layers) x width x #images x #epsilons`:

| network | formula | calls (whole benchmark) |
|---|---|---|
| 3layer | 2 x 100 x 30 x 16 | **96,000** (matches pstats: 96000) |
| big | 4 x 1024 x 30 x 16 | **1,966,080** (matches pstats: 1966080) |

**2. Typical shapes of x and zs?**
Instrumented reduced run (1 image, 2 epsilons, both layers, 400 calls):
**every call is `x=[25,2]`, `zs=[25]`, float64** (grid = 5x5 cartesian product,
STEPS=5 hardcoded). The solved system is `[25,3] \ [25]` -> 3 coefficients.

**3. Do all calls share the same shape?** Yes — 400/400 in the instrumented
run, and structurally always: STEPS is a constant default, never overridden.
This makes the calls perfectly stackable.

**4. Does the design matrix xplusone repeat?**
No. SHA-1 hashing of all 400 xplusone matrices in the reduced run: **400
distinct, 0 exact duplicates**. The bounds (lx,ux,ly,uy) differ per neuron /
epsilon / image, so exact-value repetition is essentially impossible. What DOES
repeat is *structure*: the constant intercept column `ones(25,1)` (freshly
allocated 1.97M times on big) and the canonical 5x5 grid pattern that every
matrix is an affine image of.

**5. Can the calls be merged into one batched call?**
Yes. Per layer, the n per-neuron systems stack to `A:[n,25,3]`, `B:[n,25,1]`
(n=100 or 1024; float64; identical m). Memory is trivial (1024*25*3*8 = 600 KB).
Probe result: batched `gelsd` solution vs per-item loop over the same
matrices — **max abs difference 0.000e+00, bit-identical**. Stacking the
*existing* per-neuron grids preserves their exact values, so a batched solve of
stacked inputs is provably the same computation.

**6. Does torch.linalg.lstsq support batched A and B here (Windows CPU,
torch 2.12.1+cpu)?** Yes — `A:[64,25,3], B:[64,25,1]` solved fine both with
`driver='gelsd'` and with the default driver.

**7. Does driver='gelsd' work for batched input?** Yes, and bit-identically to
the per-matrix loop (0.000e+00 max diff in the probe). ATen iterates the batch
at C++ level over the same LAPACK `?gelsd` routine, so per-matrix results are
unchanged.

**8. Would omitting the driver change numerical behavior?**
Yes, subtly. On CPU the default driver is `gelsy` (QR with pivoting), not
`gelsd` (SVD). Probe: well-conditioned system differs by 3.3e-16 (ULP-level but
nonzero); on rank-deficient systems `gelsy` does not guarantee the minimum-norm
solution that `gelsd` returns (our degenerate-box probe happened to agree to
1e-16, but that is not a guarantee). **Keep `driver='gelsd'`** — batching does
not require changing it.

**9. Is anything safely cacheable?**
Only trivial constants: the `ones(25,1)` intercept column, the cartesian-product
index pattern, the canonical `[0,.25,.5,.75,1]` grid. No full design matrix ever
repeats (Q4), so exact-input caching has zero hit rate. The constants are
subsumed by batching (one `ones(n*25,1)`-equivalent per layer instead of n) —
a standalone cache is not worth building. **Caching rejected.**

**10. Python loop dispatch or the numerical solver?**
Overwhelmingly dispatch/overhead. Measured per-call cost of the `lstsq` builtin:
37 us (3layer cProfile) / 61 us (big cProfile). The actual math — SVD of a
25x3 float64 matrix — is a few thousand FLOPs, well under 1 us. Even allowing
for cProfile inflation, >90% of the per-call time is Python frame + argument
parsing + LAPACK workspace setup + tensor allocation, repeated 1.97M times on
big. The same applies to the per-call `torch.cat` (45 us/call for a 25x3
concat) and `torch.linspace`/`cartesian_prod`. This is the textbook
many-tiny-ops pattern; the solver itself is nearly free.

## Grid-construction detail (informs Phase 6 scope)

* `torch.cartesian_prod(xs, ys)` **is bit-identical** to
  `stack((xs.repeat_interleave(5), ys.repeat(5)), 1)` (0/2000 probe mismatches)
  -> the per-neuron `cartesian_prod` calls can be batched away exactly.
* `torch.linspace(l,u,5)` is **NOT always bit-identical** to the affine forms
  `l+(u-l)/4*arange(5)` / `l+(u-l)*[0,.25,.5,.75,1]` (280/2000 trials differ,
  ~1 ULP). A fully batched grid construction is therefore float64-equivalent
  but not bit-exact.

## Cost at stake (big network, cProfile attribution)

| piece | self s | eliminated by |
|---|---|---|
| `linalg_lstsq` (1.97M calls) | 119.6 | batched solve (bit-exact) |
| `torch.cat` xplusone (1.99M) | 89.4 | batched construction (bit-exact) |
| `lin_reg_tensor` self + `ones` + `.item` | ~36 | batched construction (bit-exact) |
| `cartesian_prod` (1.97M) | 39.2 | batched assembly (bit-exact) |
| `sigmoid_prime_times_y` self (1.97M) | 33.6 | batched eval (bit-exact, elementwise) |
| `torch.linspace` (3.93M) + `get_linspace` self | 54.3 | batched linspace (**1 ULP, not bit-exact**) |

~280 s of ~300 s is removable with **bit-identical** results; the last ~54 s
(linspace) costs 1 ULP of grid position if batched.

## Recommendation

1. **Batched lstsq: YES — implement in Phase 6.** Stack the existing per-neuron
   grids -> batched zs evaluation -> batched xplusone -> one
   `torch.linalg.lstsq(A, B, driver='gelsd')` per layer. Every step of that
   chain is proven bit-identical on this platform. Keep `driver='gelsd'`.
   Gate behind `PASADO_BATCHED_LSTSQ` (already stubbed, default off until
   validated end-to-end).
2. **Batch `cartesian_prod` away too** (repeat/interleave form): bit-exact,
   included in step 1's construction.
3. **Batched linspace: optional second step.** Not bit-exact (1 ULP). Only do
   it if the bit-exact variant leaves `linspace` as a measurable cost, and then
   document the ULP-level difference explicitly.
4. **Caching: NO** — zero exact-repetition (Q4/Q9).
5. **Closed-form / normal equations / pinv: NO** — unnecessary (batched gelsd
   already native and bit-exact) and numerically riskier (squares the condition
   number).
