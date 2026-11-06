# Outward rounding of D - PRELIMINARY prototype (2026-11-06)

Protocol: `PROTOCOL.md`. Switch: `PASADO_OUTWARD_D=1` / `batched_splits.USE_OUTWARD_D` (batched path
only, **off by default**). Code: `batched_splits.outward_error`; tests: `experiments/test_outward_d.py`;
measurements: `experiments/measure_outward_d.py` -> `effect.json`, `cost.json`, `environment.json`.
Machine: Apple M1 Pro CPU (8 threads), torch 2.12.1, float64, code = working tree of
`checkpoint` (post-fix). **Preliminary**: the margin `4 eps T` is chosen to cover the
measured shortfall (at most 2.36 eps T); it is not a proof that D >= the exact maximum in floating
point for every input.

## Results

| item | result |
|---|---|
| switch off: split bounds against `batched_splits.py` of `1d754c9` (3layer S=1, 8; 5layer S=4; big S=2; 2 images x 2 eps) | bit-identical in all 16 cases |
| coverage on the 4992 archived boxes (CPU recomputation from the eager A, B, C and box) | boxes below the exact maximum: 2560 before, **0 after** |
| how much D grows (median over the 4992 boxes) | 2.4e-10 relative |
| effect on the Lipschitz bound (3 images, eps 0 and 8, n_splits 1 and 64, all 4 networks) | largest relative increase: 3layer 1.2e-14, 4layer 3.2e-14, 5layer 1.2e-13, big 3.2e-14 |
| bound with the switch on >= bound off, and >= sampled autograd maximum | yes, every case checked |

Cost, median of 30 interleaved rounds after 10 warm-up calls (closed-form plane fit A2 inside in
both variants, as in `results/compiled_sigmoid_v3_cuda_2026-10-29/`; the compiled variant is the whole
`precise_sigmoid` under `torch.compile(fullgraph=True)`, Inductor C++ backend):

| configuration | eager off | eager on | compiled off | compiled on | compiled vs eager (on), max rel |
|---|---:|---:|---:|---:|---:|
| 3layer, S = 1 | 1.119 ms | 1.132 ms | 0.252 ms | 0.250 ms | 4.5e-16 |
| 3layer, S = 64 | 5.862 ms | 5.944 ms | 1.561 ms | 1.592 ms | 1.9e-15 |
| big, S = 8 | 10.366 ms | 10.260 ms | 4.242 ms | 4.332 ms | 1.2e-15 |

The switch costs 0-2% (within the run-to-run variation of this laptop, which has 8 performance and
2 efficiency cores and cannot be pinned on macOS). CUDA cost: not measured.

## What it is not

Not a sound floating-point implementation: T itself is computed in round-to-nearest, the factor 4
is empirical, and only the batched path has the switch. A sound version needs directed rounding
(or interval arithmetic) through the whole error evaluation; this prototype only shows that a margin
of this size costs nothing measurable and does not change the bounds at the precision they are
reported.
