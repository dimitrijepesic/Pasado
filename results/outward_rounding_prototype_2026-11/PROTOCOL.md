# Outward rounding of the error term D - preliminary prototype (protocol)

Written 2026-11-06, before the measurement. **Preliminary**: a heuristic margin that covers
the observed shortfall, not a proof of soundness in floating point.

## What changes (behind a switch, off by default)

`PASADO_OUTWARD_D=1` (module flag `batched_splits.USE_OUTWARD_D`, batched path only) replaces
the error term of every (split, neuron) box by

    D_out = nextafter(D + k eps T, +inf),   k = 4 (OUTWARD_D_EPS),

where T = max over the four corners of 0.25|y| + |A x| + |B y| + |C| (the size of the terms D
is computed from; `results/d_shortfall_verification_2026-11/`: the observed shortfall is at
most 2.36 eps T), eps the machine epsilon of the dtype, and `nextafter` rounds the addition
up. With the switch off the code path is unchanged (checked bit for bit).

## Measured

1. Coverage: on the 4992 archived boxes (`gate_d_contract.csv`, eager A, B, C and box),
   D_out >= exact maximum (archived D_true) for every box. Pass/fail.
2. Effect on the bound: the relative increase of the Lipschitz bound, CPU, float64, seed 7,
   3 images, eps indices 0 and 8, n_splits 1 and 64, all four networks.
3. Soundness and equivalence: with the switch off every existing test passes and the bounds are
   bit-identical; with it on, every bound is >= the bound without it and >= the sampled
   autograd maximum (`experiments/test_outward_d.py`).
4. Cost: median time of one batched pass with and without the switch, eager and with the whole
   `precise_sigmoid` compiled by TorchInductor (default mode), 3layer S = 1 and S = 64, big
   S = 8, on the local CPU (Apple M1 Pro; 10 warm-up and 30 timed passes, interleaved); the
   CUDA numbers are not measured here.
