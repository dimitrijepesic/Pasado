# First CUDA run of the gelsd replacement comparison: stopped by the self-test (2026-10-26)

Colab, A100, code at `6e0acc6` (rule of `385397e`, candidates of `077ab86`).
`box_level.log` is the console output of `--level box --device cuda`.

- G0 contract: every candidate passes on CUDA, P included. On the CPU, P fails C2
  (a shuffled batch changes some results by 1 ulp with LAPACK gesdd); cuSOLVER
  gesvdj gives bit-identical results for a shuffled batch.
- Self-test: 9 of 10 planted bugs caught; S4 (centring on fl(mean) instead of a data
  point) survived, so the run stopped before the measurement, as the rule requires.

Why S4 survived, and what changed. On the CPU (runs v1 and v2) S4 had been counted as
caught through G1 on near_degenerate, float64, but its parent A1 fails that gate on the
same reduced boxes, so that catch proved nothing. The real effect of S4 is in the
accuracy: median distance to the exact solution 2.5e-8 against 1.0e-16 for A1 on
near_degenerate, float64, which R3 hides because it takes the worst family over both
dtypes. Commit `67f4654` counts a catch only for a check or gate the parent passes, adds
a per-family accuracy criterion (at least 100 times the parent's median error) and a
deterministic self-test family `offset_narrow`. On the CPU all 10 planted bugs are then
caught, S4 on three families in float64. The candidates, the measured families, the
gates and the ranking are unchanged.
