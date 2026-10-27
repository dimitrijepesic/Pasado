# Comparison of GPU replacements for gelsd: runs v1 and v2 (2026-10-25)

Both runs apply the same pre-registered rule (`experiments/compare_lstsq_solvers.py`,
committed in `385397e`) with the same seeds and box families, on the CPU (Apple M5 Pro,
torch 2.12.1, float64 and float32).

Update 2026-10-27: the A100 timing has run. With it (`timing_cuda.csv`, copied from
`../lstsq_comparison_cuda_2026-10-27/`) the rule picks **A2** on this run
(`decide_with_a100_timing.log`): A2 and H tie through R4 and R5 prefers A2. A2 is the
choice; the full CUDA run and its sensitivity result are described in
`../lstsq_comparison_cuda_2026-10-27/README.md`. The sections below describe the state
before that timing.

| Run | Folder | Candidate code | Source hash |
|---|---|---|---|
| v1 | `../lstsq_comparison_2026-10-25/` | `385397e` (run at `4509e1f`) | `fc5aad1ebcf2...` |
| v2 | this folder | `077ab86` | `002b7bf6cf47...` |

v1 stands as recorded. It found a defect: the closed forms overflowed on boxes with
|y| = 1e300 (float64) or 1e30 (float32). `077ab86` fixes that defect and nothing else,
the rule is unchanged, and v2 repeats the run. The fix was motivated by the bug, not by
the ranking; both outcomes are reported here.

## What the fix changes

The closed forms now divide the centred x, y and target values by a power of two near
their maximum before forming sums of squares and products, and multiply the power back
afterwards; `gram_eigenvalues` takes standard deviations instead of variances. Dividing
by a power of two is exact as long as no scaled intermediate becomes subnormal, so the
result has the same bits as before wherever the old sums were finite.

Checked on boxes not used by the comparison (seed 777, the ten synthetic families,
2000 boxes each, both dtypes, A1, A2, A3 and H): of 160 000 outputs, 159 682 are
bit-identical to v1, 318 that were non-finite in v1 are finite, and none differs.
This is strong evidence, not a proof for every possible input.

## Outcome

| | v1 | v2 |
|---|---|---|
| Self-test (planted bugs) | 10 of 10 caught | 10 of 10 caught |
| G0 | P fails C2 | P fails C2 |
| G1 | all pass | all pass |
| G2 | A1, A2, H fail (18 non-finite boxes per dtype in `large_magnitude`) | all pass |
| Survivors | G, A3 | G, A1, A2, A3, H |
| G3, R1 (180 network cases) | all sound; tie, max difference to G 1.97e-13 | all sound; tie, max difference to G 1.97e-13 |
| R2a (W / W_best, p99) | G 1.00, A3 1.25 | G, A2, H 1.00; A1, A3 1.25 |
| R2b | G -1, A3 -2 | G, A2, H -1 (A1 -1, A3 -2) |
| R3 (order of magnitude of the worst median error) | G 7, A3 -4 | G 7; A2, H -7 |
| Result (`decide.log`) | **Winner: G**, by R2a | **A2 and H tied after R3; undecided at R4** |

- P fails G0 C2 in both runs: on the CPU, `torch.linalg.pinv` (LAPACK gesdd) changes
  11 of 200 results by 1 ulp (4.5e-16) when the batch is shuffled, and the rule asks for
  bit-identical results. P stays provisional until the box level runs on CUDA.
- R2a separates candidates only on `large_magnitude`: there gelsd drops the x axis
  (its spread is negligible next to |y| = 1e300), and so do A2 and H through the
  analytic cutoff. A1 and A3 keep the exact least-squares slope and their radius W is
  25% larger. On the other hard families every candidate is at 1.00.
- R3 removes G: its worst family median distance to the exact solution is 3.8e7
  (`cutoff_band_y`, float64, where gelsd returns coefficients up to 5.2e9), against
  4.1e-7 for A2 and H (`cutoff_band_y`, float32).
- R4 needs the A100 timing at n = 65536 in float64. A2 is elementwise; H adds a host
  sync and a CPU gelsd for the boxes it flags. If they land within 20% of each other,
  R5 prefers A2 (no host sync).

## Real boxes and the rank rule

- float64: 119 904 boxes recorded from the batched forward pass; gelsd truncates none
  beyond its exact rank and none lies within a factor 2 of its cutoff
  (`real_rank_stats.csv`). In float64 the rank rules of A1, A2 and gelsd coincide on
  real data.
- float32: the 2000 sampled real boxes, cast to float32, are truncated by gelsd in 129
  cases. A2's analytic cutoff agrees on all but one box (`5layer`, image 1, eps index 15,
  layer 2), which lies at s3 / s1 = 0.9975 rcond; A2 truncates there and gelsd does not,
  so their coefficients differ by O(1), as they must at a rank switch. Both planes are
  sound. In float32 (MPS) the rank rule therefore matters on real data, and A1 (exact
  endpoints only) differs from gelsd on those 129 boxes.

## Next

1. Colab, A100: `--level timing --device cuda` for R4, and `--level box --device cuda`
   so that P is judged on cuSOLVER gesvdj, with the GPU model and versions recorded.
2. Apply the rule again with the CUDA timing file in this folder.

## Files

Same layout as v1 (see its README). To apply the rule again:

```bash
gzip -dk results/lstsq_comparison_2026-10-25_v2_scaled/box_per_box.csv.gz
python experiments/compare_lstsq_solvers.py --level decide --out results/lstsq_comparison_2026-10-25_v2_scaled
```

The source hash in `box_meta.json` refers to `077ab86`. A later commit that only edits
comments changes the hash, so `--level network` refuses these box results after it;
`--level decide` does not check the hash.
