# Comparison of GPU replacements for gelsd, run v1 (2026-10-25)

First run of the pre-registered comparison in `experiments/compare_lstsq_solvers.py`
(rule committed in `385397e`; code at `4509e1f`, source hash `fc5aad1ebcf2...` in
`box_meta.json`). CPU only (Apple M5 Pro, torch 2.12.1, float64 and float32); the A100
timing (R4) and the CUDA box level for P have not run.

This result stands as recorded. A follow-up run with an overflow fix in the closed
forms and the same rule is in `../lstsq_comparison_2026-10-25_v2_scaled/`.

## Outcome under the rule

`decide.log`: **Winner: G** (keep gelsd), decided by R2a.

| Candidate | Result |
|---|---|
| G | passes G0-G3 |
| A3 | passes G0-G3; R1 ties with G (max relative difference 1.97e-13 over 180 network cases); R2a 1.25 against G's 1.00 |
| A1, A2, H | out at G2: 18 non-finite boxes per dtype in `large_magnitude`, G has none |
| P | out at G0 C2: a shuffled batch changes 11 of 200 results by 1 ulp (4.5e-16) on the CPU (gesdd); provisional until CUDA |

Self-test: all 10 planted bugs caught (`g0_selftest.csv`).

## What decided it

- All 18 non-finite boxes of A1, A2 and H have |y| = 1e300 (float64) or 1e30 (float32)
  with a y width of 1e-6 |y|. The closed form multiplies centred y values by centred
  targets (about 1e294 times 1e293), which overflows; gelsd survives because LAPACK
  rescales the whole matrix. A3 does not form that product.
- A3's R2a of 1.25 comes from the same family only: on those boxes gelsd drops the x
  axis (its spread is negligible next to |y| = 1e300) and its radius W is 25% smaller.
  On every other hard family A3 equals or beats G.
- Such pre-activation magnitudes do not occur in Pasado. On the realistic families the
  closed forms are far more accurate than gelsd, for example `cutoff_band_y`, float64:
  median distance to the exact solution 3.8e7 for gelsd (coefficients up to 5.2e9)
  against 5.5e-16 for A1 (coefficients up to 3.1).

## Real boxes

119 904 boxes recorded from the batched forward pass in float64 (`real_rank_stats.csv`):
gelsd truncates none of them beyond their exact rank and none lies within a factor 2 of
its cutoff, so in float64 the rank rules of A1 and A2 coincide on real data. This does
not hold in float32: cast to float32, the 2000 sampled real boxes are truncated by gelsd
in 129 cases (`box_summary.csv`, family `real`). See the joint report of v1 and v2 in
`../lstsq_comparison_2026-10-25_v2_scaled/README.md`.

## Files

| File | Content |
|---|---|
| `box_level.log`, `network_level.log`, `decide.log` | console output of the three levels |
| `box_meta.json`, `network_meta.json` | environment, code commit and source hash, configuration |
| `g0_contract.csv` | checks C1-C7 per candidate, family and dtype |
| `g0_selftest*.csv` | planted bugs: verdicts, their contract checks and reduced box level |
| `real_rank_stats.csv` | per network, image, epsilon and layer: degenerate axes, gelsd truncations |
| `box_summary.csv` | per dtype, family and candidate: gates and ranking inputs |
| `box_per_box.csv.gz` | every box and candidate (needed by R2a); SHA-256 of the raw file in `box_per_box.csv.sha256` |
| `network_summary.csv` | per network case and candidate: bound, excess over G, soundness |

To apply the rule again:

```bash
gzip -dk results/lstsq_comparison_2026-10-25/box_per_box.csv.gz
python experiments/compare_lstsq_solvers.py --level decide --out results/lstsq_comparison_2026-10-25
```
