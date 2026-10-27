# planar_fit (A2) in the batched path on CUDA (2026-10-27)

Colab, NVIDIA A100-SXM4-80GB, torch 2.11.0+cu130, CUDA 13.0, code at `c2dcc4e`
(`planar_fit.py` behind `batched_splits.USE_CLOSED_FORM_FIT`), notebook
`COLAB_PLANAR_FIT_VERIFICATION.ipynb` of the workspace. float64 unless noted.

## Correctness

- `test_planar_fit_cuda.log`: same bits as candidate A2 on all 12 box families and on
  2000 recorded boxes, checks C1-C7, close to gelsd, and 1.9e-15 (float64) / 1.5e-6
  (float32) from the CPU result. All pass.
- `test_batched_splits_{gelsd,closed_form}_cuda.log`: every per-split bound against the
  sequential loop (largest relative difference 5.5e-13 with planar_fit), chunking, split
  independence, autograd soundness; the solver check reports 0 closed-form calls with
  gelsd and 396 with planar_fit. All pass.
- `speed_same_run.csv`: all 96 bounds agree with the sequential loop to 2.1e-13.

## Speed in one run (`speed_same_run_summary.csv`, median of 3, mean over 3 images)

| network | n_splits | sequential (ms) | batched + gelsd | batched + planar_fit | planar_fit vs sequential | planar_fit vs batched gelsd |
|---|---:|---:|---:|---:|---:|---:|
| big, eps 0 | 1 | 61.7 | 52.6 | 31.6 | 1.95x | 1.66x |
| big, eps 0 | 32 | 2012.0 | 1250.1 | 155.0 | 12.98x | 8.07x |
| big, eps 0 | 64 | 4023.9 | 2485.2 | 287.3 | 14.01x | 8.65x |
| big, eps 8 | 32 | 1934.8 | 1187.1 | 153.2 | 12.63x | 7.75x |
| big, eps 8 | 64 | 3872.7 | 2337.7 | 286.8 | 13.50x | 8.15x |
| 3layer, eps 0 | 1 | 12.7 | 10.9 | 16.1 | 0.79x | 0.68x |
| 3layer, eps 0 | 8 | 100.7 | 22.0 | 16.1 | 6.25x | 1.37x |
| 3layer, eps 0 | 64 | 808.2 | 109.0 | 16.3 | 49.70x | 6.70x |

4layer and 5layer behave like 3layer (0.68x against batched gelsd at 1 split, 1.36x to
1.42x at 8 splits, 6.5x to 7.2x at 64 splits).

Share of `planar_fit.fit_planes` in the batched pass on `big` (`planar_fit_share_big.csv`,
synchronized around every call): 46.0% at 1 split, 24.8% at 8, 10.4% at 32, 5.4% at 64.

## Criteria of the task brief

| criterion | result |
|---|---|
| `big` at least 10x faster than the sequential loop at 32 and 64 splits | pass: 12.6x to 14.0x |
| solver share below 20% at 32 and 64 splits | pass: 10.4% and 5.4% |
| small networks not slower than batched gelsd | **fail at 1 split**: 0.68x |

`fit_planes` costs about 3.8 ms per call whatever the number of boxes (15.4 ms for four
calls at 1 split, 16.0 ms at 64; the solver timing of 2026-10-27 gives 3.96 ms at
n = 1024 and 3.72 ms at n = 65536). With 100 boxes per call (small networks, 1 split)
gelsd on the CPU is cheaper; from 800 boxes per call on, planar_fit wins. The crossover
depends on the number of boxes per call (n_splits times layer width), not on the network
as such, and has not been measured precisely. The constant cost is presumably the launch
latency of many small elementwise kernels; that is not measured here.

Decision: the switch stays off by default. Next: TorchInductor on `fit_planes`, the
crossover in n for an automatic dispatch, and A3 end to end.
