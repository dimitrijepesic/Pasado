# Automatic choice of the plane fit on CUDA (2026-10-28)

Colab, NVIDIA A100-SXM4-80GB, torch 2.11.0+cu130, CUDA 13.0, code at `1d0c49b`, notebook
`COLAB_AUTO_PLANE_FIT_CUDA.ipynb` of the workspace. `test_auto_plane_fit_cuda.log` is the
raw output of `python experiments/test_auto_plane_fit.py --device cuda`, exit code 0.

On CUDA the test does not widen `AUTO_PLANE_FIT_DEVICES`, so it checks the real setting:

- the thresholds in `batched_splits` are the measured ones, 512 (float64) and 640
  (float32), equal to `../plane_fit_crossover_2026-10-28/thresholds.json`;
- threshold - 1 boxes go to gelsd, threshold and threshold + 1 to the eager closed form,
  in both dtypes (counted calls);
- the override `PASADO_PLANE_FIT_THRESHOLD` works, and combining the automatic choice with
  `PASADO_CLOSED_FORM_FIT` or `PASADO_COMPILED_FIT` raises `ValueError`;
- batched passes give the same bits as the explicit path for their size and stay within
  2.1e-15 of the sequential loop (3layer at 4 and 8 splits, chunk 4 and 8; `big` at 1
  split); chunking changes the boxes per call and the choice with them as expected;
- the CPU keeps gelsd with the default devices, and no TorchInductor graph is compiled.

All 37 checks pass. The automatic choice stays opt-in (`PASADO_AUTO_PLANE_FIT=1`): its
thresholds were measured on the A100 only, so it is not made the default for every CUDA
device.
