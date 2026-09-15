# REPRODUCE — exact commands (PowerShell, from the Pasado repo root)

Everything below was run on: Windows 11 (10.0.26200), Python 3.13.3,
torch 2.12.1+cpu, numpy 2.5.1, scikit-learn 1.9.0, torchvision 0.27.1+cpu,
Intel Family 6 Model 154, 8 torch threads, CUDA unavailable.

## 0. Environment

```powershell
cd C:\everything\prakse\UIUC\Pasado
.venv\Scripts\python.exe -c "import torch, numpy, sklearn; print(torch.__version__, numpy.__version__, sklearn.__version__, torch.cuda.is_available())"
```

MNIST data is expected under `Section_5_4\MNIST_Data` (the harnesses download
it on first use).

## 1. Implementation selectors

Both default to the **currently intended** behavior; neither changes the
mathematics — only which implementation of the same computation runs.

| variable | default | effect |
|---|---|---|
| `PASADO_VECTORIZED_PRECISE` | `1` | `1` = vectorized corner + nonlinear-boundary checks; `0` = original per-neuron Python-loop checks |
| `PASADO_VEC_BOUNDARY` | `1` | `0` = keep vectorized corners but restore the original boundary check ("corners only" variant) |
| `PASADO_BATCHED_LSTSQ` | `0` | `1` = one batched `lstsq` per layer instead of one per neuron |

## 2. Unit tests

```powershell
cd forward_mode_tensorized_src
..\.venv\Scripts\python.exe test_check_corners_tensor.py
..\.venv\Scripts\python.exe test_check_nonlinear_boundary_tensor.py
..\.venv\Scripts\python.exe test_compute_max_error_tensor.py
..\.venv\Scripts\python.exe test_selector_paths.py
cd ..
.venv\Scripts\python.exe experiments\test_batched_lin_reg.py
```

## 3. Reduced end-to-end correctness (old vs new), no full benchmark needed

```powershell
# old per-neuron path vs vectorized checks       -> logs\correctness_results.csv
.venv\Scripts\python.exe experiments\correctness_e2e.py

# old per-neuron path vs vectorized + batched    -> logs\correctness_results_batched.csv
.venv\Scripts\python.exe experiments\correctness_e2e.py --batched
```

## 4. Smoke test (single network, no saving)

```powershell
cd Section_5_4
..\.venv\Scripts\python.exe .\get_lipschitz.py --network 3layer --no-save
cd ..
```

`--no-save` skips only the final `torch.save` calls; the computation is
unchanged.

## 5. Wall-clock benchmarks (the headline numbers)

```powershell
# 3layer, all four variants, 3 repetitions each (median reported)
.venv\Scripts\python.exe experiments\benchmark_lipschitz.py `
    --network 3layer --variant original vec_corners vec_both vec_both_batched --runs 3

# 4layer + 5layer, baseline vs final optimized
.venv\Scripts\python.exe experiments\benchmark_lipschitz.py `
    --network 4layer 5layer --variant original vec_both_batched --runs 3

# big, single run per variant (expensive - ~30 min each)
.venv\Scripts\python.exe experiments\benchmark_lipschitz.py `
    --network big --variant original vec_both_batched --runs 1
```

Results append to `logs\benchmark_results.csv` with git commit, diff hash and
full environment metadata.

**Important:** run these on an otherwise idle machine. Two runs in this sprint
were corrupted by background system activity (a `vec_both` big run and one
3layer cProfile run took 1.5-20x longer than physically explicable); both are
documented and excluded from the reported figures.

## 6. cProfile + pstats extraction

```powershell
cd Section_5_4
# baseline (original per-neuron checks)
$env:PASADO_VECTORIZED_PRECISE="0"
..\.venv\Scripts\python.exe -m cProfile -o ..\profiles\lipschitz_3layer_original.prof .\get_lipschitz.py --network 3layer --no-save
Remove-Item Env:\PASADO_VECTORIZED_PRECISE

# batched lstsq
$env:PASADO_BATCHED_LSTSQ="1"
..\.venv\Scripts\python.exe -m cProfile -o ..\profiles\lipschitz_3layer_batched_lstsq.prof .\get_lipschitz.py --network 3layer --no-save
Remove-Item Env:\PASADO_BATCHED_LSTSQ
cd ..

.venv\Scripts\python.exe experiments\extract_pstats.py `
    profiles\lipschitz_3layer_batched_lstsq.prof `
    logs\lipschitz_3layer_batched_lstsq_pstats.txt
```

`extract_pstats.py` writes top-150 by cumulative time, top-150 by self time,
and callers/callees for the functions of interest.

## 7. Microbenchmarks

```powershell
# per-neuron loop vs batched regression (n = 1, 10, 100, 1024)
.venv\Scripts\python.exe experiments\benchmark_lstsq.py

# shape / driver / batching probes behind LSTSQ_ANALYSIS.md
.venv\Scripts\python.exe experiments\probe_lstsq.py
```

## 8. torch.profiler (reduced workload)

```powershell
.venv\Scripts\python.exe experiments\torch_profiler_compare.py
```

Writes `logs\torch_profiler_{old,vec,batched}.txt` and Chrome traces under
`profiles\`.

## 9. torch.compile experiment

```powershell
.venv\Scripts\python.exe experiments\compile_experiment.py
```

**Prerequisite not met on this machine:** TorchInductor's CPU backend needs
MSVC `cl.exe`. Without the Visual Studio C++ Build Tools on PATH every
`torch.compile` call fails with `Compiler: cl is not found`. Dynamo *capture*
still works and is what the analysis reports.

## 10. GPU / Colab

```python
# in a Colab notebook, after uploading or cloning the repo
!pip install torch torchvision scikit-learn
!python experiments/colab_gpu_profile.py
```

Prints CUDA availability, GPU model, fp32-vs-fp64 matmul throughput, runs the
reduced CPU test, and compares CPU-vs-GPU numerics for the device-ready
kernels. If no GPU is present it says so and skips the GPU section — it never
fabricates GPU numbers. Known compatibility fix for a fresh environment:
`conv.py` uses `np.product`, removed in NumPy >= 2.0 (`np.prod` is the
replacement); this affects the CNN path only, not the FC benchmark.
