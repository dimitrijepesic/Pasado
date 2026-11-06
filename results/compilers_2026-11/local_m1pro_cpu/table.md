Devices: cpu, cpu:0; commit(s): 1d754c9128; float64 inputs (complex_cubic computes in complex64 by design); steady = median of 30 calls after 10 warm-up calls; post-fix code.

| kernel | size | tool / mode | vs eager (normwise) | NaN pattern | cold first call (s) | warm first call (s) | steady wall ms | steady CUDA-event ms | kernels | graph breaks | recompiles for 2 new sizes | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| bounds | big | torch eager | 0.0e+00 | same | - | - | 69.165 | - | - | - | - |  |
| bounds | big | torch default | 3.6e-15 | same | 4.62 | 1.05 | 68.291 | - | 1 generated + 2 extern | 0 | 1 |  |
| bounds | big | torch max-autotune | 3.6e-15 | same | 3.91 | 1.07 | 71.348 | - | 1 generated + 2 extern | 0 | 1 |  |
| bounds | big | torch reduce-overhead | 3.6e-15 | same | 3.94 | 0.98 | 68.010 | - | 1 generated + 2 extern | 0 | 1 |  |
| bounds | big | jax jit | 5.1e-16 | same | 0.19 | 0.15 | 121.843 | - | 3 HLO ops (3 fusions) | n/a (traces whole function) | 2 |  |
| bounds | small | torch eager | 0.0e+00 | same | - | - | 1.074 | - | - | - | - |  |
| bounds | small | torch default | 5.1e-16 | same | 3.51 | 0.73 | 0.977 | - | 1 generated + 2 extern | 0 | 1 |  |
| bounds | small | torch max-autotune | 5.1e-16 | same | 3.59 | 0.77 | 1.040 | - | 1 generated + 2 extern | 0 | 1 |  |
| bounds | small | torch reduce-overhead | 5.1e-16 | same | 3.77 | 0.73 | 0.968 | - | 1 generated + 2 extern | 0 | 1 |  |
| bounds | small | jax jit | 1.6e-16 | same | 0.09 | 0.02 | 1.353 | - | 3 HLO ops (3 fusions) | n/a (traces whole function) | 2 |  |
| complex_cubic | 65536 | torch eager | 0.0e+00 | same | - | - | 3.339 | - | - | - | - |  |
| complex_cubic | 65536 | torch default | 0.0e+00 | same | 5.27 | 0.73 | 4.164 | - | 4 generated + 0 extern | 0 | 1 | Torchinductor does not support code generation for complex operators. Performance may be worse than eager. |
| complex_cubic | 65536 | torch max-autotune | 0.0e+00 | same | 4.53 | 0.80 | 4.213 | - | 4 generated + 0 extern | 0 | 1 | Torchinductor does not support code generation for complex operators. Performance may be worse than eager. |
| complex_cubic | 65536 | torch reduce-overhead | 0.0e+00 | same | 4.54 | 0.77 | 3.568 | - | 4 generated + 0 extern | 0 | 1 | Torchinductor does not support code generation for complex operators. Performance may be worse than eager. |
| complex_cubic | 65536 | jax jit | 3.6e-06 | same | 0.27 | 0.03 | 1.648 | - | 4 HLO ops (4 fusions) | n/a (traces whole function) | 2 |  |
| fit | 1024 | torch eager | 0.0e+00 | same | - | - | 0.615 | - | - | - | - |  |
| fit | 1024 | torch default | 6.7e-16 | same | 4.97 | 0.79 | 0.182 | - | 1 generated + 0 extern | 0 | 1 |  |
| fit | 1024 | torch max-autotune | 6.7e-16 | same | 4.88 | 0.79 | 0.275 | - | 1 generated + 0 extern | 0 | 1 |  |
| fit | 1024 | torch reduce-overhead | 6.7e-16 | same | 4.75 | 0.80 | 0.189 | - | 1 generated + 0 extern | 0 | 1 |  |
| fit | 1024 | jax jit | 4.4e-16 | same | 0.19 | 0.07 | 0.223 | - | 56 HLO ops (56 fusions) | n/a (traces whole function) | 2 |  |
| fit | 65536 | torch eager | 0.0e+00 | same | - | - | 21.783 | - | - | - | - |  |
| fit | 65536 | torch default | 6.9e-16 | same | 5.22 | 0.82 | 2.072 | - | 1 generated + 0 extern | 0 | 1 |  |
| fit | 65536 | torch max-autotune | 6.9e-16 | same | 5.08 | 0.87 | 1.830 | - | 1 generated + 0 extern | 0 | 1 |  |
| fit | 65536 | torch reduce-overhead | 6.9e-16 | same | 4.86 | 0.78 | 1.563 | - | 1 generated + 0 extern | 0 | 1 |  |
| fit | 65536 | jax jit | 6.9e-16 | same | 0.23 | 0.07 | 6.753 | - | 60 HLO ops (60 fusions) | n/a (traces whole function) | 2 |  |
| lstsq | 1024 | torch eager | 0.0e+00 | same | - | - | 2.554 | - | - | - | - |  |
| lstsq | 1024 | torch default | 0.0e+00 | same | 3.56 | 0.70 | 2.722 | - | 1 generated + 0 extern | 1 | 2 | Dynamic shape operator |
| lstsq | 1024 | torch max-autotune | 0.0e+00 | same | 3.51 | 0.65 | 2.621 | - | 1 generated + 0 extern | 1 | 2 | Dynamic shape operator |
| lstsq | 1024 | torch reduce-overhead | 0.0e+00 | same | 3.65 | 0.66 | 2.524 | - | 1 generated + 0 extern | 1 | 2 | Dynamic shape operator |
| lstsq | 1024 | jax jit | - | - | - | - | - | - | - | - | - | not ported to JAX (see PROTOCOL.md) |
| sigmoid | 3layer_S1 | torch eager | 0.0e+00 | same | - | - | 1.045 | - | - | - | - |  |
| sigmoid | 3layer_S1 | torch default | 0.0e+00 | same | 7.26 | 1.32 | 0.628 | - | 3 generated + 0 extern | 3 | - | Dynamic shape operator |
| sigmoid | 3layer_S1 | torch max-autotune | 0.0e+00 | same | 6.36 | 1.26 | 0.676 | - | 3 generated + 0 extern | 3 | - | Dynamic shape operator |
| sigmoid | 3layer_S1 | torch reduce-overhead | 0.0e+00 | same | 6.36 | 1.30 | 0.773 | - | 3 generated + 0 extern | 3 | - | Dynamic shape operator |
| sigmoid | 3layer_S1 | jax jit | - | - | - | - | - | - | - | - | - | not ported to JAX (see PROTOCOL.md) |
| sigmoid | big_S8 | torch eager | 0.0e+00 | same | - | - | 23.274 | - | - | - | - |  |
| sigmoid | big_S8 | torch default | 0.0e+00 | same | 6.71 | 1.35 | 21.546 | - | 3 generated + 0 extern | 3 | - | Dynamic shape operator |
| sigmoid | big_S8 | torch max-autotune | 0.0e+00 | same | 6.63 | 1.34 | 22.402 | - | 3 generated + 0 extern | 3 | - | Dynamic shape operator |
| sigmoid | big_S8 | torch reduce-overhead | 0.0e+00 | same | 6.54 | 1.28 | 21.156 | - | 3 generated + 0 extern | 3 | - | Dynamic shape operator |
| sigmoid | big_S8 | jax jit | - | - | - | - | - | - | - | - | - | not ported to JAX (see PROTOCOL.md) |
| viete | 65536 | torch eager | 0.0e+00 | same | - | - | 2.036 | - | - | - | - |  |
| viete | 65536 | torch default | 0.0e+00 | same | 3.71 | 0.72 | 0.841 | - | 1 generated + 0 extern | 0 | 1 |  |
| viete | 65536 | torch max-autotune | 0.0e+00 | same | 3.71 | 0.75 | 0.702 | - | 1 generated + 0 extern | 0 | 1 |  |
| viete | 65536 | torch reduce-overhead | 0.0e+00 | same | 3.79 | 0.72 | 0.645 | - | 1 generated + 0 extern | 0 | 1 |  |
| viete | 65536 | jax jit | 1.4e-12 | same | 0.08 | 0.03 | 2.263 | - | 1 HLO ops (1 fusions) | n/a (traces whole function) | 2 |  |
