# Porting the Pasado precise-zonotope analysis to AWS Trainium: scoping note

Status: discussion draft, 2026-11-06. Nothing has been installed or implemented. The facts below come from the AWS Neuron documentation as of Neuron 2.33.0 (released 2026-10-19). Anything this note could not confirm from a source is marked "not confirmed".

## 1. What Trainium / Neuron offers today

- **TorchNeuron** is a native PyTorch backend that registers the device `neuron` through PrivateUse1. It was announced at re:Invent in December 2025 and first shipped in Neuron 2.27.0. In 2.33 it is in **public beta**, runs on Trn2 and Trn3 only (not Trn1), and targets PyTorch 2.13/2.14. The older XLA-based `torch-neuronx` stopped at PyTorch 2.9 and is now legacy. [S1, S2, S3]
- **Eager mode** runs on the device. Each op is lowered to StableHLO and compiled by `neuronx-cc` to a NEFF, which is cached per shape. An op with no Neuron lowering falls back to the CPU, and that fallback is silent unless you call `torch_neuronx.get_fallback_ops()`. [S1, S4]
- **`torch.compile(backend="neuron")`** is a TorchDynamo backend, not Inductor. It supports only `dynamic=False`: "Dynamic shapes and symbolic ints are not supported" and "each distinct shape triggers a separate compile". Each graph break produces a separate NEFF. Data-dependent shapes (`nonzero`, boolean masking) and host syncs (`.item()`, `.cpu()`) cause breaks. [S2, S4]
- **Dtypes**: "float64 runs as float32 and produces a warning" [S2], or in the words of the best-practices page, "float64 is cast to float32 internally" [S4]. The Trn2 engines list FP8, BF16, FP16, TF32 and FP32, and no FP64 [S6]. NKI's dtype table also stops at float32 and has no complex types [S7]. The compiler's default is to cast FP32 to BF16, which `--auto-cast` controls [S5, S8]. Whether that default applies inside TorchNeuron eager mode: not confirmed. Complex types are listed as supported in general [S2], but `torch.complex` and `conj_physical` on complex64 fall back to the CPU [S3].
- **torch.linalg**: `lstsq`, `svd`, `qr`, `solve`, `inv`, `pinv` and `eigh` are all `CPU_FALLBACK`. Only `matmul`, `multi_dot`, `solve_triangular` and the norms are native. `acos`, `cos`, `where`, `sign`, `cumsum`, `mm` and `bmm` are native. [S3]

## 2. How Pasado's requirements fit

| Pasado requirement | Neuron support | Consequence |
|---|---|---|
| float64 everywhere (`torch.set_default_dtype(torch.float64)`, HyperDuals.py) | **No FP64 hardware.** float64 is cast to float32 with only a warning. [S2, S4, S6] | **Blocker.** The results would be float32 results labelled float64. Pasado's float32 tolerance is 1e-5 against 1e-9 for float64, and the measured float32 CPU/CUDA gap is 8.3e-7 against 1.8e-15 (results/gpu_hybrid_verification_2026-10-08). Without directed rounding, the soundness of the error bounds in float32 is also unestablished. |
| Plane fit, `lstsq(driver='gelsd')` | `linalg.lstsq`: CPU_FALLBACK [S3] | Runs on the host anyway, plus a device-to-host copy each layer. No gain. |
| Plane fit, closed form A2 (planar_fit.py) | Elementwise ops and small reductions: native in float32 [S3] | Feasible, but only in float32, and A2 was validated for bit-equality in float64. |
| Cubic, real Viete (`PASADO_REAL_CUBIC=1`) | `acos`, `cos`, `where` native [S3] | Feasible in float32. Note that the repo default is still the complex64 path (`PASADO_REAL_CUBIC` defaults to `"0"`). |
| Cubic, legacy complex64 | Partial: `torch.complex` falls back to CPU [S3] | Avoid. It is already known to be inaccurate (negative control). |
| Generator matmuls (`AffineZonotope`, `z.rg @ W`) | `matmul`/`mm`/`bmm` native, 20 FP32 dense TFLOPS per NeuronCore-v3 [S3, S6] | The good fit, but see the next row. |
| Noise-symbol count grows each layer (`torch.cat` of new generators) | Static shapes only, one compile per shape [S2, S4] | One NEFF per (layer, symbol count, number of splits). Padding to fixed buckets would be needed, and the compile time could exceed the run time on small nets. |
| `np.random` host noise on A, one vector per layer | Host op, then a copy to the device. `torch.random` seeding is native [S3] | A small host-to-device transfer per layer. It breaks a graph under `compile` only if it is drawn inside the compiled region. The seeded protocol can stay on the host. |

## 3. Minimal feasibility experiment (only if the mentor wants to proceed)

Port one isolated kernel in **float32 only**: `affine` + `precise_sigmoid` from batched_splits.py for a single layer, with A2 and the real cubic, at fixed padded shapes. Run it in TorchNeuron eager mode on a trn2 instance, then under `compile(fullgraph=True, dynamic=False)` with auto-cast disabled. Compare it against the same float32 kernel on CPU, using the existing 1e-5 protocol.

- **It could tell us** whether the kernel lowers without CPU fallbacks (`get_fallback_ops()`), how far Neuron FP32 is from CPU FP32, the compile time per shape against the run time, and the matmul throughput.
- **It could not tell us** anything about float64 results, soundness, or end-to-end speed. It also could not tell us whether the paper's numbers would survive, because by construction it cannot reproduce the float64 reference.

## 4. Open questions for the mentor

1. Is a float32 analysis acceptable at all for the paper's claims? If not, Trainium is ruled out today, and the rest is moot.
2. If it is, do the bounds need sound rounding? Trn2 offers RNE or stochastic rounding [S6]; directed rounding: not confirmed.
3. Is the goal throughput (many images or splits) or latency? Static-shape bucketing only pays off for large batched runs.
4. Is access to trn2 available (account, budget)? TorchNeuron does not support Trn1.
5. Would a beta backend be acceptable as the platform for reported results?

## Sources

- [S1] TorchNeuron overview: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/torch/about-pytorch-neuron.rst
- [S2] Neuron 2.33.0 TorchNeuron release notes: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/release-notes/components/torchneuron.rst
- [S3] TorchNeuron Torch API compatibility table: https://github.com/aws-neuron/torch-neuronx/blob/main/docs/torch_api_compatibility.md
- [S4] PyTorch Neuron best practices: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/torch/reference/best-practices-pytorch-neuron.rst
- [S5] Neuron data types: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/about-neuron/arch/neuron-features/data-types.rst
- [S6] Trainium2 architecture (NKI guide, v2.32): https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/nki/guides/architecture/trainium2_arch.rst
- [S7] NKI data types (v2.32): https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/nki/api/nki.api.shared.rst
- [S8] neuronx-cc FAQ (v2.32): https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/_sources/compiler/neuronx-cc/faq.rst
- Migration from torch-neuronx (XLA) to TorchNeuron: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/torch/index.rst (via search summary; not opened directly)
