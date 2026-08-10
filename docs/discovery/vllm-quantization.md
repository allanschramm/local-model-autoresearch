# vLLM Quantization — Supported Formats & Hardware Matrix

vLLM's quantization surface: which quant methods it runs, on which hardware. Reference for engine selection (see [inference-engines-landscape.md](./inference-engines-landscape.md)); complements the NVFP4 format doc ([nvfp4-quantization.md](./nvfp4-quantization.md)). Per-format mechanics (kernels, group sizes, min SMs, NVFP4 status from source): [vllm-quant-deep-dive.md](./vllm-quant-deep-dive.md).

## Supported quantization formats

- **AutoAWQ** — weight-only 4-bit
- **BitsAndBytes** — 4/8-bit (QLoRA-style, LLM.int8)
- **GPTQModel** — GPTQ 4-bit weight-only
- **Intel Neural Compressor**
- **LLM Compressor** (recommended entry point; supports FP8, INT8, INT4):
  - FP8 W8A8
  - INT4 W4A16
  - INT8 W4A8
  - INT8 W8A8
- **NVIDIA Model Optimizer** — NVFP4/FP8 export path (TensorRT/Blackwell formats)
- **Online Quantization** — runtime quantization
- **AMD Quark**
- **Quantized KV Cache** (FP8/E5M2/E4M3 KV cache)
- **TorchAO**
- **FP8 ViT Encoder Attention** (multimodal encoders)
- **GGUF** — native GGUF quant loading (GPU only, see matrix)
- **Marlin kernels** — high-perf GEMM for GPTQ/AWQ/FP8/FP4

Tip from docs: start with **LLM Compressor** (vLLM's optimizer library) for FP8/INT8/INT4 workflows.

## Hardware compatibility matrix

SM mapping: Volta = SM 7.0, Turing = SM 7.5, Ampere = SM 8.0/8.6, Ada = SM 8.9, Hopper = SM 9.0.

| Implementation | Volta | Turing | Ampere | Ada | Hopper | AMD GPU | Intel GPU | x86 CPU | Arm CPU |
|---|---|---|---|---|---|---|---|---|---|
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Marlin (GPTQ/AWQ/FP8/FP4) | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| llm-compressor INT8 (W8A8) | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| llm-compressor INT8 (W4A8) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| llm-compressor FP8 (W8A8) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| bitsandbytes | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| DeepSpeedFP | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

- ✅ = supported, ❌ = not supported
- *Turing does not support Marlin MXFP4
- Intel Gaudi quant support moved to **vLLM-Gaudi** (separate project)
- TPU support: see TPU-Inference docs (not in this matrix)
- Chart changes as vLLM evolves; source of truth: `vllm/model_executor/layers/quantization`

## Out-of-tree quantization plugins

vLLM allows custom quant methods without modifying the codebase:

1. Subclass `QuantizationConfig`, decorate with `@register_quantization_config("my_quant")`
2. Implement required methods:

| Method | Purpose |
|---|---|
| `get_name()` | Quant method name |
| `get_supported_act_dtypes()` | Allowed activation dtypes (e.g. `torch.float16`) |
| `get_min_capability()` | Min GPU compute capability (e.g. 80 = Ampere; −1 = any) |
| `get_config_filenames()` | Config files searched in model dir |
| `from_config(config)` | Build config from model's quant config dict |
| `get_quant_method(layer, prefix)` | Dispatch per layer type; return `None` to skip |

3. Linear layers → return a `QuantizeMethodBase` subclass (extend `UnquantizedLinearMethod`); MoE layers → `FusedMoEMethodBase` (extend `UnquantizedFusedMoEMethod`; implement `get_fused_moe_quant_config`)
4. Reference impl: `Fp8MoEMethod` in `vllm/model_executor/layers/quantization/fp8.py`
5. Use via `LLM(model="...", quantization="my_quant")` after importing the plugin module

## Relevance to this repo

- vLLM is a server-side engine, not the llama.cpp Trial backend this project benchmarks. Value = engine-comparison reference + a possible serving path for existing GGUF quants (GGUF row: GPU-only in vLLM — no CPU fallback).
- NVFP4 reaches vLLM through **NVIDIA Model Optimizer** (see [nvfp4-quantization.md](./nvfp4-quantization.md)); vLLM's own format list has no standalone NVFP4 entry — Marlin FP4 is the 4-bit GPU kernel path.
- The operator host (8 GB, older GPU): relevant rows are bitsandbytes / GPTQ / AWQ / GGUF on NVIDIA; FP8/INT8 paths need Ada+.

## Sources / Verification

- vLLM docs, *Quantization* (latest): https://docs.vllm.ai/en/latest/features/quantization/ (page dated June 12, 2026)
- Hardware matrix + plugin API extracted verbatim from the page above.
