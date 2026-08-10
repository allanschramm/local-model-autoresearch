# NVFP4 — NVIDIA 4-bit Floating-Point Format (Blackwell)

NVFP4 is NVIDIA's 4-bit floating-point format for Blackwell Tensor Cores, built to keep model accuracy at ultra-low precision. It sits outside the GGUF/K-quant world: it is a hardware format deployed through NVIDIA's stack (TensorRT-LLM, vLLM), not a llama.cpp quant.

## Format structure

E2M1 micro-float, same as FP4 / MXFP4:

- 1 sign bit, 2 exponent bits, 1 mantissa bit
- Representable range ≈ −6 … +6 (e.g. 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6, and negatives)
- Each encoded value is reconstructed as `x = x_q × s`

NVFP4's two innovations over MXFP4:

1. **Two-level scaling**
   - Per-16-value micro-block scale `s` in **E4M3 (FP8)** precision — fractional (non-power-of-two), so it can fit the block's actual distribution
   - Second-level per-tensor **FP32 scalar** that rescales the tensor so micro-blocks land in the E4M3 encodable range
2. **Half the MXFP4 block size**: 16 values/block vs 32 → twice as many opportunities to match local dynamic range

E4M3 vs E8M0 scaling: E8M0 snaps the scale to nearest 2ⁿ (cheap, can blow up error on the block maximum); E4M3 picks one fractional scale minimizing collective block error. Example in the source blog: E4M3 blocks average MSE 0.08 vs E8M0. E8M0's only advantage is simplicity (no extra per-tensor scalar), adequate for scale-insensitive weights/activations.

## Memory

- 4 bits/value + 1 FP8 scale per 16 values ≈ **4.5 bits/value**, plus one FP32 per tensor
- ≈ **3.5× less memory than FP16**, ≈ **1.8× less than FP8**

## Accuracy & performance (from NVIDIA's data)

- DeepSeek-R1-0528: ≤1% degradation FP8 → NVFP4 via PTQ across 7 evals; AIME 2024 measured 2% *better* than FP8
- Energy: up to 25× (Blackwell) / 50× (Blackwell Ultra) energy efficiency per token vs H100 baseline on GPT-MoE 1.8T
- Memory + accuracy wins scale to rack systems: GB300 NVL72 = 40 TB memory budget (36 Grace Blackwell Ultra Superchips) — relevant for test-time scaling

## Ecosystem / getting started

- Quantize: TensorRT Model Optimizer, LLM Compressor (PTQ / QAT)
- Export: Unified Hugging Face checkpoint; ONNX for non-LLM models
- Serve: TensorRT-LLM and vLLM (early NVFP4 support), SGLang upcoming — in vLLM the NVFP4 path is NVIDIA Model Optimizer (see [vllm-quantization.md](./vllm-quantization.md) for the full vLLM quant matrix)
- Pre-quantized HF checkpoints: DeepSeek-R1-0528, Llama 3, FLUX.1-dev

## Relevance to this repo

- **Hardware-gated**: requires Blackwell (consumer RTX 50-series; datacenter B100/B200/GB200/GB300). Older GPUs — including the operator host's 8 GB card — get no NVFP4 benefit; the Tensor Cores do the micro-scaling, it is not a software-emulated GGUF quant.
- **Not a GGUF/K-quant**: NVFP4 lives in the NVIDIA inference stack (TensorRT-LLM, vLLM). llama.cpp quant surface (K-quants, UD, FP8) is separate.
- Unsloth ships NVFP4 quants (e.g. Qwen3.6 27B / 35B-A3B, MTP-integrated) — see [unsloth-qwen-guides.md](./unsloth-qwen-guides.md) for running them.
- If the operator host ever gets a Blackwell GPU, candidate path: pre-quantized NVFP4 HF checkpoint → vLLM/TensorRT-LLM (not llama.cpp Trials).

## Sources / Verification

- NVIDIA Developer Blog, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference*, Jun 24, 2025 — Alvarez, Almog, Chung, Layton, Stosic, Krashinsky, Aubrey. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Table 1 (FP4 vs MXFP4 vs NVFP4), Figure 4 (E4M3 vs E8M0 MSE), Figure 6 (DeepSeek-R1-0528 evals), Figure 7 (energy efficiency) from the same post.
