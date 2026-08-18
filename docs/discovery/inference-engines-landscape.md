# High-Performance LLM Inference Engines — Architecture & Taxonomy Guide

## Executive Summary

The LLM inference engine ecosystem has matured into distinct specialized tiers based on deployment constraints (datacenter server high-concurrency vs local consumer hardware) and execution strategies (dynamic runtime vs compiled engine graph).

This guide surveys the primary inference engines, analyzing their memory managers, attention mechanisms, KV cache strategies, and hardware targets.

> Fastest-TPS on the operator host (discrete 8 GB-class NVIDIA / Windows): see [fastest-tps-inference-engine.md](./fastest-tps-inference-engine.md) — llama.cpp CUDA stays baseline; ExLlamaV3/EXL3 is the only Windows-capable raw-TPS challenger.

### Comparing engines without confusing formats

| Level | Weights | What you may claim |
| :-- | :-- | :-- |
| **Same-bytes** | Identical checkpoint on every engine that can load it (e.g. GGUF on llama.cpp and on vLLM/SGLang NVIDIA GGUF paths) | Engine difference |
| **Native-stack** | Best practical pack per engine (GGUF vs AWQ vs NVFP4, …) | Compound engine+format — label it |

vLLM/SGLang are Linux-first (WSL2/Docker on Windows). Size every pack against Baseline `VRAM_LIMIT_MB` / no-spill rules; HF multimodal 4-bit packs are often much larger than text GGUF Q4 of the same family ([quantization-cascade.md](./quantization-cascade.md) § HF vs GGUF).

---

## Quantization equivalence — the "Q4_K_M of each engine"

Q4_K_M is llama.cpp's classic 4-bit "good enough" pack: mixed 4/6-bit (attention + first/last blocks at 6-bit), ~0.1–0.15 PPL of FP16 on WikiText, 1–3% MMLU loss, ~95% of BF16 quality on 8B (**external / unverified on this rig**). GGUF-side quality cascade (UD-Q4_K_XL / UD-Q4_K_M > Q4_K_M > IQ4_XS for compression): [quantization-cascade.md](./quantization-cascade.md). Every other engine has a format filling the same niche (≈4-bit, ~1–3% quality-loss band):

| Engine | Q4_K_M-equivalent | Hardware gate | Same-bytes path? |
| :-- | :-- | :-- | :-- |
| **llama.cpp** | Q4_K_M (classic) · UD-Q4_K_XL/UD-Q4_K_M (best 4-bit) · IQ4_XS (smallest) | any (CPU→GPU) | — (native) |
| **vLLM** | AWQ W4A16 gs128 (best 4-bit quality) · GPTQ W4A16 (4/8-bit flexible) · NVFP4 W4A16 (ModelOpt) | AWQ/GPTQ SM75+ (Marlin; Conch SM80+, Machete SM90+) · NVFP4 W4A4 SM100+ / W4A16-Marlin SM75+ | GGUF via `vllm-gguf-plugin` (OOT, experimental, GPU-only) |
| **SGLang** | AWQ/GPTQ (Triton/vLLM kernels; `awq_marlin`/`gptq_marlin` CUDA-only) · torchao int4wo-128 · NVFP4 (SM100+) | matrix: [sglang-inference-engine.md](./sglang-inference-engine.md) §5.1 | GGUF (`--load-format gguf`, NVIDIA-only) |
| **TensorRT-LLM** | FP8 · NVFP4 · MXFP4 (ModelOpt) · INT4 AWQ (legacy) | compiled per-GPU; W4A4 needs Blackwell | ❌ |
| **ExLlamaV3** | EXL3 / QTIP (~4 bpw) | consumer NVIDIA, source build on Windows | ❌ (EXL3-only format) |
| **LMDeploy** | AWQ / INT4 / INT8 | NVIDIA datacenter | ❌ |
| **TGI** | GPTQ / AWQ / FP8 | datacenter | ❌ |

Best-per-engine on this repo's hardware class (discrete 8 GB-class NVIDIA, SM89):

- **llama.cpp**: UD-Q4_K_XL — repo-measured (Qwen3.5-9B-UD, gemma-4-E4B rows in `results.tsv`; cascade doc).
- **vLLM / SGLang**: AWQ (Marlin W4A16, SM75+ — runs on this GPU class) — or GGUF when a same-bytes claim is wanted (vLLM plugin experimental/under-optimized; SGLang GGUF NVIDIA-only). See [vllm-quant-deep-dive.md](./vllm-quant-deep-dive.md) §5 for the SM75/SM80-usable list (GPTQ/AWQ/INT8/FP8-W8A16-Marlin/GGUF; bnb = slow dequant path).
- **Blackwell-only tier** (future rig): NVFP4 W4A4/W4A16 or MXFP4 — native FP4 tensor cores, ~2.3× INT4 throughput (**external**); FP8 W8A8 is the 8-bit tier (Q8_0-class quality), **not** a Q4 equivalent.

Quality ordering (**external / unverified on this rig**): UD-Q4_K_XL ≈ AWQ > GPTQ ≈ Q4_K_M > Q4_0. NVFP4 W4A16 sits in the AWQ band (2–4× better KLD than W4A4 from the same weights — config switch, not re-quant; NVIDIA forums 2026); NVFP4 W4A4 slightly behind INT4/AWQ on some tasks; AWQ beats GPTQ in PPL/MMLU and is more calibration-robust (AWQ paper arXiv 2306.00978; gingerlabs 2024; Microsoft Data Science guide). Caveat: one vLLM-GGUF-plugin benchmark showed GGUF worst-PPL-but-best-HumanEval among quants — GGUF path was unoptimized there; not a llama.cpp verdict (r/LocalLLaMA, 2026-05).

**TBDs (research sharpened, still open):**

- **TBD:** AWQ vs GGUF Q4_K_M quality parity **on this rig** — external evidence only; no local measurement. Falsifiable: same model, same tasks, llama.cpp Q4_K_M vs SGLang/vLLM AWQ (needs the SGLang path from issue #59).
- **TBD:** SGLang `--load-format gguf` same-bytes run — support documented (NVIDIA-only), never executed on this hardware (`venv-sglang` absent; upstream no-Windows).
- **TBD:** NVFP4 W4A16 vs W4A4 delta on Qwen3.6-35B-A3B (Unsloth NVFP4 GGUF) — Blackwell-only, out of rig scope.

---

## Technical Comparison Matrix

| Engine | Primary Memory Innovation | KV Cache Sharing Mechanism | Primary Target Hardware | Ideal Workload Pattern | Status / Lifecycle |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vLLM** | PagedAttention (Virtual Memory Paging) | Block-level Prefix Caching | Datacenter GPUs (NVIDIA/AMD/Intel) | General Production Serving (Multi-tenant API) | Active (Production Standard) |
| **SGLang** | RadixAttention (Trie/Radix Tree) | Token-level Radix Tree Sharing | Datacenter / High-End Workstation GPUs | RAG, Agentic Multi-turn, Structured JSON | Active (Frontier Speed) |
| **TensorRT-LLM** | TensorRT Compiled Engine Graph | Static / Custom Memory Pools | NVIDIA Enterprise GPUs (H100/H200/B200) | Static High-Traffic Monolithic Models | Active (Peak Throughput) |
| **LMDeploy** | TurboMind Engine / Memory Pools | Page-based KV Allocation | NVIDIA Datacenter GPUs | High-throughput server deployment | Active |
| **llama.cpp** | Mmap / Direct CPU/GPU Offload | Static / Slot-based KV | Consumer CPUs, Apple Silicon, Consumer GPUs | Single-user local execution, Edge devices | Active (Local Standard) |
| **Colibrì** | 3-Tier Hierarchy (VRAM $\rightarrow$ RAM $\rightarrow$ NVMe) | LRU Cache + Usage Ledger (`.coli_usage`) | Consumer PCs (~25 GB RAM) | Ultra-large MoE (GLM-5.2 744B) on low RAM | Active (Streaming MoE) |
| **TGI** | Paged KV / FlashAttention | Chunked Prefix Caching | Datacenter GPUs | Legacy HuggingFace serving | Maintenance Mode |

---

## Detailed Engine Breakdown

### 1. vLLM (PagedAttention Architecture)
- **Core Mechanism**: Introduced **PagedAttention**, dividing KV cache into fixed-size physical memory blocks, eliminating external memory fragmentation and reducing internal fragmentation to under 4%.
- **Scheduling**: Continuous Batching (iteration-level scheduling) ensuring decode steps execute without waiting for full sequence completion.
- **Hardware & Architecture**: Broadest support across NVIDIA, AMD (ROCm), Intel Gaudi, and Google TPU. Supports over 400+ model architectures.
- **Quantization Support**: AWQ, GPTQ, bitsandbytes, GGUF (GPU-only), FP8/INT8/INT4 via LLM Compressor, NVFP4 via NVIDIA Model Optimizer. Full matrix: [vllm-quantization.md](./vllm-quantization.md).
- **Strengths**: Enterprise standard, robust OpenAI API server compatibility, seamless multi-GPU Tensor Parallelism (TP) and Pipeline Parallelism (PP).

### 2. SGLang (RadixAttention & Structured Decoding)
- **Core Mechanism**: Uses **RadixAttention**, organizing the KV cache as a dynamic radix tree (trie). Matches prefix tokens hierarchically rather than via discrete block chunks.
- **Optimization Strategy**: Token-level prefix reuse drastically lowers **Time to First Token (TTFT)** when multiple queries share system prompts, RAG context blocks, or multi-turn agent histories.
- **Structured Output**: Native integration with compressed finite-state machine (FSM) grammar execution for accelerated JSON/schema decoding.
- **Strengths**: Best-in-class performance for agentic coding workflows, long-context RAG pipelines, and complex prompt DAGs.
- **Deep dive**: [sglang-inference-engine.md](./sglang-inference-engine.md) — RadixAttention internals, XGrammar structured outputs, speculative decoding (EAGLE/MTP/DFlash), quantization matrix, hardware support, GGUF/Windows limits.

### 3. TensorRT-LLM (Compiled Graph & Custom Kernels)
- **Core Mechanism**: Converts PyTorch model definitions into optimized C++ **TensorRT engine graphs**, performing ahead-of-time (AOT) kernel fusion, fp8/fp4 precision selection, and custom GEMM tuning.
- **In-Flight Batching**: Highly customized C++ execution runtime with custom CUDA kernels (e.g. FlashAttention-3, XQA kernels).
- **Trade-Offs**: Requires compilation per GPU architecture and configuration. High operational setup complexity; less flexible for fast model iteration.
- **Strengths**: Industry-peak raw token throughput for fixed production models on NVIDIA hardware.

### 4. LMDeploy (TurboMind Backend)
- **Core Mechanism**: Developed by OpenMMLab, utilizing the TurboMind C++ inference engine for aggressive memory allocation and fused matrix multiplications.
- **Quantization Support**: Native support for AWQ, INT4, and INT8 weight-only and KV-cache quantization.
- **Strengths**: Extremely low latency and high concurrency serving on NVIDIA H100/A100 clusters with minimal memory overhead.

### 5. llama.cpp (Cross-Platform & Edge Execution)
- **Core Mechanism**: Pure C/C++ engine utilizing custom GGUF quantization formats (Q4_K_M, IQ3_XS, etc.).
- **Hardware Agnostic**: Runs across x86 CPU (AVX2/AVX-512/AMX), ARM (NEON), Apple Silicon (Metal), NVIDIA (CUDA), AMD (HIP), and Vulkan.
- **Local Autotuning Target**: The core backend engine integrated into `local-model-autotuning` via `llama-server` and `llama-cli`. Default Search path is **upstream** `llama.cpp` (MoE: `--n-cpu-moe`). Arch forks only when required by the GGUF.
- **Related fork (not default):** [Randozart/VITRIOL](https://github.com/Randozart/VITRIOL) — page-locked host experts + GPU PCIe DMA (+ optional Chimera CUDA/Vulkan). Study notes: [vitriol-technique.md](../models/vitriol-technique.md).

### 6. Colibrì (Streaming MoE Runtime)
- **Core Mechanism**: Specialized zero-dependency single-file C runtime (`c/glm.c`) designed for 700B+ MoE models (GLM-5.2).
- **Storage Tiering**: Holds dense core in RAM (~9.9 GB int4) while streaming 19,456 routed experts (~370 GB int4) from NVMe SSD on demand.
- **Strengths**: Enables frontier-scale MoE execution on ~25 GB RAM consumer PCs without quality degradation.

---

## Architectural Decision Framework

```
                          [Inference Requirement]
                                     |
           +-------------------------+-------------------------+
           |                                                   |
   [Local / Edge / Consumer]                          [Datacenter / Server]
           |                                                   |
     +-----+-----+                                       +-----+-----+
     |           |                                       |           |
[MoE Stream] [General Local]                       [Prefix Heavy] [Fixed High-Traffic]
     |           |                                       |           |
 [Colibrì]   [llama.cpp]                             [SGLang]   [TensorRT-LLM]
                                                         |
                                                 [General Production]
                                                         |
                                                       [vLLM]
```

1. **Choose vLLM**: For standard enterprise deployment needing broad model support, reliable scaling, and OpenAI-compatible API endpoints.
2. **Choose SGLang**: For long-context RAG, multi-turn AI agents, or heavy prompt sharing where RadixAttention token reuse delivers 20–40% TTFT reduction.
3. **Choose TensorRT-LLM**: For maximum static hardware throughput on dedicated NVIDIA infrastructure where engine compilation cost is acceptable.
4. **Choose llama.cpp / Ollama**: For local development, Apple Silicon, CPU-only execution, or GGUF quantized model deployment.
5. **Choose Colibrì**: For streaming 700B+ MoE models on RAM-constrained consumer hardware.
