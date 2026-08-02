# AGENTS.md — docs/discovery

## Purpose
User-facing guides for **discovering, evaluating, selecting, and optimizing** local LLMs and inference runtimes (GPU and CPU) that fit the `local-model-autotuning` workflow. Covers tooling (`whichllm`), evaluation methodology (Pareto frontier, Zellinger economic evaluation), quantization strategies, CPU optimization (AVX-512/AMX/NUMA), and engine architectures.

## Ownership
- Owned by: `local-model-autotuning` developers.
- Stable contracts: whichllm CLI contract, Pareto frontier method, scoring rules.

## Local Contracts
- **Read-mostly**: discovery docs are guides for users to follow. Code under `autoresearch/` is the loop surface.
- **No model-specific claims**: docs reference model families (e.g. "Qwen3.6 MoE") not private paths or single-user hardcoded values.
- **No alias registry in repo**: `model-up` alias names, ports, and `models/aliases/*/config.yaml` are machine-local (`/models/` gitignored). Tracked docs use GGUF basenames + benchmark scores only.
- **No external-source citations in technique claims**: methodology names allowed (Zellinger framework), but no HF/Unsloth/blog URLs in technique descriptions.

## Work Guidance
- New guides go here when they describe a reusable methodology, not a single session's data.
- Single-run session logs → `docs/sessions/` (cavity for empirical capture).
- Per-model GGUF specs and architecture notes → `docs/models/`.

## Verification
- Guides should be **runnable** by a user with the documented steps. Steps referencing a tool should include the install path (`uvx whichllm@latest`, etc).
- Cross-check examples against current `whichllm --help` before claiming a flag/option.

## Child DOX Index

### 1. Tooling, Onboarding & Hard Gates
- [`discover-models.md`](./discover-models.md) — end-to-end workflow: discovery (whichllm/llmfit) → Pareto frontier → autoloop target selection.
- [`whichllm-reference.md`](./whichllm-reference.md) — full whichllm CLI reference (commands, flags, profiles, examples).
- [`llmfit-reference.md`](./llmfit-reference.md) — full llmfit CLI/TUI reference (hardware sizing, planning, model search, examples).
- [`agent-onboarding.md`](./agent-onboarding.md) — onboarding guide for future agents.
- [`good-enough-tuning.md`](./good-enough-tuning.md) — default speed path: validation smoke → autoloop `--mode tps` → champion quality check.
- [`agentic-coding-benchmarks.md`](./agentic-coding-benchmarks.md) — migration guide from direct coding tasks to long-horizon agentic coding benchmarks.
- [`claw-eval-leaderboard.md`](./claw-eval-leaderboard.md) — ranked Claw-Eval full/quick scores on this 8 GB rig + operational lessons.
- [`coding-leaderboard.md`](./coding-leaderboard.md) — ranked coding-10 (HE/MBPP/LCB/BC) scores on this 8 GB rig.
- [`pareto-leaderboard.md`](./pareto-leaderboard.md) — global Pareto Set (ctx × TPS × agentic × coding) + Day/Night picks (ADR 0006/0008). Live recompute: `scripts/rank_results.py`.
- [`pareto-selection.md`](./pareto-selection.md) — method note: maximin/Chebyshev Night + ε-constraint IQ band Day ([ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md)).
- [`best-model-8gb-vram.md`](./best-model-8gb-vram.md) — web-sourced selection guide: fastest + smartest model fitting 8 GB VRAM (primary publisher cards only; no local measurements).
- [`agent-shell-hard-gates.md`](./agent-shell-hard-gates.md) — live gate inventory, disable/rollback playbook (§3), threat model (Cursor + Claude Code).
- [`../models/README.md`](../../models/README.md) — nested GGUF store shared with LM Studio.

### 2. Quantization & Low-VRAM Optimizations
- [`quantization-cascade.md`](./quantization-cascade.md) — quantization format selection guide (UD vs standard, VRAM tiers, decision matrix).
- [`quantization-cascade-agent.md`](./quantization-cascade-agent.md) — agent quick reference for quant selection (terse, grog-readable).
- [`nvfp4-quantization.md`](./nvfp4-quantization.md) — NVIDIA NVFP4 4-bit FP format (Blackwell): structure, scaling, memory, ecosystem, repo relevance.
- [`advanced-inference-optimizations.md`](./advanced-inference-optimizations.md) — high-performance techniques: CUDA graphs, tcmalloc/jemalloc, KV cache optimizations, and offload bottlenecks.
- [`low-vram-optimizations.md`](./low-vram-optimizations.md) — strategies for VRAM-constrained GPUs: GGUF/EXL2/HQQ quants, KV cache compression, MoE offloading, and preventing system paging.
- [`../models/vitriol-technique.md`](../models/vitriol-technique.md) — stock `--n-cpu-moe` vs Randozart/VITRIOL DMA fork (study only; Search stays upstream).
- [`local-models-low-vram-configs.md`](./local-models-low-vram-configs.md) — optimal llama.cpp parameters for local and LM Studio models on 8 GB VRAM.

### 3. Inference Engines & Speculative Runtimes
- [`inference-engines-landscape.md`](./inference-engines-landscape.md) — technical comparison & taxonomy guide of LLM inference engines (vLLM, SGLang, TensorRT-LLM, LMDeploy, llama.cpp, Colibrì, TGI).
- [`vllm-quantization.md`](./vllm-quantization.md) — vLLM quantization formats, hardware compatibility matrix, out-of-tree quant plugin API.
- [`vllm-quant-deep-dive.md`](./vllm-quant-deep-dive.md) — per-format quantization mechanics from vLLM source: kernels, group sizes, min SMs, NVFP4 status, config flags, repo relevance.
- [`colibri-inference-engine.md`](./colibri-inference-engine.md) — architectural & performance guide for Colibrì zero-dependency C streaming MoE runtime.
- [`sglang-inference-engine.md`](./sglang-inference-engine.md) — SGLang research guide: RadixAttention, scheduler, structured outputs (XGrammar), speculative decoding, quantization matrix, hardware support, GGUF/Windows limits.
- [`fastest-tps-inference-engine.md`](./fastest-tps-inference-engine.md) — fastest-TPS engine research on this rig (RTX 4060 8 GB / Win): llama.cpp CUDA baseline vs ExLlamaV3/EXL3, TRT-LLM, vLLM/SGLang; measured results.tsv evidence + MoE/MTP speed levers.
- [`speculative-decoding-formats.md`](./speculative-decoding-formats.md) — architectural and performance comparison of speculative formats (MTP vs Eagle vs DFlash vs N-gram).
- [`mtp-baseline-guide.md`](./mtp-baseline-guide.md) — guide on verifying and benchmarking MTP speculative decoding with llama-bench/llama-cli.
- [`small-model-mtp-tps.md`](./small-model-mtp-tps.md) — inventory of local MTP packaging + fair TPS matrix (8 GB, 2026-07-20).
- [`unsloth-qwen-guides.md`](./unsloth-qwen-guides.md) — reference guide on Unsloth dynamic quantization and Qwen fine-tuning mechanics.

### 4. CPU Inference & Build Optimization
- [`cpu-inference-guide.md`](./cpu-inference-guide.md) — CPU-optimized llama.cpp build flags (AVX-512/AMX), Intel vs AMD notes, NUMA, thread affinity, allocators (tcmalloc/jemalloc), and GGUF quant selection for CPU cache.
- [`openvino-genai-cpu-igpu-guide.md`](./openvino-genai-cpu-igpu-guide.md) — OpenVINO GenAI export, INT8/INT4 choices, CPU/iGPU device selection, and reproducible TPS benchmarking.
