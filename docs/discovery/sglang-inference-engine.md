# SGLang — Inference Engine Research Guide

> Research note. Compiled 2026-08-02 against primary sources: [sgl-project/sglang](https://github.com/sgl-project/sglang) `main` tree, [docs.sglang.io](https://docs.sglang.io), [LMSYS blog](https://lmsys.org/blog/), arXiv. Claims cite their owning source inline.

## 1. What It Is

SGLang (Structured Generation Language) is a high-performance serving framework for LLMs and multimodal models, hosted under [LMSYS](https://lmsys.org/about/) and Apache-2.0 ([repo](https://github.com/sgl-project/sglang), 31k+ stars, 2026-08). Originally (Jan 2024) a *structured generation language* + runtime for multi-call LLM programs (agents, reasoning chains, extraction); it has grown into a production serving engine for language, vision, embedding, reward, and diffusion models ([README](https://github.com/sgl-project/sglang/blob/main/README.md)).

Design lineage acknowledged in repo: learned from Guidance, vLLM, LightLLM, FlashInfer, Outlines, LMQL ([README](https://github.com/sgl-project/sglang/blob/main/README.md)).

Origin paper: [arXiv 2312.07104](https://arxiv.org/abs/2312.07104) *SGLang: Efficient Execution of Structured Language Model Programs* — claims up to **5x throughput** vs Guidance and vLLM on agent/reasoning/chat workloads (Llama-7B A10G, Mixtral-8x7B TP8, Jan 2024 [LMSYS blog](https://lmsys.org/blog/2024-01-17-sglang/)).

Current headline features ([README](https://github.com/sgl-project/sglang/blob/main/README.md)):
- Fast runtime: RadixAttention prefix caching, zero-overhead CPU scheduler, prefill-decode (PD) disaggregation, speculative decoding, continuous batching, paged attention, TP/PP/EP/DP parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), multi-LoRA batching.
- Hardware: NVIDIA (GB200/B300/H100/A100/Spark/5090), AMD (MI355/MI300), Intel Xeon CPUs, Google TPUs, Ascend NPUs, Intel Arc XPU, Apple Silicon.
- Adoption: xAI, Cursor, NVIDIA, Intel, LinkedIn, etc.; claims 400,000+ GPUs serving, trillions of tokens/day ([README](https://github.com/sgl-project/sglang/blob/main/README.md)).
- RL/post-training rollout backend: verl, AReaL, slime, Tunix, Miles ([README](https://github.com/sgl-project/sglang/blob/main/README.md)).

Version: latest PyPI release **0.5.16**, requires Python ≥ 3.10 ([PyPI](https://pypi.org/project/sglang/)).

## 2. Core Architecture

### 2.1 RadixAttention (prefix KV caching)

KV cache organized as a **radix tree (trie) over token sequences**; prefixes shared across requests are computed once and reused automatically — no explicit prompt-caching API needed ([blog](https://lmsys.org/blog/2024-01-17-sglang/), [paper](https://arxiv.org/abs/2312.07104)). Benefits: multi-turn agent sessions, shared system prompts, RAG context blocks — any repeated prefix.

- Eviction: LRU (radix tree), with a `UnifiedRadixCache` variant (`SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`) that adds **session-aware eviction**: KV registered to an active `session_id` is evicted only after unreferenced KV; soft references, not pins ([session_radix_cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache)).
- **HiCache**: hierarchical 3-tier KV cache — GPU memory → host RAM → external storage (HF3FS, Mooncake). Flags: `--enable-hierarchical-cache --hicache-ratio 2 --hicache-size 100 --hicache-write-policy write_through` ([hicache_best_practices](https://docs.sglang.io/docs/advanced_features/hicache_best_practices)). This is the closest thing SGLang has to llama.cpp's KV offload — but targeted at long-context/multi-turn datacenter setups, not single-GPU VRAM relief.

### 2.2 Scheduler & batching

- Continuous batching (iteration-level) + chunked prefill (`--chunked-prefill-size` for OOM control) ([FAQ](https://docs.sglang.io/docs/references/faq)).
- **Zero-overhead batch scheduler** (v0.4): CPU scheduler runs one batch ahead, overlapping scheduling/prefix matching with GPU compute; measured 1.1x throughput over v0.3, GPU idle time ≈ 0 under Nsight ([v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- Cache-aware load balancer (v0.4): routes requests to the DP worker with the best predicted prefix hit rate — up to 1.9x throughput, 3.8x hit rate ([v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- CUDA graphs, optional torch.compile (`--enable-torch-compile --torch-compile-max-bs`), overlap scheduling pipelining decode batches.

### 2.3 Parallelism

- Tensor (TP), pipeline (PP), expert (EP), data (DP) parallelism; DP attention specifically for DeepSeek-style MLA models (up to 1.9x decode throughput, v0.4).
- **Expert parallelism** distributes MoE experts across GPUs; backends for all-to-all: DeepEP, Mooncake (RDMA), NIXL-EP, MORI (AMD), FlashInfer; MoE compute: grouped GEMM, FlashInfer TRTLLM; flags `--moe-a2a-backend --moe-runner-backend`, with TBO/SBO overlap and EPLB load balancing ([expert_parallelism](https://docs.sglang.io/docs/advanced_features/expert_parallelism)).
- **PD disaggregation**: separate prefill (compute-bound) and decode (memory-bound) worker pools over Mooncake/NIXL transfer engines + router/gateway ([pd_disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation)).

### 2.4 Determinism

Outputs are **not deterministic even at temperature 0** — dynamic batching (~95% of variance) and prefix caching dispatch different CUDA kernels. `--disable-radix-cache` + one request at a time ≈ deterministic; newer `--enable-deterministic-inference` mode exists ([FAQ](https://docs.sglang.io/docs/references/faq), [blog](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)). Relevant to anyone using SGLang as an eval harness: batch shape changes scores.

## 3. Structured Outputs

Constraints: **JSON schema, regex, EBNF** (one per request), plus **structural tags** for tool/function-call output. Three grammar backends ([structured_outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs)):

| Backend | JSON schema | Regex | EBNF | Notes |
| :-- | :-: | :-: | :-: | :-- |
| **XGrammar** (default) | ✅ | ✅ | ✅ | Compile-time grammar compilation, GPU-side; uses **GGML BNF format** (same grammar lineage as llama.cpp GBNF) |
| Outlines | ✅ | ✅ | ❌ | `--grammar-backend outlines` |
| Llguidance | ✅ | ✅ | ✅ | `--grammar-backend llguidance` |

- Speed: compressed FSM gave 3x faster JSON decoding (Feb 2024 [blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)); XGrammar integration up to **10x faster** structured outputs ([v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- Reasoning models (DeepSeek R1, QwQ): `--reasoning-parser deepseek-r1|qwq` keeps `<think>...</think>` free-form while constraining the final answer ([structured_outputs_for_reasoning_models](https://docs.sglang.io/docs/advanced_features/structured_outputs_for_reasoning_models)).
- API: `response_format={"type":"json_schema",...}` (OpenAI-compatible), `sampling_params: {"json_schema"|"regex"|"ebnf"|"structural_tag"}` (native `/generate`), Pydantic schemas supported everywhere.

## 4. Speculative Decoding

Six algorithms ([speculative_decoding](https://docs.sglang.io/docs/advanced_features/speculative_decoding)):

| Method | Draft source | Flag | Constraints |
| :-- | :-- | :-- | :-- |
| EAGLE-2 | EAGLE draft model (feature drafting + tree) | `--speculative-algorithm EAGLE` | tune num-steps/topk/num-draft-tokens |
| EAGLE-3 | EAGLE3 draft model | `--speculative-algorithm EAGLE3` | best throughput |
| MTP | built-in multi-token-prediction heads (DeepSeek, MiMo) | EAGLE workflow, small topk | no separate draft needed |
| DFLASH | DFlash draft checkpoint, linear block verify | `--speculative-algorithm DFLASH` | no DP attention, pp=1 |
| STANDALONE | smaller draft LLM | `--speculative-algorithm STANDALONE` | no DP attention |
| NGRAM | n-gram cache from prior tokens | `--speculative-algorithm NGRAM` | CUDA-only, no draft |

Measured (Llama-3.1-8B, MT-bench, 1x H100, [docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)): baseline 158 t/s → EAGLE-2 244 t/s → EAGLE-3 373 t/s. Adaptive variant exists for changing acceptance over time ([adaptive_speculative_decoding](https://docs.sglang.io/docs/advanced_features/adaptive_speculative_decoding)). Next-gen DFlash/Spec-V2 announced 2026-06 ([blog](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)).

## 5. Quantization

### 5.1 Method support matrix ([quantization docs](https://docs.sglang.io/docs/advanced_features/quantization))

| Method | NVIDIA | AMD (MI300X/325X/350X) | Ascend NPU | Notes |
| :-- | :-: | :-: | :-: | :-- |
| `fp8` | ✅ | ✅ | WIP | Aiter/Triton on AMD |
| `mxfp4` | ✅ | ✅ | ✅ (A5) | CDNA3/4 MXFP on GPU |
| `mxfp8` | ❌ | ❌ | ✅ (A5) | Ascend-only |
| `blockwise_int8` | ✅ | ✅ | ❌ | Triton |
| `w8a8_int8` / `w8a8_fp8` | ✅ | ✅ | ❌ | CUTLASS kernels |
| `awq` / `gptq` | ✅ | ✅ | ✅ | Triton/vLLM kernels on AMD, CANN on Ascend |
| `compressed-tensors` | ✅ | ✅ | partial | |
| `quark` / `quark_int4fp8_moe` / `quark_mxfp4` | — | ✅ | — | AMD paths |
| `auto-round` | ✅ | ✅ | partial | Intel INC |
| `awq_marlin` / `gptq_marlin` | ✅ | ❌ | ❌ | CUDA-only Marlin |
| **`gguf`** | ✅ | ❌ | ✅ | CUDA kernels in sgl-kernel; Ascend CPU pre-dequant at load |
| `modelopt_fp8` | ✅ (SM90+) | ❌ | ❌ | NVIDIA ModelOpt |
| `modelopt_fp4` / `nvfp4_online` | ✅ (SM100+) | ❌ | ❌ | Blackwell FP4; online MoE-only NVFP4 |
| `petit_nvfp4` | ❌ | ✅ (MI250+) | ❌ | NVFP4 on ROCm |
| `bitsandbytes` | ✅ | experimental | ❌ | |
| `torchao` (`int4wo-128` etc.) | ✅ | partial | ❌ | PyTorch AO |
| `modelslim` | ❌ | ❌ | ✅ | Ascend |

- Offline (pre-quantized) loading is preferred; quantization method auto-detected from HF config — **do not** also pass `--quantization`. Online quantization flags exist (`--quantization fp8`, `--torchao-config int4wo-128`, `--quantization nvfp4_online`, `--quantization auto-round-int8`).
- FP4/FP8 GEMM backends: `--fp8-gemm-backend` / `--fp4-gemm-backend` — deep_gemm (SM90/100), flashinfer_trtllm/cutlass (SM100/120), marlin (SM80-90), cutlass, triton fallback, aiter (ROCm) ([quantization docs](https://docs.sglang.io/docs/advanced_features/quantization)).
- KV cache quantization (FP8) documented under [quantized_kv_cache](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache).

### 5.2 GGUF specifics

- Load via `--load-format gguf`, **auto-detected from a `.gguf` model path** ([model_loading](https://docs.sglang.io/docs/advanced_features/model_loading)).
- Implementation: CUDA dequant kernels in sgl-kernel (`python/sglang/kernels/aot/csrc/quantization/gguf/`, `python/sglang/srt/layers/quantization/gguf.py`), tests in `test/registered/quant/test_gguf.py` ([repo tree](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/layers/quantization/gguf.py)).
- Supported only on NVIDIA (and Ascend via load-time CPU dequant). **Not on AMD.** Docs still list gguf under "online quantization coming soon".
- Other load formats: auto (safetensors→bin), pt, npcache, dummy, sharded_state, fastsafetensors (GPU Direct Storage), layered, bitsandbytes, mistral, runai_streamer (S3/GCS/Azure), remote, remote_instance ([model_loading](https://docs.sglang.io/docs/advanced_features/model_loading)).

## 6. Hardware Support

| Platform | Status / notes | Source |
| :-- | :-- | :-- |
| **NVIDIA** | Primary target. FlashInfer default attention backend, **sm75+** minimum; sm90/100/120 needed for FP8/FP4/DeepGEMM paths. Fallback `--attention-backend triton --sampling-backend pytorch`. | [install](https://docs.sglang.io/docs/get-started/install), [attention_backend](https://docs.sglang.io/docs/advanced_features/attention_backend) |
| **AMD** | ROCm (MI300X/MI325X/MI350X), Aiter acceleration, DeepSeek V3/R1 day-one support | [amd_gpu](https://docs.sglang.io/docs/hardware-platforms/amd_gpu) |
| **Intel Xeon CPU** | Intel **AMX** required (4th-gen Xeon Scalable+). Docker images tagged `-xeon`; `SGLANG_USE_CPU_ENGINE=1`; TP rank = sub-NUMA cluster; W8A8 quant recommended; torch.compile for decode. Dense ≤10B or MoE ≤10B-active on single socket; >20B needs flagship dual-socket. | [cpu_server](https://docs.sglang.io/docs/hardware-platforms/cpu_server) |
| **Apple Silicon** | MLX backend: `SGLANG_USE_MLX=1`, needs Xcode CLT, `mlx-community/*-4bit/-8bit` repos or on-the-fly `--quantization mlx_q4|mlx_q8` (gs=64). Slower/limited vs CUDA. | [apple_metal](https://docs.sglang.io/docs/hardware-platforms/apple_metal) |
| **Intel Arc XPU** | Arc B-series only, BF16; source install, torch-xpu wheels. | [xpu](https://docs.sglang.io/docs/hardware-platforms/xpu) |
| **Google TPU** | Separate sglang-jax backend, TPU v6e/v7. | [tpu](https://docs.sglang.io/docs/hardware-platforms/tpu) |
| **Ascend NPU** | A2/A3/A5, CANN kernels, extensive model cookbooks. | [ascend](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation) |
| **NVIDIA Jetson** | AGX Orin, JetPack 6.1+, `--dtype half --context-length 8192`, torchao int4wo-128 recommended. Proof it runs on consumer-class GPUs. | [nvidia_jetson](https://docs.sglang.io/docs/hardware-platforms/nvidia_jetson) |

Attention backends (MHA/MLA): FlashInfer, FA3, FA4, Triton, FlashMLA (DeepSeek MLA), TRTLLM MLA, hybrid — auto-selected by hardware/architecture ([attention_backend](https://docs.sglang.io/docs/advanced_features/attention_backend)).

## 7. Installation & Serving

- `uv pip install --prerelease=allow sglang` (Python ≥3.10). **CUDA 13 by default**; CUDA 12 needs torch cu129 wheels + `sglang-kernel`/`sgl-deep-gemm` from `docs.sglang.ai/whl/cu129/`. Nightly wheels via `--extra-index-url https://docs.sglang.ai/whl/cu130/` ([install](https://docs.sglang.io/docs/get-started/install)).
- Docker: `lmsysorg/sglang` — `latest`/`dev` mutable tags (pin immutable `v0.5.16`), `-runtime` variant ~40% smaller, `-cu129`/`-cu12` for CUDA 12, `-xeon` for CPU ([install](https://docs.sglang.io/docs/get-started/install)).
- Serve: `sglang serve --model-path <id> --host 0.0.0.0 --port 30000` (new CLI) or `python -m sglang.launch_server`. Ready log: "The server is fired up and ready to roll!".
- APIs: OpenAI-compatible `/v1` (chat, completions, embeddings, rerank), native `/generate`, offline `sgl.Engine` Python API, plus the original frontend DSL ([openai_api](https://docs.sglang.io/docs/basic_usage/openai_api), [offline_engine_api](https://docs.sglang.io/docs/basic_usage/offline_engine_api)).
- Benchmark: `python -m sglang.bench_serving --dataset-name random --random-input-len 1024 --random-output-len 1024 --num-prompts 1 --request-rate inf`.

## 8. Limitations

- **No Windows support.** Install docs target Linux; sgl-kernel has no Windows build path — issues [#2249](https://github.com/sgl-project/sglang/issues/2249), [#7766](https://github.com/sgl-project/sglang/issues/7766) closed on inactivity with users still asking. Windows users run WSL2/Docker.
- **GGUF is NVIDIA-only** (plus Ascend load-time dequant); AMD ROCm has no GGUF path ([quantization docs](https://docs.sglang.io/docs/advanced_features/quantization)).
- Big footprint: CUDA 12/13 wheels, FlashInfer, DeepGEMM, kernels compiled per-arch; not a lightweight embeddable runtime like llama.cpp.
- Nondeterminism under batching (see §2.4).
- Online quantization methods ("soon" list in docs: awq, gptq, marlin, bitsandbytes, gguf) lag offline paths ([quantization docs](https://docs.sglang.io/docs/advanced_features/quantization)).
- Mixed-bit / quantized-MoE offline loading has kernel gaps (auto-round path) ([quantization docs](https://docs.sglang.io/docs/advanced_features/quantization)).

## 9. Repo Relevance (local-model-autotuning)

| Axis | llama.cpp (this repo's engine) | SGLang |
| :-- | :-- | :-- |
| Target | single-user local, CPU→GPU, Windows native | multi-tenant serving, Linux/datacenter, single-GPU up to clusters |
| Weights | GGUF first-class | safetensors first; **GGUF only on NVIDIA**, no Windows build |
| Windows | ✅ native | ❌ (WSL2/Docker) |
| 8 GB VRAM rig | ✅ mainline (MoE `--n-cpu-moe`) | ❌ — scheduler/CUDA-graph/FlashInfer assumptions target ≥sm75 with headroom; no partial offload story for dense models |
| Prefix caching | static slot-based (caveat: no token-level reuse) | RadixAttention token-level radix tree — the differentiator |
| Structured output | GBNF grammars | XGrammar (same GGML BNF lineage) + JSON schema + EBNF + structural tags, 10x faster claims |
| Speculative | MTP via llama.cpp (draft heads) | EAGLE-2/3, MTP, DFLASH, NGRAM, STANDALONE |
| KV offload | llama.cpp KV offload to CPU RAM | HiCache 3-tier (GPU→RAM→storage) |
| MoE big-model | Colibrì / `--n-cpu-moe` on CPU+RAM | EP across GPUs (DeepEP/NIXL), needs multi-GPU |

**Verdict:** SGLang is not a candidate engine for this rig (8 GB VRAM, Windows, GGUF store). It stays relevant here as (1) the reference implementation for token-level radix prefix caching — the biggest architectural gap vs llama.cpp for multi-turn agent workloads, (2) the standard for server-side structured output (XGrammar; note its grammar format is the same GGML BNF family llama.cpp grammars use, so XGrammar-shaped schemas port), and (3) the engine to reach for if a future rig is an NVIDIA Linux box with 24 GB+ VRAM and a serving/agentic workload — GGUF baselines from this repo can be served directly via `--load-format gguf`. EAGLE/MTP draft models consume significant extra VRAM, so on modest GPUs the no-draft NGRAM path or llama.cpp MTP remains the cheaper speculative option.

## 10. Sources

- Repo & README: https://github.com/sgl-project/sglang
- Docs (docs.sglang.io ↔ `docs_new/docs/` in repo `main`): install, quantization, structured_outputs, speculative_decoding, session_radix_cache, hicache_best_practices, expert_parallelism, pd_disaggregation, model_loading, attention_backend, cpu_server, apple_metal, xpu, tpu, nvidia_jetson, faq
- Paper: https://arxiv.org/abs/2312.07104
- LMSYS blogs: 2024-01-17 (RadixAttention), 2024-02-05 (compressed FSM), 2024-12-04 (v0.4), 2025-09-22 (deterministic), 2026-06-15 (DFlash/Spec-V2)
- PyPI: https://pypi.org/project/sglang/ (0.5.16)
- GitHub issues: #2249, #7766 (Windows/sgl-kernel)
