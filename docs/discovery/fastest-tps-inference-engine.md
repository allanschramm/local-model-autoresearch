# Fastest-TPS Inference Engine — Research Guide

> Research note. Compiled 2026-08-02. Scope: **operator `discrete_gpu` host** (Baseline `VRAM_LIMIT_MB` / class from `scripts/check_hardware.py` — no SKU). Question: which inference engine maximizes single-user generation TPS. Claims cite owning primary source; cross-engine TPS claims that are not measured on the operator host are flagged as unverified.

## 1. Ground truth: what the operator host already achieves

Measured llama.cpp (upstream, CUDA build, vendored submodule at tag `b10173`) numbers from `results.tsv` — the repo's only 8GB-class-TPS ground truth (repo rule: TSV wins over docs/estimates):

| Model | Quant | TPS | ctx | Why fast |
| :-- | :-- | --: | --: | :-- |
| LFM2.5-8B-A1B | Q4_K_M | **178.5** | 65k | MoE, 1B active params — compute-bound decode on tiny active set |
| LFM2.5-1.2B | Q8_0 | 166.4 | 65k | tiny dense |
| gemma-4-E4B + draft-MTP | Q4_K_XL | 122.0 (vs 67.6 base) | 65k | MTP speculative +80% ([small-model-mtp-tps.md](./small-model-mtp-tps.md)) |
| Qwen3.5-4B-MTP | Q4_K_M | 87.0 | 131k | MTP embedded heads |
| Qwen3.5-9B-MTP | Q4_K_M | 67.5 | 131k | MTP embedded heads |
| Ornith-1.0-9B-MTP | Q4_K_M | 63.7 | 32k | MTP |
| Qwythos-9B-v2 | Q4_K_M | 41.6 | 100k | dense 9B at huge ctx |
| Qwen3.8-2B (SGLang 0.5.17, WSL2) | fp16 safetensors | 56.8 | 32k | dense 2B; ≈ llama.cpp 60.5 — same class, no engine win ([issue #59](./sglang-inference-engine.md#91-measured-on-the-operator-host-2026-08-18-issue-59)) |

Pattern: **small-active MoE + speculative MTP heads beat dense 9B by 2.5–4x** on the operator host. Context is expensive: dense 9B at 100k ctx ≈ 42 TPS, at 32k ctx ≈ 64 TPS (KV cache + attention dominate).

## 2. Candidate matrix (Windows, 8 GB VRAM)

| Engine | Windows native | GGUF (repo store) | Fits 8 GB | TPS evidence | Verdict |
| :-- | :-: | :-: | :-: | :-- | :-- |
| **llama.cpp CUDA** (current) | ✅ | ✅ | ✅ | measured in-repo (above) | **Baseline — stays** |
| **ExLlamaV3 / EXL3** | ✅ (source build) | ❌ (EXL3 format) | ✅ (consumer target) | vendor claims memory-bound GEMM; 8GB-class numbers not published | **Only credible raw-TPS challenger** |
| ExLlamaV2 / EXL2 | ✅ | ❌ | ✅ | vendor 4090 tables | **archived → ExLlamaV3** |
| koboldcpp / Ollama | ✅ | ✅ | ✅ | same llama.cpp kernels | wrapper, no TPS gain |
| TensorRT-LLM (Windows) | ⚠️ 2023-era, repo removed | ❌ | — | "up to 4x" (2023 blog, vague baseline) | deprecated on Windows; model coverage stale |
| vLLM | ❌ (WSL2) | ✅ (CUDA) | ⚠️ | — | server engine, Linux-first, heavy for single-user 8 GB |
| SGLang | ❌ (WSL2) | ⚠️ (NVIDIA-only) | ⚠️ (2B-class only) | measured: Qwen3.8-2B 56.8 t/s ≈ llama.cpp (issue #59) | server engine, Linux-first; no TPS win on 8 GB-class; see [sglang-inference-engine.md](./sglang-inference-engine.md) |
| LMDeploy | ❌ | ❌ | — | — | Linux |
| llama.cpp Vulkan | ✅ | ✅ | ✅ | CUDA > Vulkan on NVIDIA (qualitative) | fallback only |
| Colibrì | ✅ | n/a | — | — | niche: biggest MoE on RAM, not fastest TPS ([colibri-inference-engine.md](./colibri-inference-engine.md)) |

## 3. ExLlamaV3 — the one challenger (primary sources)

ExLlamaV2 is **archived** ("This project is archived for now. Development continues on ExLlamaV3") ([ExLlamaV2 README](https://github.com/turboderp/ExLlamaV2)). ExLlamaV3 ([repo](https://github.com/turboderp-org/exllamav3), Apache? — active, pushed 2026-07-31, 1.1k stars) is the live project: "optimized quantization and inference library for running LLMs locally on modern consumer-class GPUs."

Headline facts from the [ExLlamaV3 README](https://github.com/turboderp-org/exllamav3):
- **EXL3 format**: streamlined variant of **QTIP** (Cornell RelaxML); one-step quantization (fused Viterbi kernel), minutes for small models on one RTX 4090; coherent at 1.6 bpw (Llama-3.1-70B in <16 GB). TPS-relevant: **Marlin-inspired GEMM kernel ≈ memory-bound at 4 bpw on RTX 4090**; author notes efficiency on Ampere (30-series) "still needs work" — Ada (8GB-class, sm89) unverified.
- **Tensor-parallel + expert-parallel** for consumer setups; 2–8-bit KV cache quantization (helps long ctx on 8 GB); speculative decoding; LoRA; multimodal.
- **Windows**: build from source with VS Build Tools + CUDA ≥12.4 + `flash-attn-2` wheel + `triton-windows` (recommended). Prebuilt wheels on releases are `linux_x86_64` only — Windows = source/JIT build.
- **NVIDIA-only** ("ROCm support" on to-do list).
- Arch list (vs this repo's inventory, [README](https://github.com/turboderp-org/exllamav3)):
  - ✅ **LFM 2.5** (`Lfm2MoeForCausalLM`) — the 178-TPS king *could* run on ExLlamaV3
  - ✅ **Qwen 3.5 / Qwen 3.5 MoE** — dense 4B/9B supported (MTP-head behavior unverified)
  - ✅ **Laguna 2.1** (`LagunaForCausalLM`) — Laguna-XS
  - ❌ **Gemma 4 E2B/E4B not supported** — kills the 122-TPS MTP champion
  - ❓ Ornith-1.0-9B, Qwythos-9B, POCKET-35B, KAT-Coder — archs not in the supported list; must check HF config per model before assuming anything

Historical TPS context (ExLlamaV2-era, **4090**, primary README tables): Llama-7B EXL2 4.0 bpw **211 t/s**, TinyLlama-1.1B EXL2 4.0 bpw 700 t/s. Directional secondary sources claim ExLlamaV2 was 50–85% faster than llama.cpp on 3090/4090 (blog aggregator; ExLlamaV2-era; not primary, treat as directional). **No primary 8GB-class-class ExLlamaV3-vs-llama.cpp TPS benchmark exists in the sources gathered** — any speedup on this specific rig is unverified until measured.

### Why not adopt now (repo-relevant)

1. **Format lockout**: EXL3 is a new quant format; the repo's entire GGUF store + `docs/models/*` cards + Pareto `results.tsv` lineage are GGUF. Every candidate model must be re-quantized from HF safetensors (new format, new quality baselines) — contradicts "keep the harness stable" contract.
2. **Model coverage gaps**: top TPS models on the operator host are MoE (LFM2.5-8B-A1B) and MTP-packaged (gemma-4-E4B, Qwen3.5-MTP). Gemma-4 E2B/E4B is explicitly unsupported; MTP-head behavior on Qwen3.5 unverified. ExLlamaV3's win condition (dense Llama/Qwen EXL3 4bpw) is exactly the class that's *slower* than MoE/MTP on the operator host under llama.cpp.
3. **Context regime mismatch**: this repo's Pareto runs 65k–131k ctx (KV-heavy). ExLlamaV3 targets shorter-context consumer chat; KV quant helps but no evidence at 131k on 8 GB.
4. **Harness contract**: benchmark_search/validation run llama-server only; ExLlamaV3 (TabbyAPI/OpenAI server) is a different process, no `config.py` Baseline path, would violate "no ad-hoc eval" until a harness adapter exists (not requested).
5. **Windows build burden**: CUDA toolkit + VS Build Tools + flash-attn wheel + triton-windows for a JIT-compiled torch extension — vs llama.cpp prebuilt release (download, no build).

## 4. Why the others lose on the operator host

- **TensorRT-LLM**: NVIDIA shipped Windows support Oct 2023 with "up to 4x faster on RTX" ([blog](https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/)) targeting Llama 2/Code Llama era, via a dedicated Windows repo that is now gone (API 404; current [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) tree has no Windows path). Per-model engine compilation + stale arch coverage (no Qwen3.5/LFM2.5/Gemma-4) + WSL2-only today = no.
- **vLLM**: no Windows support in README; Linux/WSL2; server-engine overhead; on 8 GB single-user it optimizes concurrency, not single-stream TPS ([vLLM README](https://github.com/vllm-project/vllm)).
- **SGLang**: no Windows (issues #2249/#7766); see [sglang-inference-engine.md](./sglang-inference-engine.md).
- **koboldcpp / Ollama**: llama.cpp cores (same kernels) — identical TPS ceiling, extra wrapper.
- **llama.cpp Vulkan**: portable backend; on NVIDIA, the CUDA backend is the optimized path (llama.cpp backend list: CUDA for NVIDIA, Vulkan generic) — no reason to switch on the operator host.
- **OpenVINO**: Intel CPU/iGPU play; operator CPU has no iGPU, CPU-only INT8 won't beat CUDA on a 8GB-class ([openvino-genai-cpu-igpu-guide.md](./openvino-genai-cpu-igpu-guide.md) covers the AMD/Intel-CPU story).
- **Colibrì**: different goal (700B-class MoE on RAM), not TPS ([colibri-inference-engine.md](./colibri-inference-engine.md)).

## 5. Faster TPS *without* switching engines

llama.cpp already owns the operator host's speed levers (all repo-measured, `config.py`-only):

1. **Small-active MoE** (LFM2.5-8B-A1B: 178.5 TPS) — the biggest lever.
2. **MTP speculative heads** (+48% Qwen3.5-9B, +80% gemma-4-E4B via external draft) — [mtp-baseline-guide.md](./mtp-baseline-guide.md), [speculative-decoding-formats.md](./speculative-decoding-formats.md).
3. **KV q4_0 + smaller ctx** — KV is the 8 GB bottleneck at 131k; every ctx cut raises TPS.
4. **MoE `--n-cpu-moe`** for big MoE (POCKET-35B etc.) — [inference-engines-landscape.md](./inference-engines-landscape.md) / vitriol note.
5. **CUDA build freshness** — vendored submodule at b10173; updating the pinned llama.cpp commit picks up new CUDA kernels (worth a targeted re-bench on the Pareto set when upstream lands kernel work; keep `results.tsv` comparison fair).

## 6. Verdict

**Upstream llama.cpp CUDA remains the fastest *practical* TPS engine on the operator host.** It holds the measured records (MoE 178.5, MTP +80%), natively supports Windows + the GGUF store, and all remaining speed lives in model/flag choice, not engine choice.

**ExLlamaV3 (EXL3)** is the only credible raw-TPS challenger that runs on Windows NVIDIA, but it targets the dense-short-context niche the operator host already beats via MoE/MTP, breaks the GGUF store and harness contracts, and has **zero measured 8GB-class evidence** (web search, §7). Track it in the landscape; if a future need is "maximum single-stream TPS for one dense Qwen3.5-class model at ≤16k ctx," measure ExLlamaV3 vs llama.cpp on *the operator host* before switching (repo rule: measure, never estimate; no eval-score floors).

## 7. Web-search addendum (external benchmarks)

Cross-engine TPS numbers found in web sources, with fidelity flags. None overturn the repo-measured llama.cpp baseline on the operator host — and several are from old builds/formats.

### 7.1 8 GB-class discrete NVIDIA llama.cpp CUDA (lab data, primary)

[Puget Systems](https://www.pugetsystems.com/labs/articles/llm-inference-consumer-gpu-performance/), llama.cpp **build b3140** (2024-04), CUDA 12.2, Phi-3-mini-4k-instruct **Q4 GGUF**, pp 512 / tg 128:

| GPU | TG (t/s) | PP (t/s) |
| :-- | --: | --: |
| 8 GB-class discrete NVIDIA | **15.11** | 272 |
| 8 GB-class discrete NVIDIA Ti | 22.06 | 288 |
| RTX 3080 Ti | ~31 | >4070 SUPER |
| RTX 2080 Ti | ≈8GB-class (bandwidth-bound) | — |

Caveats vs this repo: b3140 predates years of CUDA kernel work (repo runs b10173); Phi-3-mini is dense 3.8B, not the small-active MoE/MTP class that produces the operator host's records (LFM2.5-8B-A1B 178.5 t/s). Takeaways: (a) 8GB-class prompt processing is strong (4th-gen tensor cores, 272 t/s pp even on b3140); (b) generation on dense models ≈ bandwidth-bound — 2080 Ti ≈ 8GB-class — which is exactly why small-active MoE wins here.

### 7.2 Official NVIDIA reference (primary)

NVIDIA developer blog (2024-10-02): llama.cpp, Llama 3 8B, ~**150 t/s on RTX 4090** (seq 100/100) — [link](https://developer.nvidia.com/blog). Scaling reference: 4090 is ~3.3x the 8GB-class's bandwidth/compute; dense-8B TPS on the operator host (Qwythos-9B 41–42 @ 100k ctx) is consistent with that ratio.

### 7.3 Cross-engine claims (secondary, EXL2-era — flag)

- insiderllm: ExLlamaV2 50–85% faster than llama.cpp on RTX 3090/4090 (EXL2 quants). aliteq: "~2x faster" with EXL2, GPU-only, no CPU offload. Both describe the **archived ExLlamaV2**, not ExLlamaV3.
- Reddit (r/LocalLLaMA): on older GPUs "no speed gains to expect using exllama2" — llama.cpp flash-attention optimizations closed the gap; advantage is GPU-generation dependent.
- ExLlamaV3 vendor: Marlin-style GEMM memory-bound at 4bpw **on 4090**; Ampere efficiency "needs work"; **Ada (8GB-class, sm89) unverified**.
- A LinkedIn post benchmarking llama.cpp vs ExLlamaV3 on Qwen3.6-27B @ 4.5 bpw surfaced in search but the page 404'd on retrieval — anecdotal, not citable.

**Net:** every ExLlama speed claim is either EXL2-era (archived engine), 4090/3090-class, or unverified on Ada. No web source shows ExLlamaV3 beating modern llama.cpp on a 8GB-class 8 GB with big context. The only way to know would be an on-rig measurement; per harness contract that means a config.py Baseline experiment through the search loop (llama.cpp) plus a manual ExLlamaV3 run outside it — not requested, so documented here instead.

## 8. Sources

- Rig: `scripts/check_hardware.py` output, 2026-08-02 (discrete 8 GB-class NVIDIA, operator CPU)
- Measured TPS: `results.tsv` (llama.cpp CUDA, upstream); [small-model-mtp-tps.md](./small-model-mtp-tps.md)
- ExLlamaV2 (archived): https://github.com/turboderp/ExLlamaV2 (README incl. 4090 TPS tables)
- ExLlamaV3: https://github.com/turboderp-org/exllamav3 (README: EXL3/QTIP, arch list, Windows build, no-ROCm)
- TensorRT-LLM Windows: https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/ (2023-10-17, "up to 4x"); https://github.com/NVIDIA/TensorRT-LLM (current tree, no Windows path)
- vLLM: https://github.com/vllm-project/vllm (README, no Windows)
- SGLang: [sglang-inference-engine.md](./sglang-inference-engine.md) (Windows issues #2249/#7766)
- llama.cpp: https://github.com/ggml-org/llama.cpp (backend list; vendored submodule b10173)
- Puget Systems 8GB-class llama.cpp lab bench (b3140, Phi-3-mini Q4): https://www.pugetsystems.com/labs/articles/llm-inference-consumer-gpu-performance/
- NVIDIA developer blog, llama.cpp 4090 ~150 t/s reference (2024-10): https://developer.nvidia.com/blog
- Secondary cross-engine claims (EXL2-era, flag): insiderllm.com, aliteq.com, r/LocalLLaMA threads — see §7.3
