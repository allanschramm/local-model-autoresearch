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

ExLlamaV2 is **archived** ("This project is archived for now. Development continues on ExLlamaV3") ([ExLlamaV2 README](https://github.com/turboderp/ExLlamaV2)). ExLlamaV3 ([repo](https://github.com/turboderp-org/exllamav3) — active, **v1.4.5 published 2026-08-31** ships `win_amd64` prebuilt wheels `cp310–cp313`, `cu128`/`torch2.10.0` on [releases](https://github.com/turboderp-org/exllamav3/releases) — verified via GitHub API 2026-08-31) is the live project: "optimized quantization and inference library for running LLMs locally on modern consumer-class GPUs."

Headline facts from the [ExLlamaV3 README](https://github.com/turboderp-org/exllamav3):
- **EXL3 format**: streamlined variant of **QTIP** (Cornell RelaxML); one-step quantization (fused Viterbi kernel), minutes for small models on one RTX 4090; coherent at 1.6 bpw (Llama-3.1-70B in <16 GB). TPS-relevant: **Marlin-inspired GEMM kernel ≈ memory-bound at 4 bpw on RTX 4090**; Ampere efficiency addressed in v1.0.0 (2026-07-14 release body: "Greatly improved GEMM/GEMV performance on Ampere", also removed `flash-attn-2`/`xformers` deps — verified 2026-08-31); Ada sm_89: community sm_89 CUDA extension builds exercised in the wild (PR #303, §7.3) — still unverified on this rig.
- **Tensor-parallel + expert-parallel** for consumer setups; 2–8-bit KV cache quantization (helps long ctx on 8 GB); speculative decoding; LoRA; multimodal.
- **Windows (updated 2026-08-31):** **v1.4.5** (2026-08-31) ships `win_amd64` wheels for `cp310–cp313`, `cu128`/`torch2.10.0` on [releases](https://github.com/turboderp-org/exllamav3/releases) (GitHub API, 2026-08-31). Caveat: torch/CUDA pin must match host env; PyPI path still requires VS Build Tools per README. Recent release notes of note: v1.4.3 partial CPU layer expert offload + auto-calibrated dynamic-draft thresholds; v1.4.4 vision tower quantization + vision offload from system RAM ("performance penalty is small"); v1.4.5 improved MoE MTP performance.
- **NVIDIA-only** ("ROCm support" on to-do list).
  - Arch list (vs this repo's inventory, [README](https://github.com/turboderp-org/exllamav3), re-checked 2026-08-31 via direct README fetch):
  - ✅ **LFM 2.5** (`Lfm2MoeForCausalLM`) — the 178-TPS king *could* run on ExLlamaV3
  - ✅ **Qwen 3.5 / Qwen 3.5 MoE** — dense 4B/9B supported; **embedded MTP works** (resolved 2026-08-31: PR #303 scope is "Qwen3.5/3.6 embedded MTP"; issue #260 runs Qwen3.6-27B EXL3 5bpw with embedded MTP tensors, ~2.9–3.1× accept speedup; TabbyAPI `draft_mode: mtp`) — external / unverified on this rig; v1.2.0 GDN state-rewind bug affects multi-GPU splits only, single-GPU unaffected (issue #260)
  - ✅ **Laguna 2.1** (`LagunaForCausalLM`) — Laguna-XS
  - ✅ **Gemma 4** (dense 31B/12B etc.: `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration`) — **now supported** as of 0.0.29+ (see [HaoweiShen/Gemma-4-31B-it-EXL3](https://huggingface.co/HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw) EXL3 pack; the earlier "Gemma 4 not supported" for this repo's era was accurate at 2026-07-31 but is stale for the family). Caveat: a `layer_scalar` inference bug required a patched fork for correct 31B inference (fixed upstream after); verify current wheel before use.
  - ❌ **Gemma 4 E2B/E4B specifically still not supported** — the E2B/E4B efficient variants remain listed as unsupported (kills the 122-TPS MTP champion on this rig) — confirmed via `web_search` 2026-08-22.
  - ❓ Ornith-1.0-9B, Qwythos-9B, POCKET-35B, KAT-Coder — archs not in the supported list; must check HF config per model before assuming anything

Historical TPS context (ExLlamaV2-era, **4090**, primary README tables): Llama-7B EXL2 4.0 bpw **211 t/s**, TinyLlama-1.1B EXL2 4.0 bpw 700 t/s. Directional secondary sources claim ExLlamaV2 was 50–85% faster than llama.cpp on 3090/4090 (blog aggregator; ExLlamaV2-era; not primary, treat as directional). **No primary 8GB-class-class ExLlamaV3-vs-llama.cpp TPS benchmark exists in the sources gathered** — any speedup on this specific rig is unverified until measured.

### 3.1 Qwen3.5-9B-exl3 — trigger artifact exists (added 2026-08-31)

[turboderp/Qwen3.5-9B-exl3](https://huggingface.co/turboderp/Qwen3.5-9B-exl3) is the exact §6 trigger case: a dense Qwen3.5-class EXL3 pack published by the format author. Requires ExLlamaV3 ≥ v0.0.23 (any current 1.4.x satisfies it); branches 2.00–6.00 bpw; KLD-vs-bpw chart on the card.

Measured branch sizes (HF API tree, 2026-08-31; single `model.safetensors` per branch) vs the Baseline `VRAM_LIMIT_MB` keepout (7676 MB class, ±200 MB WDDM variance):

| branch | size | fit (est., before KV + runtime) |
| :-- | --: | :-- |
| 3.00bpw | 5.89 GiB | plausible fit at moderate ctx — hybrid arch has 8/32 full-attn layers (small KV) + EXL3 2–8-bit cache quant |
| 3.50bpw | 6.29 GiB | borderline |
| 4.00bpw | 6.69 GiB | over — only with vision offload (v1.4.4 streams vision tower from system RAM) and/or small ctx |

Δ(3.00→4.00bpw) = 0.80 GiB → ~3.5 GiB of the pack does not scale with bpw (vision tower + fixed-precision parts; per-tensor bitrates in `quantization_config.json` not decomposed). Pack is multimodal (`Qwen3_5ForConditionalGeneration`).

Repo-measured llama.cpp targets to beat ([model card](../models/qwen3.5-9b.md)): base **38.7** / MTP **57.3** t/s (llama-cli fair matrix, 2026-07-20), bench_tg **67.5** @ 32k+MTP (Q4_K_M).

ExLlamaV3-side levers: embedded-MTP draft (`draft_mode: mtp` in TabbyAPI; v1.4.3 auto-calibrated dynamic-draft thresholds) — works only on packs that keep the MTP head (see measured block below); cache quantization; vision offload.

**Falsifiable experiment** (manual pass outside the harness — benchmark/validation are llama-server-only; never autoloop):
1. Fresh venv, not the harness venv: `torch 2.10.0+cu128`, then the exllamav3 `win_amd64` wheel (v1.4.5) matching the venv's cp version (cp310–cp313).
2. Download the 3.00bpw branch (~6.3 GB disk).
3. Single-stream tg at fixed ctx (16k/32k), MTP off then `mtp` draft on, batch 1, fixed prompt; nvidia-smi peak vs the keepout ceiling. Cache: EXL3 quantized (2–8 bit) at 32k to match the llama.cpp q4_0-KV comparison style — fp16 KV at 131k does not fit 3.50+ bpw. Tooling: no standalone bench script upstream (examples listing checked 2026-08-31) — measure via `examples/chat.py` / a `generator.py`-style loop, or TabbyAPI for the server path.
4. Win condition: EXL3-MTP > 57.3 t/s at the same ctx class → then consider a wider comparison; otherwise llama.cpp stays.

**Measured on the operator host (2026-08-31; ExLlamaV3 v1.4.5+cu128.torch2.10.0 wheel, triton 3.8.0, dedicated Python 3.12 venv outside the harness):**

| config | decode t/s | peak VRAM | notes |
| :-- | --: | --: | :-- |
| EXL3 3.00bpw, base (no MTP) | **49.6** | **5437 MB** | 511 tokens greedy, batch 1, 32k q4/q4 cache |
| llama.cpp Q4_K_M base | 38.7 | — | fair matrix, 2026-07-20 |
| llama.cpp Q4_K_M + MTP | **57.3** | — | fair matrix, 2026-07-20 (bench_tg 67.5 @ 32k) |

**EXL3 base is +28% over llama.cpp base but −13% vs llama.cpp-with-MTP — llama.cpp stays.** The MTP-on leg is impossible on this pack: the safetensors header (1363 tensors audited) contains **zero `mtp`/`nextn` tensors** — turboderp's EXL3 conversion dropped the embedded MTP head, and `Model.from_config(config, component="mtp")` fails at load (`mtp.pre_fc_norm_hidden.weight not found`). EXL3-embedded-MTP works only on packs that keep the head (e.g. Qwen3.6-27B EXL3 5bpw, issue #260). **Fair-comparison follow-up (operator-directed, 2026-08-31): attempted, hardware-blocked.** Self-quantizing from the BF16 source with the head kept is mechanically possible — convert.py v1.4.5 wires the MTP component automatically for qwen3_5 (source-verified; a completed run produced a 1399-tensor pack with **39 MTP tensors** at 3 bpw text / 4 bpw MTP / unquantized head, zero vision) — but two hardware walls block the fair benchmark on this rig: (1) **quantizing the 248K-vocab lm_head exceeds 32 GB host RAM on every head path** (6 bpw mul1 Viterbi: 55 GB commit, frozen at 0 % for 45 min; 8 bpw retry: free RAM 24 → 0.8 GB in ~2 min, killed per the circuit-breaker protocol — the quantizer's head-stage state scales with vocab size); (2) **storing the head unquantized (`-hb 16`) doesn't fit 8 GB inference** — the pack is 7.18 GiB (bf16 head 2.03 + bf16 embed 2.03 + 3 bpw layers ≈ 3.2) and autosplit load fails with "Insufficient VRAM in split for model and cache" (turboderp's pack only fits because its vision tower is skipped at text-gen and its head is 4-bpw-quantized — the exact quantization blocked by (1)). exl3.md's "embedding layer can be relegated to system RAM" is not exposed in v1.4.5 — the `Model.load` signature has no embedding-offload (only MoE CPU-offload exists, `model_ls.py`) and convert.py has no embedding-bits option, so the residual lever is upstream.
**Ecosystem scan (2026-08-31, `hf models ls` across authors):** the only MTP-bearing EXL3 pack for the Qwen3.5 family on HF is `komeijishiki/DeepSeek-V4-Pro-Qwen3.5-9B-EXL3-6.50bpw-H8-V8-MTP8` (created 2026-09-01T01:08Z, unsloth finetune base). Its H8 recipe proves 8-bit head quant works on bigger-RAM hosts, but the pack totals **9.6 GB** — ~3 GB over this rig's placement ceiling. No ≤4 bpw base-model EXL3 pack with an MTP head exists yet; embedding quantization is not a converter option (embeddings are hard-coded 16-bit copies), and v1.4.5's `Model.load` exposes no embedding-offload (only MoE CPU-offload exists in `model_ls.py`).

Setup gotchas for any future pass: (1) `triton-windows` is **mandatory** on Windows — v1.4.5 `bc_dsa.py` imports triton kernels unconditionally, so without it the package fails at import (README says "suboptimal"; reality is "won't import"); (2) the fit estimate for 3.00bpw is now measured — 5437 MB peak incl. torch context, big headroom under the keepout; (3) load took 35.9s first run (triton JIT compile), 3.4s with warm JIT cache; (4) the PyPI/release **wheel omits `standard_cal_data/`** (c4/code/multilingual/technical/tiny/wiki .utf8) that convert.py expects — fetch from the GitHub tag into the installed package, else calibration fails; (5) EXL3 packs keep embeddings at 16 bits (they are not in the quant walk) — budget ~2 GiB per 248K-vocab embed when fitting.

### Why not adopt now (repo-relevant)

1. **Format lockout**: EXL3 is a new quant format; the repo's entire GGUF store + `docs/models/*` cards + Pareto `results.tsv` lineage are GGUF. Every candidate model must be re-quantized from HF safetensors (new format, new quality baselines) — contradicts "keep the harness stable" contract.
2. **Model coverage gaps**: top TPS models on the operator host are MoE (LFM2.5-8B-A1B) and MTP-packaged (gemma-4-E4B, Qwen3.5-MTP). Gemma-4 E2B/E4B is explicitly unsupported; on this rig the turboderp Qwen3.5-9B EXL3 pack carries no MTP head, so EXL3 runs base-only and loses to llama.cpp-with-MTP (§3.1, measured). ExLlamaV3's win condition (dense Llama/Qwen EXL3 4bpw) is exactly the class that's *slower* than MoE/MTP on the operator host under llama.cpp.
3. **Context regime mismatch**: this repo's Pareto runs 65k–131k ctx (KV-heavy). ExLlamaV3 targets shorter-context consumer chat; KV quant helps but no evidence at 131k on 8 GB.
4. **Harness contract**: benchmark_search/validation run llama-server only; ExLlamaV3 (TabbyAPI/OpenAI server) is a different process, no `config.py` Baseline path, would violate "no ad-hoc eval" until a harness adapter exists (not requested).
5. **Windows build burden (softened 2026-08-31):** prebuilt `win_amd64` wheels now exist for `cu128`/`torch2.10.0` (`cp310–cp313`, v1.4.5), so source-build with CUDA toolkit + VS Build Tools + `triton-windows` is fallback, not the only path. Remaining pin: wheel's torch/CUDA must match host env vs llama.cpp prebuilt release (download, no build, no torch pin).

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

**ExLlamaV3 (EXL3)** is the only credible raw-TPS challenger that runs on Windows NVIDIA, but it targets the dense-short-context niche the operator host already beats via MoE/MTP, breaks the GGUF store and harness contracts. **Measured on the operator host (2026-08-31, §3.1): EXL3 3.00bpw base 49.6 t/s vs llama.cpp base 38.7 (+28%) but llama.cpp-with-MTP 57.3 (−13%) — and the EXL3 pack ships without the MTP head, so the engine swap loses on the one model class it targeted. The self-quant path to a fair EXL3-MTP number is hardware-blocked (head quant needs >32 GB RAM at 248K vocab; unquantized head doesn't fit 8 GB VRAM, §3.1). Verdict unchanged: llama.cpp stays.** Re-test only if (a) a Qwen3.5-9B EXL3 pack appears with the MTP head kept, (b) ExLlamaV3 exposes embedding offload making a bf16-head self-quant fit on 8 GB, or (c) a dense-model need arises where MTP is unavailable.

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
- r/LocalLLaMA (Aug 2026, [thread 1vqh5s7](https://www.reddit.com/r/LocalLLaMA/comments/1vqh5s7/exllamav3_vs_llamacpp_2x_3060/)): ExLlamaV3 vs llama.cpp on 2× RTX 3060 TP — Qwen3.8-27B EXL3 4.0bpw ~95 vs ~40 t/s; Qwen3.6-35B-A3B ~150 vs ~84 t/s. **External / unverified on this rig; multi-GPU TP setup; numbers from search-result summary — direct fetch blocked (403).**
- ExLlamaV3 PR #303 (2026-08-22, open, `dev`): Qwen3.8-27B EXL3 3.00bpw MTP-4, full 248K-vocab head **36.58 t/s** mean on an RTX 4060 Ti 16 GB with a CUDA 12.8 extension built for SM 8.9 (selected-head variant 44.58 t/s). **External / unverified on this rig** (16 GB card, 27B model, unmerged PR) — but proof that Qwen3.5-family embedded MTP and sm_89 builds run in the wild.

**Net:** every ExLlama speed claim is either EXL2-era (archived engine), 4090/3090-class, or unverified on Ada. No web source shows ExLlamaV3 beating modern llama.cpp on a 8GB-class 8 GB with big context. The only way to know would be an on-rig measurement; per harness contract that means a config.py Baseline experiment through the search loop (llama.cpp) plus a manual ExLlamaV3 run outside it — not requested, so documented here instead.

## 8. Sources

- Rig: `scripts/check_hardware.py` output, 2026-08-02 (discrete 8 GB-class NVIDIA, operator CPU)
- Measured TPS: `results.tsv` (llama.cpp CUDA, upstream); [small-model-mtp-tps.md](./small-model-mtp-tps.md)
- ExLlamaV2 (archived): https://github.com/turboderp/ExLlamaV2 (README incl. 4090 TPS tables)
- ExLlamaV3: https://github.com/turboderp-org/exllamav3 (README: EXL3/QTIP, arch list, Windows build, no-ROCm)
- ExLlamaV3 v1.0.0 release (2026-07-14; Ampere GEMM/GEMV, attention-kernel rewrite): https://github.com/turboderp-org/exllamav3/releases/tag/v1.0.0
- ExLlamaV3 v1.4.3–v1.4.5 release notes (CPU expert offload, vision offload/quant, MoE MTP perf): https://github.com/turboderp-org/exllamav3/releases
- PR #303 — Qwen MTP hot-vocab head (Qwen3.5/3.6 embedded-MTP scope; 4060 Ti sm_89 measurement): https://github.com/turboderp-org/exllamav3/pull/303
- Issue #260 — GDN rewind + EXL3-embedded MTP tensors (Qwen3.6-27B EXL3 5bpw): https://github.com/turboderp-org/exllamav3/issues/260
- Qwen3.5-9B-exl3 pack: https://huggingface.co/turboderp/Qwen3.5-9B-exl3 (branch sizes via HF API tree, 2026-08-31)
- TensorRT-LLM Windows: https://blogs.nvidia.com/blog/2023/10/17/tensorrt-llm-windows-stable-diffusion-rtx/ (2023-10-17, "up to 4x"); https://github.com/NVIDIA/TensorRT-LLM (current tree, no Windows path)
- vLLM: https://github.com/vllm-project/vllm (README, no Windows)
- SGLang: [sglang-inference-engine.md](./sglang-inference-engine.md) (Windows issues #2249/#7766)
- llama.cpp: https://github.com/ggml-org/llama.cpp (backend list; vendored submodule b10173)
- Puget Systems 8GB-class llama.cpp lab bench (b3140, Phi-3-mini Q4): https://www.pugetsystems.com/labs/articles/llm-inference-consumer-gpu-performance/
- NVIDIA developer blog, llama.cpp 4090 ~150 t/s reference (2024-10): https://developer.nvidia.com/blog
- Secondary cross-engine claims (EXL2-era, flag): insiderllm.com, aliteq.com, r/LocalLLaMA threads — see §7.3
