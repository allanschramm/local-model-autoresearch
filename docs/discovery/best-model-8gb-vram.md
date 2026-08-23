# Best Model for 8 GB VRAM (Web-Sourced)

**Research date:** 2026-08-02. **Superseded by measurement (2026-08-23):** this guide's web-sourced picks are now measured on the rig — see [pareto-leaderboard.md](./pareto-leaderboard.md) and the `docs/sessions/2026-08-23-*` sweep logs. Measured winner: `Qwen3.8-4B-Q4_K_M.gguf` (`empero-ai/Qwen3.8-4B-Distill-GGUF`, min **0.6400** @131k) — a release newer than anything in the table below. Treat §1 ranks as historical candidate research only.

**Historical web-research note (2026-08-02):**

---

## 1. Recommendation

| Rank | Model | GGUF Q4_K_M | Arch (verified) | Key official scores (publisher-measured) | Context |
| :---: | :--- | :---: | :--- | :--- | :---: |
| **1** | **Qwen3.5-4B** | **2.74 GB** | dense hybrid: Gated DeltaNet + sparse attention (8/32 full-attn layers), 4B LM + vision encoder, MTP | MMLU-Pro **79.1**, GPQA-D **76.2**, LiveCodeBench v6 **55.8**, HMMT'25 **74.0–76.8**, TAU2-Bench **79.9**, IFEval **89.8** | 262,144 native |
| **2** | **LFM2.5-8B-A1B** | **5.16 GB** | hybrid MoE, 8.3B total / **1.5B active** (18 LIV conv + 6 GQA) | IFEval **91.84**, MATH500 **88.76**, AIME26 **50.0**, BFCLv3 **64.79** | 128,000 |
| **3** | **Gemma 4 E4B** | **5.34 GB** | dense, 4.5B effective / 8B w/ embeddings (PLE), sliding-window + global attention, text+image+audio | MMLU-Pro **69.4**, GPQA-D **58.6**, LiveCodeBench v6 **52.0**, AIME'26 **42.5** | 128K |
| **4** | **Qwen3-8B** | **5.03 GB** | dense 8.2B, GQA 32Q/8KV, thinking + non-thinking | scores in Qwen3 Tech Report (gap — see §5) | 32,768 native / 131,072 YaRN |

### Top pick: Qwen3.5-4B

The strongest verified official benchmark scores of any model whose full weights + KV cache fit in 8 GB physical VRAM. The hybrid Gated DeltaNet + sparse-attention architecture (only 8 of 32 layers are full attention) keeps the KV cache roughly ¼ of a same-size dense model, so the 262K-token native context is practically usable on 8 GB, and MTP adds multi-token prediction for generation speed.

**Caveat:** brand-new architecture (Feb 2026). Needs a recent llama.cpp with Gated DeltaNet + MTP support. For maximum ecosystem maturity today:

- **Fastest mature:** **LFM2.5-8B-A1B** (day-one llama.cpp support, official Liquid GGUF).
- **Safest smart + mature:** **Qwen3-8B** (5.03 GB, thinking modes, 128K context).

### Fit math (heuristic, flagged)

| Model | File @ Q4_K_M | KV headroom note |
| :--- | :---: | :--- |
| Qwen3.5-4B | 2.74 GB | ~5 GB left for KV; 8/32 full-attn layers ⇒ 32K–128K fits comfortably |
| LFM2.5-8B-A1B | 5.16 GB | 6 GQA layers of 24 ⇒ good headroom; full 128K needs reduced ctx or quantized KV |
| Qwen3-8B | 5.03 GB | 32K f16 KV (~4.7 GB) ⇒ total ≈ 9.7 GB **exceeds 8 GB**; use Q8_0 KV or drop ctx |

---

## 2. What does NOT fit (verified)

| Model | Reason | Source-verified size |
| :--- | :--- | :--- |
| Gemma 3 12B | 7.3 GB @ Q4_K_M + any KV > 8 GB | bartowski/LM Studio card |
| gpt-oss-20b | 11.6 GB @ Q4_K_M; OpenAI states native MXFP4 "runs within 16 GB of memory" | OpenAI card + Unsloth |
| Qwen3-30B-A3B | 18.6 GB @ Q4_K_M; even full expert offload needs ~19 GB+ system RAM, PCIe-bound | Unsloth |
| Mistral Small 3.1/3.x (24B) | fits "a single RTX 4090 or 32 GB MacBook once quantized" — not an 8 GB card | Mistral card |
| Qwen3-14B | **dense** (14.77B), Q4_K_M = **9.0 GB** — exceeds 8 GB; only sub-8 GB is UD-IQ1/Q2. No "14B-A3B" variant exists (verified via `hf` CLI: card + quant list) | Unsloth file list |

MoE models larger than ~10 GB total are only viable via `--n-cpu-moe` expert offload to CPU RAM, which needs the matching system RAM and becomes PCIe-bound — viable for gpt-oss-20b (smartest thing that *could* run with aggressive offload), excluded as primary pick.

---

## 3. Dense 7–9B candidate notes

- **Qwen3-8B** — 8.2B, 36 layers, 32/8 GQA, 32K/131K ctx. No inline benchmark table on card (numbers in Qwen3 Tech Report PDF, not extracted) — see gap.
- **Llama 3.1 8B** — 128K ctx, GQA. Scores (Meta card): MMLU 69.4, GPQA 30.4, HumanEval 72.6, MBPP 72.8, IFEval 80.4. Outclassed by 2025/26 releases on every axis.
- **Gemma 3 4B** — 2.49 GB @ Q4_K_M, fits; PT scores well behind 2026 small models (MMLU 59.6, HumanEval 36.0, GPQA 15.0).
- **GLM-4-9B / GLM-Z1-9B-0414** — card claims "top-ranked … same size" but publishes no 9B table and no 9B GGUF size; Q4_K_M ≈ 5.6 GB is secondary estimate only. Not shortlisted on primary evidence.
- **Phi-4-mini-instruct** — 3.8B, dense, 128K. Scores (Microsoft card): MMLU 67.3, MMLU-Pro 52.8, GPQA 25.2, GSM8K 88.6, MATH 64.0. Outclassed by Qwen3.5-4B. Tested on A100/A6000/H100.
- **DeepSeek-R1-Distill-Qwen-7B** — 4.68 GB @ Q4_K_M, fits. AIME'24 55.5, MATH-500 92.8, GPQA-D 49.1. Pure-reasoning distill: long CoT ⇒ slow wall-clock, poor general assistant. Not shortlisted for "fastest".

## 4. MoE candidate notes

- **LFM2.5-8B-A1B** — official Liquid GGUF, 1.5B active. Best-in-class instruction following (IFEval 91.84) and strong agentic (BFCLv3 64.79); slightly behind Qwen3.5-4B on math/knowledge. "Fastest in its size class" speed claim measured on a single H100 (18.5K tok/s) — **server hardware, not 8 GB**.
- **Gemma 3n E4B** — 8B raw / 4B effective (MatFormer selective activation, "memory footprint comparable to a traditional 4B model"), 4.24 GB text @ Q4_K_M, fits. Reasoning scores well below 2026 small models (GPQA-D 23.7, AIME'25 11.6). Multimodal needs separate mmproj.
- **gpt-oss-20b** — 21B / 3.6B active, native MXFP4 MoE, harmony format. SWE-bench Verified 37.4/53.2/60.7 (low/med/high reasoning), MMLU-Pro 73.6, GPQA-D 58.59. Too large for 8 GB VRAM alone (see §2).
- **Gemma 4 26B-A4B** (2026) — 25.2B / 3.8B active; no verified GGUF size; at ~15+ GB fails the 8 GB constraint anyway.
- **Granite-4.0-H-Tiny** — verified from IBM (primary, 2026-08-02): decoder-only hybrid MoE (Mamba-2 + Transformer, shared experts, GQA, SwiGLU, RMSNorm, shared embd/out embeddings), **7B total / 1B active**, context **128k**, **Apache-2.0**. (4 attention + 36 Mamba-2 layers / 64 experts / 6 active per the official GGUF repo README.)

---

## 5. Speed claims: what is (and is not) an 8 GB measurement

- **No publisher publishes a tokens/s figure measured on a desktop 8 GB discrete GPU (e.g. 8 GB-class discrete NVIDIA)** for any of these models. Treat any local-GPU tps from secondary write-ups as unverified.
- LFM2.5-8B-A1B "18.5K tokens/s" = single **H100**, high concurrency — signals fast architecture, not an 8 GB number.
- Mistral Small 3.1 ships only a *memory* statement, not speed.
- Phi-4-mini trained/evaluated on A100/A6000/H100; no local speed claim.
- Decode speed on 8 GB scales with bytes-per-token + VRAM bandwidth. Heuristic ordering (estimate, not measured): Qwen3.5-4B (2.74 GB, hybrid attention) ≈ LFM2.5-8B-A1B (1.5B active) > Qwen3-8B / Gemma 4 E4B (~5 GB) > reasoning models (same decode speed, huge CoT overhead).

**Partially closed (2026-08-02):** NVIDIA publishes **18 tokens/s** for Nemotron 3 Nano 4B Q4_K_M (llama.cpp) on a **Jetson Orin Nano 8GB** (official blog + GGUF card). Caveat: Orin Nano's GPU is far weaker than an 8 GB-class discrete NVIDIA — it is an 8 GB-memory device figure, not an 8GB-class-discrete figure, and no candidate has a publisher-measured 8GB-class-discrete tps. The ordering above remains derived from verified file sizes + architectures, not published 8 GB-GPU figures.

---

## Sources / Verification

All URLs fetched **2026-08-02**. Primary sources; gap items are flagged in §5 and below.

| # | URL | Used for |
| :-: | :--- | :--- |
| 1 | https://huggingface.co/Qwen/Qwen3-8B | Qwen3-8B params, context, thinking modes |
| 2 | https://huggingface.co/unsloth/Qwen3-8B-GGUF | Q4_K_M = 5.03 GB |
| 3 | https://huggingface.co/Qwen/Qwen3-30B-A3B | MoE card: 30.5B/3.3B, 128 experts, ctx |
| 4 | https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF | Q4_K_M = 18.6 GB (too large) |
| 5 | https://huggingface.co/Qwen/Qwen3.5-4B | Qwen3.5-4B arch, ctx, all benchmark tables |
| 6 | https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | Q4_K_M = 2.74 GB |
| 7 | https://huggingface.co/google/gemma-3-4b-it | Gemma 3 4B params, ctx, scores |
| 8 | https://huggingface.co/google/gemma-3-12b-it | Gemma 3 12B params, ctx, scores |
| 9 | https://huggingface.co/lmstudio-community/gemma-3-4b-it-GGUF | Q4_K_M = 2.49 GB |
| 10 | https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF | Q4_K_M = 7.3 GB (fit fail) |
| 11 | https://huggingface.co/google/gemma-3n-E4B-it | Gemma 3n 8B/4B effective, scores |
| 12 | https://huggingface.co/lmstudio-community/gemma-3n-E4B-it-text-GGUF | Q4_K_M = 4.24 GB |
| 13 | https://huggingface.co/google/gemma-4-E4B-it | Gemma 4 E4B 4.5B/8B, 128K, scores, Apache-2.0 |
| 14 | https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF | Q4_K_M = 5.34 GB |
| 15 | https://huggingface.co/LiquidAI/LFM2.5-8B-A1B | Liquid card: 8.3B/1.5B, ctx, scores, H100 speed claim |
| 16 | https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF | official Q4_K_M = 5.16 GB |
| 17 | https://huggingface.co/openai/gpt-oss-20b | 21B/3.6B, MXFP4, 16 GB memory, eval results |
| 18 | https://huggingface.co/unsloth/gpt-oss-20b-GGUF | Q4_K_M = 11.6 GB |
| 19 | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | scores table (AIME 55.5, MATH-500 92.8…) |
| 20 | https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF | Q4_K_M = 4.68 GB |
| 21 | https://huggingface.co/zai-org/GLM-4-9B-0414 | Z.ai card (32B-series tables; 9B claim only) |
| 22 | https://huggingface.co/microsoft/Phi-4-mini-instruct | 3.8B, 128K, benchmark table, test hardware |
| 23 | https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503 | 24B, 128K, 4090/32 GB statement, scores |
| 24 | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct | 8B, 128K, benchmark tables |
| 25 | https://qwenlm.github.io/blog/qwen3/ | Qwen3 lineup: context, layers, experts, languages |
| 26 | https://arxiv.org/abs/2505.09388 | Qwen3 Technical Report (Qwen3-8B scores; PDF tables not extracted) |
| 27 | https://arxiv.org/abs/2508.10925 | gpt-oss-120b & 20b Model Card |
| 28 | https://arxiv.org/abs/2511.23404 | LFM2 Technical Report (referenced by Liquid card) |
| 29 | https://arxiv.org/abs/2607.02770 | Gemma 4 Technical Report (referenced by Gemma 4 card) |
| 30 | https://huggingface.co/Qwen/Qwen3-14B | dense 14.77B; no "A3B" variant exists (verified via `hf` CLI) |
| 31 | https://huggingface.co/unsloth/Qwen3-14B-GGUF | file list: Q4_K_M = 9.0 GB (fit fail); only UD-IQ1/Q2 fit |
| 32 | https://huggingface.co/ibm-granite/granite-4.0-h-tiny | Granite 4.0 H Tiny arch/ctx/license (primary) |
| 33 | https://huggingface.co/blog/nvidia/nemotron-3-nano-4b | 18 tokens/s on Jetson Orin Nano 8GB (8 GB-memory device, not 8 GB-class discrete NVIDIA) |

**Access note:** Hugging Face returned HTTP 401 for several repos this session (some bartowski repos) — likely anti-scraping/rate-limiting, not actual gating (Qwen3 is Apache-2.0). An earlier "Qwen3-14B-A3B" 401 was a **non-existent repo** — the model is dense Qwen3-14B — resolved via the `hf` CLI (source rows 30–31).

### Verification gaps

1. Qwen3-8B official benchmark numbers — card has no tables; blog renders tables as chart images; authoritative numbers in Qwen3 Tech Report PDF (cited, not extracted).
2. ~~Qwen3-14B-A3B~~ — **resolved:** no such repo; the real model is dense Qwen3-14B (Q4_K_M 9.0 GB, fit fail). Corrected via `hf` CLI.
3. GLM-4-9B / GLM-Z1-9B-0414 GGUF size and 9B tables — not published on card; size would be secondary estimate.
4. Phi-4-mini and Llama 3.1 8B GGUF sizes — not fetched from a quant provider this session (secondary estimates ~2.4 GB / ~4.9 GB).
5. Gemma 4 26B-A4B GGUF size — not verified; likely too large for 8 GB anyway.
6. ~~Granite-4.0-H-Tiny~~ — **resolved:** verified from IBM primary card (7B/1B hybrid MoE, 128k, Apache-2.0); no 8 GB-GPU tps exists for it, only the Jetson Orin Nano 18 t/s figure (§5, flagged as not 8 GB-class discrete NVIDIA).
