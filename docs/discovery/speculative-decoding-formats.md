# Speculative Decoding Formats Guide (llama.cpp)

This guide documents the speculative decoding formats supported by `llama.cpp` (`llama-cli`/`llama-server`), their architectural requirements, and their availability for the models in this repository.

---

## 1. Overview of Speculative Decoding Formats

Speculative decoding speeds up inference by using a fast **drafting method** to propose a block of tokens, which the larger **target model** then validates in a single forward pass. 

In `llama.cpp`, the format is specified using the `--spec-type` flag. These formats are **not** software configurations that can be applied to any model; they must match the **mathematical and architectural format** of the draft model file (`-md` / `--spec-draft-model`).

| Format (`--spec-type`) | Description | Drafter Type | VRAM Cost | Availability for our Models |
| :--- | :--- | :--- | :---: | :--- |
| **`draft-mtp`** | Multi-Token Prediction (MTP) | Neural (Assistant) | Medium | **High** (Native for Qwen3.5/3.6 and Gemma-4. Pre-trained drafts available). |
| **`draft-eagle3`** | Eagle 3 tree-based drafting | Neural (Eagle Head) | Medium | **None** (No pre-trained Eagle drafts exist for Gemma-4/Qwen). |
| **`draft-dflash`** / **`dspark`** | DeepSeek parallel block drafters | Neural (DeepSpec) | Medium | **Dead end on 8 GB-class MoE+`n-cpu-moe`.** Qwen3.6-35B DFlash loads but is slower than no-spec; needs the 35B target fully on GPU. Embedded `draft-mtp` can still win. Evidence: [2026-08-12](../sessions/2026-08-12-qwen36-dflash-tps.md), [2026-08-07](../sessions/2026-08-07-qwen36-35b-dflash-tps.md). Native `draft-dspark` is in upstream since 2026-07-28 ([#25173](https://github.com/ggml-org/llama.cpp/pull/25173), i.e. inside pinned b10375) for **Qwen3-backbone drafts** only; Gemma-4-backbone drafts stay SGLang-only. |
| **`ngram-cache`** | N-gram → next-token statistics (3-level cache: context, dynamic, static) | Statistical | None (0 MB VRAM; host RAM grows with context) | **Universal** (any model, no extra files). |
| **`ngram-simple`** | History match: last n-gram → following m-gram | Statistical | None (0 MB) | **Universal** (any model, no extra files). |
| **`ngram-map-k` / `ngram-map-k4v`** | Hash-map of n-grams in the context window; k4v tracks up to 4 value m-grams | Statistical | None (0 MB) | **Universal** (any model, no extra files). |
| **`ngram-mod`** | LCG rolling-hash pool: n-gram hash → next token, shared across server slots | Statistical | None (0 MB VRAM; ~16 MB host RAM, constant) | **Universal** (any model, no extra files). `--spec-default` enables it. |
| **`none`** | Standard autoregressive decoding | None | None | **Universal** (Speculative decoding disabled). |

---

## 2. Can MTP and other Speculative Decoding methods be used together?

**Partial — updated 2026-08-18.** `--spec-type` is a **comma-separated list**; upstream combines a draft-model implementation with a draftless one. The earlier "one spec-type only" claim in this guide was stale.

1.  **MTP is already a form of Speculative Decoding:**
    *   Speculative decoding is the general concept of using a fast drafting method to propose tokens and a target model to verify them.
    *   Multi-Token Prediction (MTP) is simply a *specific implementation* of this concept, where the "draft model" is built directly into the target model as auxiliary prediction heads (e.g. Qwen's MTP heads or Gemma-4 assistant models). When you enable MTP, you *are* using speculative decoding.
2.  **Mixing rules (upstream docs/speculative.md, master; MTP merge [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) 2026-05-16):**
    *   `--spec-type` accepts a comma-separated list, e.g. `--spec-type draft-mtp,ngram-mod` (also valid as repeated `--spec-type draft-mtp --spec-type ngram-mod` flags — `--spec-type` is exempt from the duplicate-argument warning).
    *   **An implementation with a draft model can be mixed with an implementation without one**: neural (MTP/DFlash/DSpark/Eagle) + statistical (ngram-*) is supported. The MTP PR documents exactly this combination (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`, or `--spec-default --spec-type draft-mtp`), labeled *experimental, suitable for non-CUDA systems*.
    *   **If a draft model is combined with a draftless decoding, the draftless decoding has higher precedence** (upstream docs).
    *   **Two neural drafters simultaneously is still not supported** (one draft model at a time — a single `-md`). Chaining (Eagle drafts, then MTP drafts those drafts) remains unsupported.
3.  **Choosing the Best Method:**
    *   Measure per Fingerprint. For Gemma-4 / Qwen3.5 dense on this 8 GB rig, **MTP is usually best when it actually accelerates**; Mythos MTP GGUF is a counterexample (~+1%). See [small-model-mtp-tps.md](./small-model-mtp-tps.md).
    *   A cheap composite worth probing on dense MTP models: `draft-mtp` + `ngram-mod` (ngram fills where the MTP head hesitates; 0 MB VRAM). On small-active MoE (A3B), external measurements say all spec modes lose — see §4b.

---

## 3. Deep-Dive into Formats

### A. Multi-Token Prediction (MTP) — `draft-mtp`
*   **How it works:** Native to Qwen2.5/3.5/3.6 and Gemma-4. Two packaging forms:
    1. **Embedded `nextn` heads** inside the main GGUF (Qwen UD MTP builds; community `*-MTP*.gguf`).
    2. **External assistant draft** (Gemma-4): Google trained `*-assistant` models; Unsloth ships tiny draft GGUFs. The **main** Gemma UD file does **not** contain `nextn` tensors.
*   **Why it's best for us:** When acceptance is healthy, MTP hits **~1.5x–1.8x** on this 8 GB rig (see §4 matrix). Not automatic — Mythos MTP GGUF measured ~+1%.
*   **In llama.cpp:**
    - Qwen3.5-9B UD: MTP **embedded** — `--spec-type draft-mtp --spec-draft-n-max N` only (no `-md`).
    - Gemma-4 E4B: pass `--spec-draft-model models/draft/mtp-gemma-4-E4B-it.gguf` (path relative to `models/` in harness: `draft/mtp-gemma-4-E4B-it.gguf`).
    - **Ornith-1.0-9B UD:** no MTP. Use Hub `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` (local: `Ornith-1.0-9B-MTP-Q4_K_M.gguf`).
    - **Mythos 5-1M:** Hub `mradermacher/Qwythos-9B-Claude-Mythos-5-1M-MTP-GGUF` (local MTP Q4_K_M). Loads; short-gen gain negligible in 2026-07-20 matrix.
    - **Qwythos-9B-v2:** no useful CUDA GGUF MTP on Hub (MLX-only) as of 2026-07-20.
*   **Detect:** metadata keys `nextn` / `blk.*.nextn.*` (embedded) or `gemma4-assistant` (draft). See [small-model-mtp-tps.md](./small-model-mtp-tps.md). **Caveat:** low-bit quants may drop the head — verify per-file. Example: `Qwen3.8-27B-Q1Q-XYZ-v2.gguf` declares `nextn_predict_layers` value 0 and zero `nextn.*` tensors despite pack docs claiming embedded MTP (2026-08-18 validation).
*   **Acceptance vs draft length (measured upstream + community):** acceptance is prompt- and head-quality-dependent, and it is the lever the `--spec-draft-n-max` sweep tunes.
    - Upstream MTP merge [#22673](https://github.com/ggml-org/llama.cpp/pull/22673) (Qwen3.6-27B q8_0, DGX Spark): `n_max=3` aggregate accept 0.72 @ ~2.6× vs baseline; `n_max=2` accept 0.83 but fewer tokens/pass (17.4 vs 21.6 t/s on code). Per-task accept ranged 0.54 (translation) to 0.91 (code) — the same target, same head, big spread.
    - Community rule of thumb (dev.to, 2026-05-18): acceptance below ~0.6 means MTP **costs** wall-clock; temperature >~0.7 and aggressively quantized MTP heads both depress acceptance ("the MTP head suffered more than the main weights").
    - Implication for this repo: Mythos 5-1M MTP (+1%) and the Qwen3.6-35B-A3B n=2 (+1%) cases are consistent with quant-damaged/low-yield heads or draft-length mismatch, not engine failure — the falsifiable probe is a `SPEC_DRAFT_N_MAX` sweep on the same Fingerprint (see [small-model-mtp-tps.md](./small-model-mtp-tps.md) open questions).

### B. Eagle 3 — `draft-eagle3`
*   **How it works:** A tree-based speculative decoder that trains a small recurrent neural network head directly on top of the target model's hidden states.
*   **Availability:** While Unsloth does not bundle Eagle drafts for our current targets, the community hosts Eagle-3 GGUF draft models for target models on Hugging Face (e.g. `Ex0bit/Qwen3.6-27B-PRISM-PRO-DQ`, `thoughtworks/Gemma-4-31B-Eagle3`, or `RedHatAI/gemma-4-31B-it-speculator.eagle3`). They must be paired using `--spec-type draft-eagle3` and `-md <path>`.
*   **Error Case:** If you try to pass an MTP draft model to `--spec-type draft-eagle3`, it will fail to load with:
    `failed to initialize speculative decoding context: draft model is not eagle3`

### C. DFlash & DSpark — `draft-dflash` (DFlash) / `dspark` (DSpark)
*   **How it works:** Open-source research released by DeepSeek (the **DeepSpec** framework).
    - **DFlash:** A block-parallel drafter utilizing a diffusion-like block process to predict blocks of tokens in a single step (e.g. `spiritbuun/Qwen3.6-27B-DFlash-GGUF` or `williamliao/gemma-4-31B-it-DFlash-GGUF`).
    - **DSpark:** A semi-autoregressive model combining a parallel backbone with a lightweight serial head to reduce "suffix decay" (loss of coherence at the end of draft sequences).
*   **Availability:** The `Bonsai-27B` model (a quantized Qwen3.6-27B fork) can use DSpark speculative decoding (`Bonsai-27B-dspark-Q4_1.gguf`) with the external PrismML fork, which is not vendored in this repository. There are also official DeepSeek DSpark releases for Gemma-4 (e.g. `deepseek-ai/dspark_gemma4_12b_block7` and community GGUF conversions like `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Other target models require custom training via DeepSpec or finding matching community GGUF drafts.
- **Official DSpark integration (2026):** DeepSeek ships DSpark inside SGLang with Gemma-4 drafts (`deepseek-ai/dspark_gemma4_12b_block7`; community GGUF `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Vendor claims 60-85% faster per-user generation / up to 6.6x throughput on datacenter-class hardware; measured 8 GB-class verdicts stay negative (see 4b). The **Gemma-4-backbone** drafts are reachable only via the SGLang backend with a dense on-GPU target; a llama.cpp-native path exists for Qwen3-backbone drafts (next bullet).
- **Native llama.cpp path (in pinned b10375):** upstream merged `draft-dspark` on 2026-07-28 ([#25173](https://github.com/ggml-org/llama.cpp/pull/25173)), so the pinned engine already accepts `--spec-type draft-dspark` for **Qwen3-backbone** drafts (`deepseek-ai/dspark_qwen3_{4b,8b,14b}_block7`, block size 7, convert with `--target-model-dir`). DSpark = DFlash + semi-autoregressive Markov head (`markov_w1/w2` low-rank per-token bias) that recovers left-to-right signal; greedy stays lossless. The confidence head is converted but **not used at inference** in the merge (phase 1; `--spec-draft-conf-min` documented, confidence-scheduled pruning is phase 2). Upstream docs: *"Currently only drafts with a Qwen3 backbone are supported; support for other backbones (e.g. Gemma4) is planned"* — so the Gemma-4 drafts above remain SGLang-only in llama.cpp. `speculators`-format (SpecForge/RedHat) DSpark checkpoints gained conversion support 2026-08-17 ([#26275](https://github.com/ggml-org/llama.cpp/pull/26275)) — **after** the b10375 build, so not in the pinned engine.

### D. N-gram Decoders — `ngram-cache` / `ngram-simple` / `ngram-map-*` / `ngram-mod`

Five draftless (statistical) variants ship in upstream llama.cpp. All search the **token history** (not the KV cache proper — the earlier "searches the KV cache" phrasing here was loose) for repeating n-grams and draft the continuation; the target verifies the draft in the usual batched pass. **0 MB VRAM**, no extra model file, universal across architectures. Mechanism per upstream `docs/speculative.md` + `common/` source (vendored `llama.cpp/` tree, matching master):

- **`ngram-cache`** — maintains statistics of short n-grams (`LLAMA_NGRAM_MIN=1` … `MAX=4`): `n-gram → following-token → count`, built from the prompt and updated per accepted token. Drafts from the empirical distribution, with early-abort thresholds (min sample size / percentage, lax and strict tables). External statistics can be loaded/saved from files (`--spec-lookup-cache-static` / `--spec-lookup-cache-dynamic`, built/merged by the `lookup` example tools). Host-RAM cost grows with context; the dynamic cache persists across sessions to disk. Refs: #5479, #6828, #6848.
- **`ngram-simple`** — looks for the last n-gram in history matching the current context and drafts the **m tokens that followed it**. Defaults `--spec-ngram-simple-size-n 12` (lookup; docs say 12 suits code, 8 suits text), `--spec-ngram-simple-size-m 48` (draft length), `--spec-ngram-simple-min-hits 1`. Ref: #18471.
- **`ngram-map-k`** — same idea with an internal hash map of the n-grams in the current context window (`--spec-ngram-map-k-size-n/m`, `--spec-ngram-map-k-min-hits`); accepted-token counts tracked per n-gram. Ref: #18471.
- **`ngram-map-k4v`** — experimental; per key n-gram, tracks **up to 4 value m-grams** with occurrence stats and drafts the most frequent. Docs example for "a lot of longer repetitions": `--spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2`. Ref: #18471.
- **`ngram-mod`** — LCG rolling-hash pool: n-gram hash → next token, stored in a fixed-size table **shared across all server slots** (requests learn from each other). Constant ~16 MB host RAM, constant cost per step, variable draft length. Defaults `--spec-ngram-mod-n-match 24`, `--spec-ngram-mod-n-min 48`, `--spec-ngram-mod-n-max 64`; docs: *small n not recommended, MoEs require long drafts, dense models can reduce n-min/n-max*. `--spec-default` enables it. Ref: #19164.

**When it helps:** repetition in the generated text — code completion/rewriting, iterating over a block (llama.vim), reasoning models re-stating thinking, summarization. **When it doesn't:** creative text, or any target where the verify pass is the bottleneck (see §4b: small-active MoE pays the expert-union verify cost and loses even at 100% acceptance).

**Community-measured prior (unverified on this rig, labeled as external):** mixed results; not always a net speedup, especially on quantized models; reported positive when stacked with other methods or on high-active MoE/dense. See §4b for the measured A3B case and §2 for the `draft-mtp,ngram-mod` composite.

- The 2026-07-20 small-model TPS matrix has **no ngram row** — candidate Search neighbor on coding-10 (0 MB, universal); prior is negative on A3B-class per [external measurement](./capability-extraction-harness.md) §3/§8.

---

## 4. Local Performance Comparison

### 4a. Fair small-model matrix (2026-07-20) — canonical

Upstream CUDA `llama-cli`, `-n 512`, shared knobs (`q4_0` KV, batch 256/128, threads 6/8, `draft-mtp` n_max=4). Full write-up: [session](../sessions/2026-07-20-small-model-tps-matrix.md) · [operator guide](./small-model-mtp-tps.md).

| Model | Base t/s | MTP t/s | Gain | MTP form |
|---|---:|---:|---:|---|
| Gemma-4 E4B | 67.6 | **122.0** | **+80%** | external draft (~60 MB) |
| Qwen3.5-9B | 38.7 | 57.3 | +48% | embedded `nextn` |
| Ornith-1.0-9B | 38.7 | 56.3 | +46% | Hub MTP GGUF |
| Mythos 5-1M | 40.8 | 41.2 | +1% | Hub MTP GGUF (not worth it) |
| Qwythos-9B-v2 | 40.1 | — | — | no CUDA MTP |

**Default speed Baseline:** Gemma-4 E4B + draft MTP.

### 4b. Earlier spot checks (still useful)

Short `-n 128` / mixed prompts (pre-matrix):

### Gemma-4 E4B:
*   **Baseline (`none`, -n 128):** **69.9 t/s**
*   **MTP (`draft-mtp` n_max=4, -n 128):** **136.6 t/s** (+95.4%, draft ~60 MB VRAM)
*   **MTP sustained (-n 512):** **113.4 t/s** (earlier); matrix **122.0 t/s** under fixed knobs
*   *Note: MTP draft + `--spec-type draft-eagle3` → init error / silent fallback to non-spec.*

### Qwen3.5-9B:
*   **Baseline (`none`):** **39.1 t/s** (earlier spot); matrix **38.7 t/s**
*   **DFlash (`draft-dflash`):** **51.3 t/s** (+31.2% speedup, cost: ~765 MB VRAM)
*   **MTP (`draft-mtp`):** earlier spot **69.1 t/s**; matrix sustained `-n 512` **57.3 t/s** (+48% vs matrix base)

### Qwen3.6-35B-A3B (MoE, 8 GB-class + `--n-cpu-moe`):
*   **Q3 @ 65k (2026-08-12, harness `results.tsv`):** no-spec **24.6** t/s; embedded `draft-mtp` n=1 **29.5** t/s; `draft-dflash` n=15 **12.5** t/s (9.5 rejected at TPS floor).
*   **Q4 @ 32k (2026-08-07):** no-spec **27.2**; MTP n=2 **27.5**; DFlash n=15 **17.5**.
*   **Rule:** DFlash needs the 35B target fully on GPU. Expert CPU offload makes DFlash slower. Do not disable all spec — embedded MTP can still beat no-spec. Eagle-3 historically also lost on this split.
*   **External A3B-class prior (RTX 3090, 2026-04, [HF discussion #14](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/discussions/14), N=3-reproducible):** every spec-decode mode was net-negative on Qwen3.6-35B-A3B — baseline 135.7 t/s; `ngram-mod` n=8..24 129–131 t/s; `ngram-cache` 119.1 t/s (min 65.3, bimodal); 0.8B draft −39 to −60% **despite 100% draft acceptance**. Mechanism: small-active MoE (3B active, 8-of-256, sparsity 0.031) pays the expert-union verify cost — [MoESD arXiv:2505.19645](https://arxiv.org/html/2505.19645), [Utility-Driven SD arXiv:2506.20675](https://arxiv.org/pdf/2506.20675) — confirmed hardware-class-independent (A100 NVLink same magnitude) and acceptance-independent. Counter-example: Qwen3.5-122B-A10B (10B active) gets +15–45% from the same ngram machinery ([#20075](https://github.com/ggml-org/llama.cpp/pull/20075)). This repo's own MTP/DFlash numbers on A3B sit on the same side of the line; the open ngram-coding-10 Neighbor carries a **negative prior on MoE**, positive prior on dense.

### Bonsai-27B (Sparse MoE):
*   **Baseline (`none`):** **37.3 t/s** (Target model `Bonsai-27B-Q1_0.gguf` fully on GPU, ~3.80 GB VRAM)
*   **DSpark (`draft-dspark`):** **19.2 t/s** (**-48.5% performance slowdown!** with draft `Bonsai-27B-dspark-Q4_1.gguf`, ~1.79 GB VRAM)
*   *Note: Under extreme low-bit target quantizations (like 1-bit Q1_0), the main model's forward passes run incredibly fast on GPU (37.3 t/s for 27B parameters). Because the draft model is in a heavier precision (4-bit Q4_1, 1.79 GB), its forward passes are slower per token, making speculation slower than simply running the target model directly (Quantization-Speed Inversion).*

---

## 5. Key Takeaways & Trade-offs (VRAM vs Context Size)

For consumer GPU rigs with constrained VRAM (like our discrete 8 GB-class NVIDIA):

1.  **VRAM and Context Trade-Off:**
    *   Every megabyte saved on the model/draft weights is a megabyte gained for the active KV cache context window.
    *   For a 9B model using a `q4_0` KV cache quantization, saving **~385 MB** of VRAM (MTP vs DFlash) translates directly to an extra **~12,000 tokens of context depth**.
2.  **MTP is usually optimal (but measure):**
    *   On the 2026-07-20 matrix: Gemma draft **+80%**, Qwen/Ornith embedded/Hub MTP **~+46–48%**, Mythos Hub MTP **~+1%** (skip).
    *   Lowest VRAM overhead for Gemma is the tiny assistant draft (~60 MB), not a second full model.
    *   Qwen embeds heads in the main GGUF — no secondary file to track.
3.  **DFlash and Eagle-3 Standby Value:**
    *   While MTP is superior, DFlash and Eagle-3 remain valuable fallback architectures when testing custom fine-tuned models that do not support or were not trained with MTP layers.
