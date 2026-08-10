# 2026-08-02 — Research: repo information/data gap closure (primary sources)

## Goal

Inventory the repo's `**TBD:**` / open-question markers and other information gaps, then close every **web-resolvable** gap from primary sources (official publisher cards, first-party docs, upstream source code). Gaps that need a local GGUF read or a hardware measurement are listed separately and left open. No model was downloaded and no benchmark ran in this session.

## Method

1. Grepped the repo for `TBD`, `TODO`, "not verified", "not surfaced", "gap" markers across `docs/` (models, discovery, sessions) and skills.
2. For each web-resolvable gap, fetched the primary source (HuggingFace card/raw README, unsloth.ai docs page, vLLM docs/GitHub) and recorded exact claims with URL + extraction date.
3. Gap inventory is below; resolutions are keyed to the file each gap lives in, so card edits can be applied one file at a time later.

## Gap inventory

| File | Gap | Resolvable via web? |
| :--- | :--- | :--- |
| `docs/models/qwen-agentworld-35b-a3b.md` | Open Q: relationship to Qwen3.6-35B-A3B — same base? which fine-tune? | ✅ resolved |
| `docs/models/qwen3.6-35b-a3b.md` | AesSedai license unknown; thinking kwargs exact fields; upstream MTP status | ✅ resolved |
| `docs/models/gemma-4-26b-a4b.md` | Recommended settings + llama.cpp guide truncated; MTP "sub-folder in GGUF package" claim | ✅ resolved |
| `docs/models/bonsai-27b.md` | block/expert/head counts TBD; dense-vs-MoE ambiguity; base-model identity | ⚠️ partially (dense + hybrid confirmed; exact tensor dims still local) |
| `docs/models/README.md` | "Open extraction tasks": Qwen3.6 llama.cpp guide + thinking details; Gemma 4 guide + settings | ✅ resolved |
| `docs/discovery/best-model-8gb-vram.md` | Granite-4.0-H-Tiny only in secondary source; "no 8 GB-measured tokens/s" TBD | ⚠️ Granite verified; one 8 GB-device tps found (Jetson) |
| `docs/discovery/vllm-quant-deep-dive.md` | 3 open TBDs: bnb OOT RFC status; W4A8 matrix staleness; GGUF IQ2_M/IQ3_M kernel coverage | ✅ resolved |
| `docs/sessions/2026-07-31-day-model-candidates-100k.md` | Sampler fields TBD for Nemotron 3 Nano, Granite 4.0 H Tiny, Granite 4.1 3B | ⚠️ NVIDIA/IBM do not publish them; confirmed absent |
| `.agents/skills/local-model-alias/SKILL.md` | `measured_by: TBD` in the alias template | ❌ not a research gap (template default for untested aliases) |

## Findings

### 1. Qwen-AgentWorld-35B-A3B — base model identity (RESOLVED)

**Not a Qwen3.6 fine-tune.** The official card frontmatter declares:

```
base_model:
- Qwen/Qwen3.5-35B-A3B-Base
datasets:
- Qwen/AgentWorldBench
```

Qwen-AgentWorld is a **native language world model** for agentic environment simulation — CPT (environment knowledge) → SFT (next-state prediction) → RL/GSPO — covering seven interaction domains (MCP, Search, Terminal, SWE, Android, Web, OS) in one model. Same Qwen3.5 family skeleton (40 layers, 256 experts / 8 routed + 1 shared, hidden 2048, ctx 262,144, Gated DeltaNet + Gated Attention hybrid), Apache-2.0, technical report arXiv 2606.24597.

- **Sampling (official Best Practices):** `temperature=0.6, top_p=0.95, top_k=20`; thinking mode on by default; 32,768 output recommended.
- **MTP: no claim in the official card; treat as unsupported.** The card is silent on MTP, and the base is the Qwen3.5-era checkpoint (MTP is a Qwen3.6-family feature) — this is absence-of-evidence, not a verified absence. Do not assume MTP until checked locally.
- Sources: `https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B/raw/main/README.md` (2026-08-02); `https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF` (2026-08-02).

### 2. Qwen3.6-35B-A3B — license, thinking kwargs, MTP status (RESOLVED)

- **AesSedai license:** `license: apache-2.0` on the AesSedai GGUF repo; base Qwen3.6 is Apache-2.0. Source: `https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF` and `https://huggingface.co/Qwen/Qwen3.6-35B-A3B` (2026-08-02).
- **Thinking enable/disable + preserve (exact fields):** Unsloth doc confirms `--chat-template-kwargs '{"enable_thinking":false}'` to disable; use `'{"preserve_thinking":true}'` to keep the prior thinking trace in continued conversations. PowerShell escaping: `--chat-template-kwargs "{\"enable_thinking\":false}"`. **Preserve thinking is on by default.** Source: `https://unsloth.ai/docs/models/qwen3.6` § 💡 Thinking (2026-08-02).
- **Upstream MTP status:** llama.cpp **merged MTP support**; Unsloth Qwen3.6 MTP GGUFs are out of experimental. Recommended `--spec-draft-n-max 2` (try 1–6); MTP uses ~1 GB extra memory. Speedup: MoE ~1.15–1.25×, dense 1.4–2.0×; acceptance drops 83% → 50% at 4 draft tokens. Sources: `https://unsloth.ai/docs/models/qwen3.6` § ⚡ MTP Guide, § MTP Benchmarks (2026-08-02).
- **Canonical llama.cpp command (tabbed "🦙 Llama.cpp Guide" content):** `llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --alias "unsloth/Qwen3.6-35B-A3B-GGUF" --port 8080 -c 262144 -fa on -ctv q8_0 -ctk q8_0` — note **q8_0 KV cache** in the canonical command (our Baseline uses q4_0). Caveat: the guide renders behind JS tabs; the command above is the background-researcher extraction and should be confirmed against the live tab before changing the Baseline.
- **New:** Unsloth released Qwen3.6 **NVFP4 quants (2026-07-10)** — Blackwell-only (RTX 50x / DGX Spark / B200/B300), W4A4, ~2.5× faster, MTP tensors built in; Unsloth ranks 21/22 quant sizes SOTA on KLD. Relevant to `nvfp4-quantization.md` / the Blackwell future note in `vllm-quant-deep-dive.md`.
- **Local-only remainder:** the `common/arg.cpp` probe for the exact accepted `--spec-type draft-mtp` value, and the tensor-type audit, stay local.

### 3. Gemma 4 (26B-A4B) — settings, llama.cpp guide, MTP packaging (RESOLVED)

- **Recommended settings (Unsloth, verbatim):**
  > It is recommended to use Google's default Gemma 4 parameters: `temperature = 1.0`, `top_p = 0.95`, `top_k = 64`.
  Max context: 128K for E2B/E4B, 262,144 for 12B / 26B-A4B / 31B. Note **`top_k = 64`** (Qwen family uses 20).
- **llama.cpp guide (Unsloth, verbatim):**
  > `llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF --mmproj unsloth/gemma-4-26B-A4B-it-GGUF:mmproj-F16.gguf --port 8080 --ctx-size 262144`
  > To disable thinking / reasoning, use: `--chat-template-kwargs '{"enable_thinking":false}'`
- **MTP packaging — the "sub-folder within the GGUF package" claim is confirmed:**
  > "We updated the Gemma 4 GGUF files to include an additional MTP file inside a separate folder within the GGUF package, so there is no need to download a separate Gemma 4 assistant GGUF. … We uploaded `mtp-` prefixed GGUFs to each repo, so you only need to use the regular original Gemma 4 GGUFs, no separate repo is needed."
  → The MTP draft ships as a separate `mtp-*` GGUF **in the same HF repo tree**, not embedded in the main UD file. The card's recorded local file `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` showed no MTP tensors — **re-pull the repo tree and fetch the `mtp-gemma-4-26B-A4B-it*.gguf` file** if MTP is wanted.
- **MTP speedup:** Gemma 4 QAT + MTP benchmarked at **1.5×–2.2×**; dense variants benefit most (>1.4×). `--spec-draft-n-max 2` best starting point (1–6 to tune).
- Sources: `https://unsloth.ai/docs/models/gemma-4` (2026-08-02), `https://unsloth.ai/docs/models/mtp` (2026-08-02).

### 4. Bonsai-27B — dense vs MoE, hybrid attention, KV (PARTIALLY RESOLVED)

- **Dense, not MoE:** PrismML documents Bonsai-27B as a dense model with hybrid attention (no routed experts) — this supports the card's "treat as dense, no `--n-cpu-moe`" stance. ~75% of layers use linear attention; remaining layers standard attention.
- **KV sizing (from PrismML KV-CACHE guide):** FP16 KV = 64 KiB/token (~6.3 GiB @ 100K); experimental 4-bit KV (`--cache-type-k q4_0 --cache-type-v q4_0`) ≈ 18 KiB/token. Footprints: Q1_0 ≈ 4.8 GiB @ 4K / 5.2 GiB @ 10K / 10.8 GiB @ 100K (weights + activations + FP16 KV).
- **DSpark:** block size 4 (`--spec-draft-n-max 4`), target-specific, ~1.8–2× decode on CUDA; published on datacenter GPUs (~70 → ~135 tok/s at ~0.9 acceptance), off by default because `-np 1` + no cross-request prompt-cache.
- **Published throughput:** RTX 5090 ~163 tok/s (Q1_0), ~134 tok/s (Ternary); Apple M5 Max ~87 tok/s (Q1_0). Quality: Q1_0 ~76.11 avg on PrismML 15-benchmark suite (~89.5% of BF16 baseline).
- **⚠️ Base-model discrepancy:** local GGUF shows `qwen35.*` metadata (card says "Base: Qwen3.5"); PrismML launch materials say the family is "Qwen3.6-27B-derived". Treat publisher wording as marketing until the GGUF/run-guide is re-read locally.
- **Still local-only:** exact `block_count` / `head_count_kv` (PrismML does not publish them; `gguf.GGUFReader` on the local file is the only source).
- Sources: `https://github.com/PrismML-Eng/Bonsai-demo` (README, SPECULATIVE.md, KV-CACHE.md; 2026-08-02), `https://prismml.com/news/bonsai-27b` (2026-08-02).

### 5. `docs/models/README.md` open extraction tasks (RESOLVED)

- Qwen3.6 "🦙 Llama.cpp Guide" → canonical command captured (see §2; KV q8_0 flag).
- Qwen3.6 "💡 Thinking: Enable/Disable + Preserve Thinking" → `enable_thinking` / `preserve_thinking` kwargs confirmed (see §2).
- Gemma 4 "🦙 llama.cpp Guide" + "Recommended Settings" → captured (see §3).

### 6. `docs/discovery/best-model-8gb-vram.md` (PARTIALLY RESOLVED)

- **Granite-4.0-H-Tiny verified from IBM (primary):** decoder-only hybrid MoE (Mamba-2 + Transformer, MoEs with **shared experts**, GQA, SwiGLU, RMSNorm, shared input/output embeddings), 7B total / 1B active, context **128k**, **Apache-2.0**. Sources: `https://huggingface.co/ibm-granite/granite-4.0-h-tiny` and `.../granite-4.0-h-tiny-base` (2026-08-02). (The "4 attention + 36 Mamba-2 layers / 64 experts / 6 active" detail in the 2026-07-31 session came from the official GGUF repo README.)
- **First publisher-measured 8 GB-memory tokens/s found:** NVIDIA reports **18 tokens/s** for Nemotron 3 Nano 4B Q4_K_M with llama.cpp on a **Jetson Orin Nano 8GB** (official blog + GGUF card). Caveat: Orin Nano's GPU is far weaker than an 8 GB-class discrete NVIDIA — this is an 8 GB-memory device figure, not an 8GB-class-discrete figure, so the "no 8 GB GPU tps" TBD is only partially closed. Source: `https://huggingface.co/blog/nvidia/nemotron-3-nano-4b` (2026-08-02).

### 7. `docs/discovery/vllm-quant-deep-dive.md` — three open TBDs (RESOLVED)

- **TBD 1 — bnb OOT migration:** RFC #39583 "[RFC]: Migrate bitsandbytes and GGUF quantization support to OOT plugin" is **still OPEN** (opened 2026-04-11, labels `RFC` + `quantization`). bnb and GGUF remain **in-tree**; the RFC proposes deprecation then removal (removing GGUF first, then bnb, or moving both OOT — feasibility doubted because both inject branches in shared weight-loading code and neither supports `weight_loader_v2`). The live quantization docs (updated 2026-06-12) still list both. Source: `https://github.com/vllm-project/vllm/issues/39583` + `https://docs.vllm.ai/en/latest/features/quantization/index.html` (2026-08-02).
- **TBD 2 — W4A8 matrix staleness:** **CONFIRMED STALE.** The live hardware matrix row `llm-compressor INT8 (W4A8)` shows ✅ **only on Arm CPU** and ❌ everywhere else (Volta → Hopper, AMD, Intel, x86). Source code still ships the CUTLASS W4A8 (SM90) kernel path (`vllm/model_executor/kernels/linear/mixed_precision/cutlass.py`). Docs lag source. Source: `https://docs.vllm.ai/en/latest/features/quantization/index.html` (2026-08-02).
- **TBD 3 — GGUF IQ2_M/IQ3_M coverage:** **CONFIRMED file-type-only.** `vllm_gguf_plugin/gguf_utils.py` `is_valid_gguf_quant_type` docstring: "Some file types (e.g. `IQ2_M`, `IQ3_XS`, `MXFP4_MOE`) have no `GGMLQuantizationType` member, so accept either enum." `IQ2_M`/`IQ3_M` are accepted as **LlamaFileType** (`MOSTLY_*`) names for file selection, but execution is **tensor-type driven** (GGMLQuantizationType dequant kernels) — no dedicated `IQ2_M`/`IQ3_M` Triton module. Source: `https://raw.githubusercontent.com/vllm-project/vllm-gguf-plugin/main/vllm_gguf_plugin/gguf_utils.py` (2026-08-02).

### 8. `docs/sessions/2026-07-31-day-model-candidates-100k.md` — sampler TBDs (CONFIRMED ABSENT)

- **Nemotron 3 Nano 4B:** NVIDIA publishes no `TOP_K` / `MIN_P` / penalty values (reasoning `TEMP=1.0, TOP_P=0.95`; tool calling `TEMP=0.6, TOP_P=0.95` only). Confirmed on the GGUF card README + HF blog. Sampler fields stay TBD. Also confirmed: 42 layers = 21 Mamba + 4 attention + 17 MLP; 262k ctx; Nemotron Open Model License (commercial allowed).
- **Granite 4.0 H Tiny / Granite 4.1 3B:** IBM cards publish **no recommended sampling**; standard Granite generation defaults (temp ≈ 0.7, top-p ≈ 0.9) apply. Granite 4.1 3B architecture fully verified from card: dense, 40 layers, hidden 2560, 40 heads / 8 KV (head size 64), MLP 8192, SwiGLU, RoPE, ctx 131,072, Apache-2.0.
- Sources: `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/blob/main/README.md`, `https://huggingface.co/blog/nvidia/nemotron-3-nano-4b`, `https://huggingface.co/ibm-granite/granite-4.1-3b/blob/main/README.md` (all 2026-08-02).

## Remaining local-only gaps (not web-resolvable)

These stay open until a GGUF read or a hardware measurement runs — do not web-invent them:

1. `bonsai-27b.md` — exact `block_count` / `head_count_kv` (and re-check base-model identity) via `gguf.GGUFReader`.
2. `qwen3.6-35b-a3b.md` — tensor-type audit; local `common/arg.cpp` probe for `--spec-type draft-mtp` acceptance.
3. `gemma-4-26b-a4b.md` — re-download repo tree to fetch the `mtp-*` GGUF; confirm MTP tensors locally before enabling.
4. Day-candidate sampler fields (Nemotron/Granite) — resolved only as "absent from publisher docs"; decide per Trial whether to seed from the general profile or run a quality experiment.
5. Any TPS/VRAM claim for AgentWorld / Granite / Nemotron on the operator host — measurement only.

## Sources / Verification

All URLs fetched 2026-08-02.

| # | URL | Used for |
| :-: | :--- | :--- |
| 1 | https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B/raw/main/README.md | AgentWorld base model, sampling, arch, no-MTP |
| 2 | https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF | Unsloth AgentWorld GGUF packaging |
| 3 | https://unsloth.ai/docs/models/qwen3.6 | Qwen3.6 samplers, thinking/preserve kwargs, llama.cpp guide, MTP status/speedups, NVFP4 |
| 4 | https://unsloth.ai/docs/models/gemma-4 | Gemma 4 recommended settings + llama.cpp guide |
| 5 | https://unsloth.ai/docs/models/mtp | Gemma 4 MTP packaging ("sub-folder in GGUF package"), QAT+MTP speedup |
| 6 | https://huggingface.co/AesSedai/Qwen3.6-35B-A3B-GGUF | AesSedai license (apache-2.0) |
| 6b | https://huggingface.co/Qwen/Qwen3.6-35B-A3B | base Qwen3.6 Apache-2.0 license |
| 7 | https://github.com/PrismML-Eng/Bonsai-demo | Bonsai dense/hybrid, KV-CACHE, DSpark, benchmarks |
| 8 | https://prismml.com/news/bonsai-27b | Bonsai launch quality numbers |
| 9 | https://huggingface.co/ibm-granite/granite-4.0-h-tiny (+ `-base`) | Granite 4.0 H Tiny arch/ctx/license |
| 10 | https://huggingface.co/ibm-granite/granite-4.1-3b/blob/main/README.md | Granite 4.1 3B full arch spec |
| 11 | https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/blob/main/README.md | Nemotron arch, license, server command |
| 12 | https://huggingface.co/blog/nvidia/nemotron-3-nano-4b | 18 tok/s on Jetson Orin Nano 8GB |
| 13 | https://github.com/vllm-project/vllm/issues/39583 | RFC #39583 status (open; bnb/GGUF in-tree) |
| 14 | https://docs.vllm.ai/en/latest/features/quantization/index.html | Hardware matrix rows (W4A8 Arm-only, GGUF row) |
| 15 | https://raw.githubusercontent.com/vllm-project/vllm-gguf-plugin/main/vllm_gguf_plugin/gguf_utils.py | GGUF file-type vs tensor-type acceptance |

## Decisions

- Resolutions are captured in **this single research file** (per the research skill contract). Applying them to the individual model cards / discovery guides (removing `**TBD:**` markers, updating sections) is a separate, per-file edit step — proposed as a follow-up so each card's DOX rules (GGUF-verified architecture, dated TBDs) are respected.
- `measured_by: TBD` in the `local-model-alias` skill is a deliberate template default for untested aliases — left as is, not a gap.
- Newly confirmed facts worth folding into other docs later: Unsloth Qwen3.6 NVFP4 (Blackwell, W4A4, MTP-in-quant) → `nvfp4-quantization.md`; Gemma 4 `top_k=64` vs Qwen `top_k=20`; Nemotron Orin-Nano 18 tok/s → `best-model-8gb-vram.md`.
