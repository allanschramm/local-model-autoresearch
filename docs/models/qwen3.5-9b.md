# Qwen3.5-9B — Model Card (Local)

**Source repo:** https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF
**Unsloth docs:** https://unsloth.ai/docs/models/qwen3.5
**License:** Apache-2.0
**Local file:** `models/Qwen3.5-9B-UD-Q4_K_XL.gguf` (6.14 GB) (previously `models/Qwen3.5-9B-Q4_K_M.gguf`)
**Family:** Qwen 3.5 (Alibaba)
**Quantization:** Unsloth Dynamic Q4_K_XL with MTP — `UD-Q4_K_XL` (QAT-lossless 4-bit)

## Architecture (from GGUF metadata)
- Causal LM (hybrid Attention + SSM — Qwen 3.5 architecture)
- **`block_count` = 32 layers**
- Hidden **4096**, context **131072**
- **Hybrid Attention + SSM (Mamba-2 style)**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - SSM: `conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=32`, `inner_size=4096`
  - 8 full attention layers, 24 SSM layers

## Hardware Requirements (RTX 4060 8GB)
- Fits entirely in GPU VRAM (NGL = 99).
- Model size ~6.14 GB, leaving adequate headroom for context cache.

## MTP (Multi-Token Prediction)
- **MTP tensors are embedded in this GGUF** (`qwen35.nextn_predict_layers`, `blk.32.nextn.*` verified 2026-07-20).
- Enable: `SPEC_TYPE = "draft-mtp"`, `SPEC_DRAFT_N_MAX = 4` — **no** `--spec-draft-model`.
- Fair matrix (2026-07-20, `llama-cli` `-n 512`): base **38.7 t/s** → MTP **57.3 t/s** (**+48%**). Evidence: [session](../sessions/2026-07-20-small-model-tps-matrix.md).

## Recommended Settings
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.05
- **Chat Template:** Jinja (requires `--jinja` flag)

## Config Baseline (2026-07-20 TPS matrix knobs)
- `MODEL = 'Qwen3.5-9B-UD-Q4_K_XL.gguf'`
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `KV_CACHE_K = 'q4_0'`
- `KV_CACHE_V = 'q4_0'`
- `NGL = 99`
- `THREADS = 6`
- `THREADS_BATCH = 8`
- `BATCH_SIZE = 256`
- `UBATCH_SIZE = 128`
- `FLASH_ATTN = 'on'`
- `SPEC_TYPE = 'draft-mtp'`
- `SPEC_DRAFT_N_MAX = 4`
- `NO_MMAP = True`

### Status
- **Measured (2026-07-20):** embedded MTP works on upstream CUDA; +48% vs base under fair knobs. Slower absolute than Gemma+draft MTP (122 t/s).

### Claw-Eval full (2026-07-25) — UD basename
- **agentic_full 0.1333** @ ctx **32768**, draft-mtp n=4, TEMP 0.4, bench_tg **65.0**, peak **7.5 GB**. Weak agentic — skip for tool loops.

### Claw-Eval full (2026-07-28) — `Qwen3.5-9B-MTP-Q4_K_M.gguf`
- First try @ 32k+MTP / limit 7900: **VRAM kill** mid T004 @ 7906 MB.
- Success @ 32k+MTP / limit **8000**: **agentic_full 0.2000** (3/15), bench_tg **67.5**, peak **7.7 GB**. Vector **complete** with coding **0.4950** (`iq_min=0.2000`). Still weak agentic.

### Coding-10 — **rejected on this rig** (2026-07-27) — UD basename only
Objective Vector **incomplete** for `Qwen3.5-9B-UD-Q4_K_XL.gguf` (coding axis never landed). All attempts hit harness `VRAM_LIMIT_MB=7900` mid-eval during long codegen:

| config | outcome |
|---|---|
| ctx 32768 + MTP n=4 | **VRAM kill** 7913 MB mid HumanEval |
| ctx 65536, no MTP | **VRAM kill** 7931 MB mid HumanEval (preflight est 7584 MB) |

Claw-full @ 32k+MTP fits because tool calls are short; coding-10 long generation does not. **Do not retry** UD coding on RTX 4060 8 GB unless `VRAM_LIMIT_MB` or ctx/KV policy changes. **SSD delete candidate** — weak agentic + no coding vector.

`Qwen3.5-9B-MTP-Q4_K_M` already has coding-10 **0.4950** in TSV (separate Trial).

## Open questions
- None for this rig — UD coding axis closed as failure; MTP basename vector complete (weak).
