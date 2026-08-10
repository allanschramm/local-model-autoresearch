# POCKET-26B — Model Card (Local, historical)

**Status:** **DELETED from `models/` + alias removed** (2026-07-26). Doc + `results.tsv` rows kept.  
**Source repo:** https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF  
**Base model:** https://huggingface.co/google/gemma-4-26B-A4B-it  
**License:** Apache-2.0  
**Local file:** ~~`models/FINAL-Bench/pocket-26b-gguf/POCKET-26B-Q4_K_M.gguf`~~ (removed; was ~15.64 GiB)  
**Family:** POCKET (FINAL-Bench / VIDRAFT) — Korean-tuned quant of Gemma4-26B-A4B  
**Architecture type:** MoE (`gemma4`) — 26B total / ~4B active  
**Quantization:** `Q4_K_M` (`general.file_type=15`)

## Why deleted

Weak on the operator host vs `pocket-35b`: claw-full **0.2000**, coding **0.490**, ~21 t/s. Prefer POCKET-35B. Scores stay in `results.tsv` / leaderboards / session log.

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-26):

| Key | Value |
|---|---|
| `general.architecture` | `gemma4` |
| `gemma4.block_count` | 30 |
| `gemma4.context_length` | 262144 |
| `gemma4.embedding_length` | 2816 |
| `gemma4.expert_count` | 128 |
| `gemma4.expert_used_count` | 8 |
| tensors | 658 |
| MTP | **none** |

Harness `is_moe_model` → **True**. Sibling Unsloth Gemma card: [gemma-4-26b-a4b.md](gemma-4-26b-a4b.md).

## Hardware requirements (discrete 8 GB-class NVIDIA + ~32 GB RAM)

| Quant | Size | Notes |
|---|---|---|
| Q2_K | 11 GB | publisher daily driver; GPQA 67.2% |
| **Q4_K_M** | **~17 GB** | **our pick**; GPQA **67.7%** (= base) |

MoE: `N_CPU_MOE=None` → auto `--n-cpu-moe 30`.

## Recommended settings

From GGUF `general.sampling.*` (2026-07-26):

| Param | Value |
|---|---|
| temperature | **1.0** |
| top_p | 0.95 |
| top_k | **64** |
| min_p | 0.0 |
| repeat_penalty | 1.0 |

Publisher card has no separate coding profile — keep GGUF defaults through TPS/quality gates.

## MTP

No MTP tensors. `SPEC_TYPE=None`.

## VITRIOL split

```text
--n-gpu-layers 99 --n-cpu-moe 30
```

## Our config baseline (validated 2026-07-26)

```python
MODEL = 'POCKET-26B-Q4_K_M.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # → 30
TEMP = 1.0
TOP_P = 0.95
TOP_K = 64
SPEC_TYPE = None
TPS_FLOOR = 15.0
```

| Metric | Value |
|---|---|
| Bench tg | **20.4–21.8 t/s** |
| Peak VRAM | **4.5–4.7 GB** |
| Claw quick (5) | **0.4000** |
| Claw full (15) | **0.2000** (3/15) |
| Coding-10 | **0.4900** (HE 0.60 / MBPP 0.60 / LCB 0.50 / BC 0.10) |
| Session | [2026-07-26-pocket-26b-pipeline.md](../sessions/2026-07-26-pocket-26b-pipeline.md) |
| Alias | ~~`pocket-26b`~~ (removed 2026-07-26) |

**GGUF deleted 2026-07-26** — prefer `pocket-35b` on the operator host. Historical scores above remain valid.

## Sources / Verification

- HF: https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF (2026-07-26)
- Local GGUF via `gguf.GGUFReader` (2026-07-26)
- Pipeline: [2026-07-26-pocket-26b-pipeline.md](../sessions/2026-07-26-pocket-26b-pipeline.md)

## Open questions

- Whether Q2_K (11 GB, GPQA 67.2%) is faster here without coding/agentic loss
- Sampler A/B (TEMP below 1.0) for English agentic — GGUF default is TEMP=1.0

