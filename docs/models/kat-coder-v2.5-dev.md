# KAT-Coder-V2.5-Dev — Model Card (Local)

**Source (weights):** https://huggingface.co/Kwaipilot/KAT-Coder-V2.5-Dev  
**GGUF:** https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF  
**Base:** Qwen3.6-35B-A3B (post-train SFT+RL)  
**License:** Apache-2.0  
**Local file:** `models/bartowski/kwaipilot-kat-coder-v2.5-dev-gguf/Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` (17.51 GiB)  
**Family:** KAT-Coder (Kwaipilot)  
**Architecture type:** MoE (`qwen35moe`) — 35B total / ~3B active  
**Quantization:** `IQ4_XS` (`general.file_type=30`)

## Architecture (from GGUF metadata)

Verified with `gguf.GGUFReader` on the local file (2026-07-27):

| Key | Value |
|---|---|
| `general.architecture` | `qwen35moe` |
| `general.name` | `KAT Coder V2.5 Dev` |
| `qwen35moe.block_count` | 40 |
| `qwen35moe.context_length` | 262144 |
| `qwen35moe.embedding_length` | 2048 |
| `qwen35moe.expert_count` | 256 |
| `qwen35moe.expert_used_count` | 8 |
| `qwen35moe.full_attention_interval` | 4 |
| tensors | 733 |

Harness `is_moe_model` → **True**. Same layout class as Ornith-35B / POCKET-35B.

## Hardware requirements (RTX 4060 8 GB)

| Quant | Size | Pick |
|---|---|---|
| **IQ4_XS** | **~18.8 GB** | **yes** — under 20 GB disk; MoE + VITRIOL |
| Q4_K_M | ~21.4 GB | over user disk budget |
| Q5+ | ≥24 GB | skip on this rig |

- Experts: `N_CPU_MOE=None` → auto `--n-cpu-moe 40`.
- Prefer ctx **65k** + KV `q4_0` on 8 GB (same band as Ornith-35B agentic).

## Recommended settings

From Kwaipilot HF card (thinking / agentic default; SWE-bench Verified footnote uses TEMP 1.0 / TOP_P 0.95):

| Param | Agentic / thinking (default) | Instruct / non-thinking |
|---|---|---|
| temperature | **1.0** | 0.7 |
| top_p | **0.95** | 0.8 |
| top_k | **20** | 20 |
| presence_penalty | **1.5** | 1.5 |
| min_p | 0.0 | 0.0 |
| repeat_penalty | 1.0 | 1.0 |
| thinking | on by default | `enable_thinking=False` |

Seed `SAMPLER_DEFAULTS` from **agentic / thinking** for Claw + coding-10 unless an explicit quality pass says otherwise.

## MTP

No `nextn_predict_layers` field in this GGUF header (same 733-tensor layout as Ornith-35B without MTP). Leave `SPEC_TYPE=None`.

## VITRIOL split

```text
--n-gpu-layers 99 --n-cpu-moe 40
```

Baseline `N_CPU_MOE=None` auto-resolves to `block_count` (40).

## Our config baseline (validated 2026-07-27)

```python
MODEL = 'Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'q4_0'
KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # → 40
TEMP = 1.0
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5
TPS_FLOOR = 15.0
VRAM_LIMIT_MB = 7900
```

### Benchmark scores (same Fingerprint)

| Axis | Score | Detail |
|---|---|---|
| Claw-Eval quick | 0.8000 | smoke |
| Claw-Eval full | **0.6000** | 9/15; bench_tg **30.2**; peak VRAM **3.4 GB** |
| coding-10 | **0.6400** | LCB 0.50 / HE 0.90 / MBPP 0.90 / BC 0.10; peak **3.3 GB** |

Session: [2026-07-27-kat-coder-v2.5-dev-pipeline.md](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md).

## Sources / Verification

- HF card `Kwaipilot/KAT-Coder-V2.5-Dev` — sampling examples + SWE-bench footnote (extracted 2026-07-27)
- Local GGUF via `gguf.GGUFReader` 2026-07-27
- Bartowski quant table (IQ4_XS size)
- Local pipeline `results.tsv` 2026-07-27

## Open questions

_(none — first Objective Vector complete)_
