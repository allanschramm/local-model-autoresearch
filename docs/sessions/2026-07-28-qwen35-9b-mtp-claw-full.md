# 2026-07-28 — Qwen3.5-9B-MTP claw-full (incomplete → complete)

## Goal
Close Objective Vector for `Qwen3.5-9B-MTP-Q4_K_M.gguf` (coding-10 **0.4950** in TSV). After deleting deepreinforce ornith-9b post-complete.

## Hardware
discrete 8 GB-class NVIDIA (~8 GB physical VRAM), Windows, upstream `llama.cpp/build-cuda`.

## Setup
1. `hf download unsloth/Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/unsloth/qwen3.5-9b-mtp-gguf`
2. Hardlink → `Qwen3.5-9B-MTP-Q4_K_M.gguf` (TSV basename; HF file omits `MTP` in name).
3. Baseline: ctx **32768**, kv q4_0, `draft-mtp` n=4, batch 256/128, threads 6/8, TEMP **0.4** / repeat **1.05** (card).

## Commands
```powershell
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
# attempt 1 — limit 7900 → VRAM kill T004 @ 7906
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full Qwen3.5-9B-MTP-Q4_K_M ctx32k mtp4"
# attempt 2 — limit 8000 → OK
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full Qwen3.5-9B-MTP-Q4_K_M ctx32k mtp4 vram8k (7906 retry)"
```

## Findings
| Attempt | Result |
| :--- | :--- |
| 32k+MTP / 7900 | `MODEL_REJECTED` — kill mid T004 @ 7906 MB |
| 32k+MTP / **8000** | **OK 0.2000** (3/15), bench_tg **67.5**, peak **7.7 GB** |

Vector complete: claw **0.2000** + coding **0.4950** (`iq_min=0.2000`). Weak agentic (UD sibling claw was 0.1333).

## Decisions
- Baseline `VRAM_LIMIT_MB` restored to **7900** after trial.
- Next incomplete by TPS: `Qwythos-9B-v2-MTP` coding-10 (~34.5) or gemma-4-12B claw (~33.6) — check disk.
