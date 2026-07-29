# 2026-07-28 — Qwen3.5-4B-MTP claw-full (incomplete → complete)

## Goal
Close Objective Vector for `Qwen3.5-4B-MTP-Q4_K_M.gguf` (coding-10 already in TSV; claw-full missing). First of the incomplete queue ordered by TPS.

## Hardware
RTX 4060 8 GB (`VRAM_LIMIT_MB=7900`), Windows, upstream `llama.cpp/build-cuda`.

## Setup
1. Hardlink disk `Qwen3.5-4B-Q4_K_M.gguf` → `Qwen3.5-4B-MTP-Q4_K_M.gguf` (same dir) so Baseline basename matches historical TSV coding row.
2. Baseline in `autoresearch/core/config.py`: ctx **131072**, kv **q4_0**, `draft-mtp` n=4, batch 256/128, threads 6/8, `NO_MMAP`, `CONT_BATCHING`.
3. Sampler: Unsloth Qwen3.5 thinking/general (agentic) — TEMP 1.0 / TOP_P 0.95 / TOP_K 20 / presence 1.5.

## Commands
```powershell
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full Qwen3.5-4B-MTP-Q4_K_M ctx131k mtp4"
```

## Findings
| Metric | Value |
| :--- | :--- |
| Val Score (claw-full) | **0.2667** (4/15) KEEP |
| bench_tg | **87.0** t/s |
| Peak VRAM | **7.6 GB** |
| Coding (existing TSV) | **0.3850** |
| Vector | **complete** — `iq_min=0.2667` @ ctx 131k |

Weak agentic (mostly tool/keyword fails on research tasks). Fast dense MTP point; not Day/Night competitive vs Laguna/POCKET/KAT.

## Errors
- Intermittent `HTTP Error 500` on some agent turns (gmail/web_real) — tasks still scored.

## Decisions
- Vector closed for this basename. GGUF deleted from disk after complete vector (scores stay in TSV).
- Next incomplete by TPS: skip `Qwen3.5-9B-UD` (coding VRAM-rejected); then `ornith-1.0-9b-Q4_K_M` claw-full.
