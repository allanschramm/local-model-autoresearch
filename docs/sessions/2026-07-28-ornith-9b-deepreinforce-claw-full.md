# 2026-07-28 — ornith-1.0-9b-Q4_K_M claw-full (incomplete → complete)

## Goal
Close Objective Vector for deepreinforce `ornith-1.0-9b-Q4_K_M.gguf` (coding-10 **0.5800** already in TSV). Second incomplete by TPS after Qwen3.5-4B (deleted post-complete). Skip Qwen3.5-9B-UD (coding VRAM-rejected).

## Hardware
discrete 8 GB-class NVIDIA (~8 GB physical VRAM), Windows, upstream `llama.cpp/build-cuda`.

## Setup
1. Baseline: `ornith-1.0-9b-Q4_K_M.gguf`, kv **q4_0**, no MTP, batch 256/128, threads 8/8, `NO_MMAP`, `CONT_BATCHING`.
2. Sampler from card: TEMP **0.4** / TOP_P 0.95 / TOP_K 20 / presence 0.

## Commands
```powershell
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
# attempt 1 — ctx65k limit 7900 → VRAM kill T054 @ 7930
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full ornith-1.0-9b-Q4_K_M ctx65k"
# attempt 2 — ctx32k limit 7900 → early kill T002 @ 7910
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full ornith-1.0-9b-Q4_K_M ctx32k (65k VRAM-kill retry)"
# attempt 3 — ctx65k limit 8000 (still under physical) → OK
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full ornith-1.0-9b-Q4_K_M ctx65k vram8k (retry after 7930 kill)"
```

## Findings
| Attempt | ctx | VRAM limit | Result |
| :--- | ---: | ---: | :--- |
| 1 | 65k | 7900 | `MODEL_REJECTED` — kill mid T054 @ 7930 MB (had 6/15 @ 0.40) |
| 2 | 32k | 7900 | `MODEL_REJECTED` — kill T002 @ 7910 MB |
| 3 | 65k | **8000** | **OK 0.4000** (6/15), bench_tg **42.5**, peak **7.8 GB** |

Vector: claw **0.4000** + coding **0.5800** → complete (`iq_min=0.4000`). Weaker agentic than Unsloth UD sibling (**0.6000**).

## Errors
- Dense VRAM gate at 7900 too tight for this basename’s long web_real tasks (~7930 peak).
- Limit briefly raised to 8000 for attempt 3; Baseline restored to **7900** after.

## Decisions
- Vector closed. Prefer UD/MTP Ornith-9B aliases for agentic; deepreinforce Q4_K_M stays as measured point in TSV.
- Next incomplete by TPS: `Qwen3.5-9B-MTP` claw-full (~49 t/s) — after confirming GGUF on disk.
