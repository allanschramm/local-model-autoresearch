# 2026-07-25 — Claw-Eval full pending-alias queue

## Goal
Run Claw-Eval full (n=15) on every alias that lacked a sane Val Score: `qwythos-9b-v2`, `qwen3.6-35b-q3xl`, `qwythos-9b-mtp`, `ornith-9b-mtp`, `gemma-4-26b-a4b`, `qwen3.5-9b`. Sequential; Baseline via `config.py` only; sampler seeded from model cards.

## Hardware
RTX 4060 8 GB (`VRAM_LIMIT_MB=7900`), Windows, upstream `llama.cpp/build-cuda`.

## Setup
1. Seed engine flags from `models/aliases/<name>/config.yaml`.
2. Seed `SAMPLER_DEFAULTS` from `docs/models/<card>.md` Recommended settings (job=agentic).
3. `$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1`
4. `.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full …"`

## Blockers hit
- Fresh venv missing `gguf` → MoE misclassified as dense. Fixed via requirements.
- Fresh venv missing `fastapi`/`uvicorn` → claw-eval mock services exit before readiness on `:9100`. Installed from `claw-eval/requirements.txt` and declared in root `requirements.txt`.

## Results

| Alias | Val Score | pass | bench_tg | peak VRAM | ctx | status | notes |
| :--- | :---: | :---: | ---: | ---: | ---: | :--- | :--- |
| `qwythos-9b-v2` | **0.3333** | 5/15 | 44.1 | 7.7 GB | **32k** | KEEP | 65k mid-full `VRAM_LIMIT` → retry 32k |
| `qwen3.6-35b-q3xl` | **0.4000** | 6/15 | 29.5 | 5.7 GB | 65k | KEEP | TEMP=1.0 thinking; `n-cpu-moe 40` MTP n=1 |
| `qwythos-9b-mtp` | **0.2667** | 4/15 | 45.6 | 7.6 GB | 65k | KEEP | Weaker than Mythos non-MTP 0.3333 |
| `ornith-9b-mtp` | **0.4667** | — | 63.7 | 7.0 GB | 32k | KEEP | Ties Bonsai; faster than Ornith UD |
| `gemma-4-26b-a4b` | **0.1333** | — | 29.2 | 4.1 GB | 65k | KEEP | Matches weak quick (0.20) |
| `qwen3.5-9b` | **0.1333** | — | 65.0 | 7.5 GB | 32k | DISCARD | Valid score; harness compared to polluted prior best 39.5 |

## Decisions
- Prefer **Ornith UD / Laguna / LFM 1.2B** still for agentic; new queue does not dethrone Laguna 0.6667.
- `ornith-9b-mtp` useful as speed sibling of Ornith UD (0.4667 @ 63.7 t/s vs 0.6000 @ 42).
- Skip agentic: Gemma 26B-A4B, Qwen3.5-9B (full 0.1333).

## See also
- [claw-eval-leaderboard.md](../discovery/claw-eval-leaderboard.md)
