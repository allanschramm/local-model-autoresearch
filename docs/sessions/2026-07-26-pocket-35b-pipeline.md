# 2026-07-26 — POCKET-35B Q3_K_M full pipeline

## Goal

First-pass champion path for `POCKET-35B-Q3_K_M.gguf` (FINAL-Bench / Darwin-36B-Opus lineage): validation → Claw-Eval full → coding-10 on the same Baseline Fingerprint.

## Hardware

- discrete 8 GB-class NVIDIA (`VRAM_LIMIT_MB=7900`)
- ~32 GB system RAM
- Windows, upstream `llama.cpp` CUDA (`build-cuda`)

## Setup

```python
MODEL = 'POCKET-35B-Q3_K_M.gguf'
CTX_SIZE = 65536
KV_CACHE_K = KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # → 40
TEMP = 0.6  # Darwin recommended
TOP_P = 0.95
TOP_K = 20
SPEC_TYPE = None  # nextn_predict_layers=0
```

GGUF: `models/FINAL-Bench/pocket-35b-gguf/POCKET-35B-Q3_K_M.gguf` (~16.8 GB).

## Commands

```powershell
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate pocket-35b-q3_k_m pipeline"
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full pocket-35b-q3_k_m"
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 pocket-35b-q3_k_m"
```

## Findings

| Stage | Result |
|---|---|
| Validation | KEEP — claw-quick **0.8000**, tg **35.2 t/s**, peak VRAM **3.7 GB** |
| Claw full (15) | KEEP — Val Score **0.6667** (10/15), tg **35.7 t/s**, VRAM **3.7 GB**, ~26 min |
| Coding-10 | coding **0.6150** — HE 0.80 / MBPP 0.90 / LCB 0.50 / BC 0.10; tg 35.0; VRAM 3.6 GB; ~19 min |

Harness printed `DISCARD` on coding-10 because `Current Score` compared coding composite to prior claw KEEP (0.6667). **Ignore that DISCARD for coding runs** — do not `git checkout`. Coding axis is separate from agentic Val Score.

### Claw-full task pattern

- Pass: email triage/draft, calendar, contacts, expense, notes, kb (partial), outage research, CVE (partial), OSS compare (partial)
- Fail: todo management, ticket triage, regulatory research, US Steel, NFLX ARPPU
- One mid-run HTTP 500 on T006 (server recovered; task still PASS 0.60)

### Objective Vector (complete)

- ctx 65k × TPS ~35 × agentic **0.6667** × coding **0.6150**
- Ties Laguna on claw-full; far ahead of Laguna on coding (0.195)

## Errors

- First `hf download` hung at 0 bytes (Xet path). Worked after `HF_HUB_DISABLE_XET=1` + `hf_hub_download`.
- `--coding-task-limit` / `--lcb-task-limit` / `--bigcode-task-limit` rejected as Baseline CLI — use `bench_config.py` defaults (already 10).

## Decisions

- Quant pick: **Q3_K_M** (not publisher Q2_K) — GPU + VITRIOL; ~16.8 GB GGUF.
- No TPS autoloop this session — 35 t/s already above floor; quality gates first.
- Card + leaderboards updated.
