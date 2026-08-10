# 2026-07-24 — Claw-Eval full queue (smoke-high)

## Goal
Run canonical Claw-Eval full (n=15) on four high claw-quick GGUFs: Laguna-XS, Ornith-9B UD, Ornith-35B UD, Bonsai. Sequential; Baseline via `config.py` only. Afterwards rank all historical claw scores in `results.tsv` and document durable findings.

## Hardware
discrete 8 GB-class NVIDIA (`VRAM_LIMIT_MB=7900`), Windows, upstream `llama.cpp/build-cuda`.

## Setup
1. Seed Baseline flags in `autoresearch/core/config.py` `ENGINE_DEFAULTS` (GGUF basename + knobs).
2. One model per harness invocation (no parallel GPU Trials).
3. `$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1` for Windows + live logs.

## Commands
```powershell
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
# edit autoresearch/core/config.py ENGINE_DEFAULTS per Trial, then:
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full <basename> ..."
```

Descs used:
- `claw-full Laguna-XS-2.1-Q3_K_XL n-cpu-moe40 ctx65k`
- `claw-full Ornith-1.0-9B-UD-Q4_K_XL ctx65k`
- `claw-full Ornith-1.0-35B-UD-Q4_K_XL n-cpu-moe40 ctx65k`
- `claw-full Bonsai-27B-Q1_0 ctx131k` → **FAIL** VRAM
- `claw-full Bonsai-27B-Q1_0 ctx65k (131k VRAM-kill retry)` → KEEP

## Findings

### Queue results

| Model (basename) | Val Score | pass | bench_tg | peak VRAM | ctx | notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Laguna-XS-2.1-Q3_K_XL` | **0.6667** KEEP | 10/15 | 37.2 | 3.6 GB | 65k | `n-cpu-moe 40`; best of queue **and** best full in `results.tsv` |
| `Ornith-1.0-9B-UD-Q4_K_XL` | **0.6000** KEEP | 9/15 | 42.1 | 7.5 GB | 65k | dense |
| `Ornith-1.0-35B-UD-Q4_K_XL` | **0.6000** KEEP | 9/15 | 25.7 | 4.9 GB | 65k | `n-cpu-moe 40` |
| `Bonsai-27B-Q1_0` | **0.4667** KEEP | 7/15 | 40.2 | 6.5 GB | **65k** | retry after 131k kill |

All four beat prior dense claw-full keep reference (nanbeige **0.2667**).

### Historical claw-full ceiling (sane `val_score` ∈ [0,1])

| Score | Model |
| :---: | :--- |
| **0.6667** | Laguna-XS-2.1 Q3_K_XL ← **max** |
| 0.6000 | Ornith-9B / Ornith-35B |
| 0.5333 | Qwythos-9B-v2-MTP |
| 0.4667 | Bonsai-27B Q1_0 @ 65k |
| 0.2667 | nanbeige4.2-3b |

Max claw-**quick**: Laguna **1.0000**.

Durable table: [docs/discovery/claw-eval-leaderboard.md](../discovery/claw-eval-leaderboard.md).

### Task pattern
Easy tool tasks (gmail/calendar/todo/contacts/helpdesk) mostly PASS. Long `web_real` research (T046 CVE, T048 OSS, T050 regulatory, T053/T054 finance) FAIL or near-zero across all four.

## Errors
- **Bonsai @ 131k:** mid-agentic `VRAM_LIMIT EXCEEDED used=7906MB > limit=7900MB` — server killed; remaining tasks `No connection could be made` → score 0.0000 discard. Retry @ 65k succeeded (peak 6.5 GB).
- Autoloop historical rows with `val_score` ≫ 1 (e.g. ~39) are **not** claw scores — exclude from leaderboard.

## Decisions
- Prefer **Laguna-XS-2.1-Q3_K_XL** as then-current agentic Val Score leader on the operator host.
- Leaderboard Score column = claw-full when measured (canonical).
- Bonsai agentic recipe locked to ctx **65536**; short-gen @ 131k still OK for TPS-only.
- New card: [docs/models/laguna-xs-2.1.md](../models/laguna-xs-2.1.md).
- Ornith-35B / Laguna: `n-cpu-moe = block_count` (40).
