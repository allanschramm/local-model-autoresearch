# Session 2026-07-24 — Claw-Eval full on top-TPS models

## Goal

Run Claw-Eval **full** (n=15) on the three highest measured TPS models from `results.tsv`: LFM2.5-1.2B, LFM2.5-8B-A1B, Gemma-4-E4B (draft MTP).

## Hardware

RTX 4060 8 GB, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA (`build-cuda`).

## Setup

Edit `autoresearch/core/config.py` Baseline per model, then:

```powershell
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full …"
```

One Trial at a time (shared GPU + port 18080).

## Commands / Baselines

| Model | Key Baseline knobs |
| :--- | :--- |
| `LFM2.5-1.2B-Instruct-Q8_0.gguf` | ctx **65k**, KV **f16**, dense, `TPS_FLOOR=15` |
| `LFM2.5-8B-A1B-Q4_K_M.gguf` | ctx **65k**, KV **q4_0**, `N_CPU_MOE=0`, temp 0.2 / top_k 80 / rp 1.05 |
| `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | ctx **65k**, draft-mtp n=4, batch 256/128, threads 6 |

## Findings

| Model | Val Score | pass | bench_tg | peak VRAM | wall | status note |
| :--- | :---: | :---: | ---: | ---: | ---: | :--- |
| LFM2.5-1.2B | **0.6000** | 9/15 | 166.4 | 3.7 GB | 178 s | KEEP |
| LFM2.5-8B-A1B (run A) | 0.1333 | 2/15 | 184.3 | 7.3 GB | 326 s | KEEP vs empty prev |
| LFM2.5-8B-A1B (run B) | **0.2000** | 3/15 | 178.5 | 7.8 GB | 347 s | accidental re-run (race) |
| Gemma-4-E4B MTP | **0.3333** | 5/15 | 113.3 | 5.8 GB | 502 s | DISCARD vs polluted prev 76.67 (engine-tps) |

### Notable task patterns

* **LFM 1.2B:** strong on contacts + several research tasks (T044/T046/T048); weak on notes/helpdesk/finance synthesis (US Steel / NFLX = 0).
* **LFM 8B:** tool-call / grader mismatch continues from quick 0.20 — almost all research tasks 0; only a few structured passes.
* **Gemma E4B:** mid pack; HTTP 500 on one turn (T006); OSS compare perfect (1.00); finance research still 0.

## Errors

1. **Config race:** Shell launched in parallel with `config.py` StrReplace → first “Gemma” job actually re-ran LFM 8B. Fix: verify `ENGINE_DEFAULTS['MODEL']` print before harness.
2. **Harness KEEP/DISCARD noise:** Gemma compared Val Score to Autoloop `76.667` (TPS leaked into previous best). Score itself valid; ignore STATUS label when previous best ∉ `[0,1]`.

## Decisions

* LFM 1.2B @ 65k f16 joins Ornith tier on Val Score (**0.6000**) with ~4× TPS — strong speed+agentic Pareto candidate.
* LFM 8B not preferred for agentic (full ≤0.20) despite top TPS.
* Gemma E4B full 0.3333 below Mythos historical discard note; speed Baseline stays valid, agentic prefer Laguna/Ornith/LFM-1.2B.

## See also

* Leaderboard: [docs/discovery/claw-eval-leaderboard.md](../discovery/claw-eval-leaderboard.md)
* Cards: [lfm2.5-1.2b.md](../models/lfm2.5-1.2b.md), [lfm2.5-8b-a1b.md](../models/lfm2.5-8b-a1b.md), [gemma-4-e4b.md](../models/gemma-4-e4b.md)
