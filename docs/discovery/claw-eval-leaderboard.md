# Claw-Eval Leaderboard (Local Rig)

Canonical **Claw-Eval full** is the Search Val Score (ADR 0004). Claw-Eval quick is smoke only — high quick does **not** guarantee high full.

Hardware context for numbers below: RTX 4060 **8 GB**, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA unless a card says otherwise.

Ground truth: `results.tsv`. Ignore rows with `val_score` outside `[0, 1]` (historical Autoloop pollution — TPS leaked into score).

## Claw-Eval full (n=15) — ranked KEEP / best discards

| Rank | Model | Val Score | bench_tg | peak VRAM | ctx | Alias | Session / note |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `POCKET-35B-Q3_K_M.gguf` | **0.6667** | 35.7 | 3.7 GB | 65k | — | [2026-07-26](../sessions/2026-07-26-pocket-35b-pipeline.md); ties Laguna; coding **0.615** |
| 1 | `Laguna-XS-2.1-Q3_K_XL.gguf` | **0.6667** | 37.2 | 3.6 GB | 65k | `laguna-xs` | [2026-07-24](../sessions/2026-07-24-claw-full-smoke-high.md) |
| 3 | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | **0.6000** | 42.1 | 7.5 GB | 65k | `ornith-9b` | same |
| 3 | `Ornith-1.0-35B-UD-Q4_K_XL.gguf` | **0.6000** | 25.7 | 4.9 GB | 65k | `ornith-35b` | `n-cpu-moe 40` |
| 3 | `LFM2.5-1.2B-Instruct-Q8_0.gguf` | **0.6000** | 166.4 | 3.7 GB | 65k f16 | `lfm2.5-1.2b` | [top-TPS full](../sessions/2026-07-24-claw-full-top-tps.md) |
| 6 | `Qwythos-9B-v2-MTP-Q4_K_M.gguf` | **0.5333** | 34.5 | 7.9 GB | 131k | — | GGUF may be missing; Hub MTP scarce |
| 7 | `Bonsai-27B-Q1_0.gguf` | **0.4667** | 40.2 | 6.5 GB | **65k** | `bonsai` | 131k VRAM-kill mid-agentic |
| 7 | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | **0.4667** | 63.7 | 7.0 GB | **32k** | `ornith-9b-mtp` | [pending queue 2026-07-25](../sessions/2026-07-25-claw-full-pending-queue.md) |
| 9 | `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | **0.4000** | 29.5 | 5.7 GB | 65k | `qwen3.6-35b-q3xl` | same; TEMP=1.0 thinking |
| 10 | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | **0.3333** | 113.3 | 5.8 GB | 65k MTP | `gemma-4-e4b` | top-TPS full; draft-mtp n=4 |
| 10 | `Qwythos-9B-v2-Q4_K_M.gguf` | **0.3333** | 44.1 | 7.7 GB | **32k** | `qwythos-9b-v2` | 65k VRAM-kill mid-full → retry 32k |
| 10 | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | **0.3333** | 43.6 | 8.0 GB | 131k | `qwythos-9b` | historical discard |
| 13 | `nanbeige4.2-3b-Q4_K_M.gguf` | **0.2667** | 32.4 | 6.9 GB | 32k | `nanbeige4.2-3b` | arch fork |
| 13 | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` | **0.2667** | 45.6 | 7.6 GB | 65k | `qwythos-9b-mtp` | pending queue; weaker than non-MTP |
| 15 | `LFM2.5-8B-A1B-Q4_K_M.gguf` | **0.2000** | 178.5 | 7.8 GB | 65k | `lfm2.5-8b-a1b` | `n-cpu-moe 0`; re-run also 0.1333 |
| 15 | `POCKET-26B-Q4_K_M.gguf` | **0.2000** | 20.5 | 4.5 GB | 65k | — | [2026-07-26](../sessions/2026-07-26-pocket-26b-pipeline.md); GGUF **deleted**; coding 0.490 |
| 17 | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | **0.1333** | 29.2 | 4.1 GB | 65k | `gemma-4-26b-a4b` | pending queue; weak |
| 17 | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | **0.1333** | 65.0 | 7.5 GB | 32k | `qwen3.5-9b` | pending queue; harness DISCARD vs polluted prior best |

## Claw-Eval quick (n=5) — smoke ceiling

| Score | Models (examples) |
| :---: | :--- |
| **1.00** | Laguna-XS only (so far) |
| **0.80** | **POCKET-35B Q3_K_M**, LFM2.5-1.2B @ **65k f16** (alias), Bonsai@131k smoke, Ornith-9B, Ornith-35B, some Qwythos |
| **≤0.40** | Prefer sibling / skip full queue unless curious (e.g. LFM2.5-8B-A1B **0.20** → full ≤0.20; Ornith-9B-MTP quick 0.40 → full **0.4667**) |

Smoke ≠ Val Score: Laguna quick 1.00 → full 0.67; Bonsai quick 0.80 → full 0.47; LFM2.5-1.2B quick 0.80 → full **0.60**; LFM2.5-8B quick 0.20 → full 0.13–0.20.

## Failure pattern (2026-07-24 queue)

Across Laguna / Ornith / Bonsai full runs:

* **Pass:** tool-heavy easy tasks (email, calendar, todo, contacts, helpdesk, many notes/finance).
* **Fail / near-zero:** long `web_real` research (CVE, OSS compare, regulatory, US Steel, NFLX ARPPU).
* **Implication:** current local leaders are strong at structured tool use; weak at long synthesis / real-web keyword graders.

## Operational lessons

1. **Queue by quick, decide by full.** Run full on quick ≥0.80 first.
2. **Dense @ max ctx can smoke-pass and agentic-fail.** Bonsai 131k: short-gen OK, mid-full `VRAM_LIMIT EXCEEDED used=7906MB > limit=7900MB` → lock agentic alias to **65k**.
3. **MoE prefer `N_CPU_MOE=block_count`.** Ornith-35B A/B: 40 beat 32 (TPS + VRAM). Laguna already at 40.
4. **One Trial at a time.** Shared GPU + port 18080.
5. **Edit `config.py` Baseline, then**  
   `.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full …"`

## Prefer for agentic use (this rig, 2026-07-24)

1. **`laguna-xs`** — best Val Score + lowest VRAM among leaders.
2. **`lfm2.5-1.2b`** — ties Ornith Val Score (0.6000) at **166 t/s** / 3.7 GB (speed pick).
3. **`ornith-9b`** — same Val Score as 35B/LFM-1.2B, denser capacity than 1.2B.
4. **`ornith-35b`** — same Val Score, more capacity headroom via MoE offload.
5. **`bonsai`** — usable but weaker full; keep ctx 65k for agentic.

Skip for agentic: **`lfm2.5-8b-a1b`** (full ≤0.20 despite top TPS).

## See also

* [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI
* [coding-leaderboard.md](coding-leaderboard.md) — direct-coding 10-task ranks
* [models/aliases/INDEX.md](../../models/aliases/INDEX.md) — live alias table
* Cards: [laguna-xs-2.1.md](../models/laguna-xs-2.1.md), [lfm2.5-1.2b.md](../models/lfm2.5-1.2b.md), [ornith-1.0-9b.md](../models/ornith-1.0-9b.md), [ornith-1.0-35b.md](../models/ornith-1.0-35b.md), [bonsai-27b.md](../models/bonsai-27b.md)
