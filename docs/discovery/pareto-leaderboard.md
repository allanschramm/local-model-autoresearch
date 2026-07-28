# Pareto Frontier Leaderboard (Local Rig)

Global **Pareto Set** on this hardware budget: maximize **ctx × TPS × agentic × coding** ([ADR 0006](../adr/0006-pareto-frontier-search.md)). Selection lenses: **Day** / **Night** ([ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md)).

Hardware: RTX 4060 **8 GB**, `VRAM_LIMIT_MB=7900`, Windows, upstream CUDA unless noted.  
Ground truth: `results.tsv`. TPS axis = claw-full `bench_tg` when available. Complete vector = claw-full **and** coding-10 (exact 10 tasks/dataset).

Recompute live (do not invent temp scripts):

```bash
.\venv\Scripts\python.exe scripts\rank_results.py
.\venv\Scripts\python.exe scripts\rank_results.py --day-iq-ratio 0.8
```

## Usage Profile picks (ADR 0008)

| Lens | Rule | Pick (this front) |
| :--- | :--- | :--- |
| **Night** | `CTX ≥ 65536` then max `min(agentic, coding)` | **POCKET-35B-Q3_K_M** (min **0.615**); KAT close second (min **0.600**) |
| **Day** | `min(ag,cod) ≥ 0.75 × IQ_best` then max TPS | **Ornith-1.0-9B-MTP** (~64 t/s, min **0.467**) |

`IQ_best` on this front ≈ **0.615** → Day IQ floor ≈ **0.461**. LFM 1.2B (min 0.35) out. Raise `DAY_IQ_RATIO` to **0.8** → Day = Ornith-9B-UD (~42 t/s, min 0.57). Method: [pareto-selection.md](pareto-selection.md).

## `on_front` (complete / merged)

Sorted by `min(agentic, coding)` descending. TPS = claw-full preferred.

| # | Model | ctx | TPS | agentic | coding | min | Note |
| :---: | :--- | ---: | ---: | :---: | :---: | :---: | :--- |
| 1 | `POCKET-35B-Q3_K_M.gguf` | 65k | 35.7 | 0.6667 | 0.6150 | **0.6150** | **NIGHT**; balanced champ |
| 2 | `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` | 65k | 30.2 | 0.6000 | 0.6400 | **0.6000** | [2026-07-27](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md); coding ties Mythos |
| 3 | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | 65k* | 42.1 | 0.6000 | 0.5700 | 0.5700 | *ag@65k / cod@32k merge |
| 4 | `Ornith-1.0-9B-MTP-Q4_K_M.gguf` | 32k | 63.7 | 0.4667 | 0.5800 | 0.4667 | **DAY** (ADR 0008 IQ band → max TPS) |
| 5 | `LFM2.5-1.2B-Instruct-Q8_0.gguf` | 65k | 166.4 | 0.6000 | 0.3500 | 0.3500 | fast; IQ below Day floor |
| 6 | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 131k | 43.6 | 0.3333 | 0.6400 | 0.3333 | coding king (tie KAT) |
| 7 | `Qwythos-9B-v2-Q4_K_M.gguf` | 32k | 44.1 | 0.3333 | 0.6150 | 0.3333 | |
| 8 | `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` | 65k | 113.3 | 0.3333 | 0.5550 | 0.3333 | fast; IQ below Day floor |
| 9 | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 65k | 29.2 | 0.1333 | 0.5900 | 0.1333 | coding strong, agentic dead |
| 10 | `Ornith-1.0-35B-UD-Q3_K_XL.gguf` | 65k | 25.2 | 0.4667 | 0.5550 | 0.4667 | Q3 vs Q4 A/B — prefer Q4 quality |
| 11 | `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` | 65k | 45.6 | 0.2667 | 0.5500 | 0.2667 | GGUF deleted; complete vector in TSV (weaker than non-MTP Mythos) |
| 12 | `LFM2.5-8B-A1B-Q4_K_M.gguf` | 65k | 178.5 | 0.2000 | 0.3650 | 0.2000 | max TPS; not Day pick |
| 13 | `POCKET-26B-Q4_K_M.gguf` | 65k | 20.5 | 0.2000 | 0.4900 | 0.2000 | GGUF deleted; complete vector in TSV |
| 14 | `Laguna-XS-2.1-Q3_K_XL.gguf` | 65k | 37.2 | 0.6667 | 0.1950 | 0.1950 | agentic top; coding weak |

Exact domination membership can shift when TPS sources differ (claw vs coding Combined TPS). Treat the table as the teaching front; recompute from `results.tsv` before deleting GGUFs. **TSV wins over this doc.**

## `dominated` (complete, someone covers)

Examples: `Bonsai-27B-Q1_0` (claw 0.4667 / coding 0.4550 in TSV), **`Ornith-1.0-35B-UD-Q4_K_XL`** (claw 0.6000 / coding 0.5800 — covered by KAT on same 65k: equal agentic, worse coding+TPS). Incomplete claw-only (no coding-10 row): `Qwen3.6-35B-A3B-UD-Q3_K_XL`, `nanbeige4.2-3b`.

## `incomplete` / rejected

| Model | Gap | Status |
| :--- | :--- | :--- |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | coding | **Rejected** — VRAM kill @ 32k+MTP and 65k no MTP mid coding-10 (attempt rows in TSV). |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | coding-10 | claw-full **0.4000** in TSV; **no** fair coding-10 row |
| `nanbeige4.2-3b-Q4_K_M.gguf` | coding-10 | claw-full **0.2667** in TSV; **no** fair coding-10 row |

## Quantizations are separate Trials

Different quants of the same family (e.g. Ornith-35B **Q3_K_XL** vs **Q4_K_XL**) are **not** duplicates. Each needs its own Objective Vector. Prefer the better quant for aliases; keep both scores in leaderboards until a delete decision.

## Prefer by job (this rig)

| Job | Prefer |
| :--- | :--- |
| Night / long agent loops | **POCKET-35B** (KAT close second) |
| Day / supervised (ADR 0008) | **Ornith-9B-MTP** (or Ornith-9B-UD if `DAY_IQ_RATIO=0.8`) |
| Direct coding | Mythos / **KAT-Coder** → POCKET-35B → Ornith-9B |
| SSD cleanup | delete weak/`rejected` first; Q3 Ornith-35B optional if keeping Q4 |

## See also

* [ADR 0006](../adr/0006-pareto-frontier-search.md) — Pareto Set membership  
* [ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md) — Day IQ ε-band → max TPS  
* [pareto-selection.md](pareto-selection.md) — method citations (maximin / ε-constraint)  
* [claw-eval-leaderboard.md](claw-eval-leaderboard.md) · [coding-leaderboard.md](coding-leaderboard.md)  
* Session: [2026-07-27 incomplete vectors + Pareto](../sessions/2026-07-27-incomplete-vectors-pareto.md), [KAT-Coder pipeline](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md)
