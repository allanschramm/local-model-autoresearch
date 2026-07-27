# Coding Preflight Leaderboard (Local Rig)

Direct-coding score (optional Search preflight):  
`coding = 0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode`  
Exactly **10 tasks** per dataset. Ground truth: `results.tsv` (`category=10-task` / `scoring_benchmark=coding`).

Hardware: RTX 4060 **8 GB**, `VRAM_LIMIT_MB=7900`, Windows, upstream CUDA unless noted.

Claw-Eval full is the agentic axis — see [claw-eval-leaderboard.md](claw-eval-leaderboard.md). Global keep surface: [pareto-leaderboard.md](pareto-leaderboard.md).

## Ranked (best per model/quant, fair 10-task)

| Rank | Model | coding | LCB | HE | MBPP | BC | ctx | Note |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | ---: | :--- |
| 1 | `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M` | **0.6400** | 0.50 | 0.90 | 0.90 | 0.10 | 131k | Historical KEEP |
| 2 | `POCKET-35B-Q3_K_M` | **0.6150** | 0.50 | 0.80 | 0.90 | 0.10 | **65k** | [2026-07-26](../sessions/2026-07-26-pocket-35b-pipeline.md); claw-full **0.6667** |
| 3 | `Qwythos-9B-v2-Q4_K_M` | **0.6150** | — | — | — | — | **32k** | claw-full 0.3333 |
| 4 | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | **0.5900** | 0.40 | 1.00 | 0.80 | 0.00 | **65k** | [2026-07-27](../sessions/2026-07-27-incomplete-vectors-pareto.md); claw-full **0.1333** |
| 5 | `Ornith-1.0-35B-UD-Q4_K_XL` | **0.5800** | 0.40 | 0.90 | 0.80 | 0.10 | **65k** | same session; claw-full **0.6000** |
| 5 | `Ornith-1.0-9B-MTP-Q4_K_M` | **0.5800** | 0.40 | 0.80 | 0.90 | 0.10 | **32k** | MTP n=4; claw-full 0.4667 |
| 7 | `ornith-1.0-9b-Q4_K_M` (legacy) | **0.5800** | 0.40 | 0.80 | 0.90 | 0.10 | 131k | Older filename/quant |
| 8 | `Ornith-1.0-9B-UD-Q4_K_XL` | **0.5700** | 0.40 | 0.90 | 0.70 | 0.20 | **32k** | Alias `ornith-9b`; LCB patched† |
| 9 | `gemma-4-12B-it-qat-UD-Q4_K_XL` | **0.5650** | 0.40 | 0.80 | 0.90 | 0.00 | 131k | Historical KEEP; GGUF may be gone |
| 10 | `Ornith-1.0-35B-UD-Q3_K_XL` | **0.5550** | 0.40 | 0.90 | 0.70 | 0.10 | **65k** | Q3 A/B vs Q4; claw-full **0.4667** |
| 10 | `gemma-4-E4B-it-qat-UD-Q4_K_XL` | **0.5550** | — | — | — | — | **65k** | draft-mtp; claw-full 0.3333 |
| 12 | `POCKET-26B-Q4_K_M` | **0.4900** | 0.50 | 0.60 | 0.60 | 0.10 | **65k** | GGUF **deleted**; claw-full 0.2000 |
| 13 | `Bonsai-27B-Q1_0` | **0.4550** | 0.40 | 0.40 | 0.80 | 0.10 | **65k** | [vs POCKET](../sessions/2026-07-26-bonsai-coding-vs-pocket.md) |
| 14 | `LFM2.5-1.2B-Instruct-Q8_0` | **0.3500** | 0.10 | 0.50 | 0.70 | 0.10 | 65k f16 | LCB patched† |
| 15 | `Laguna-XS-2.1-Q3_K_XL` | **0.1950** | 0.20 | 0.20 | 0.30 | 0.00 | 65k | Best claw-full; weak coding |

† HE/MBPP/BC from original coding-10 Trial; LCB re-measured via `scripts/lcb_only.py` after Windows cache fix — see [session](../sessions/2026-07-24-lcb-patch-gambiarra.md).

### Rejected (no coding vector)

| Model | Attempts | Note |
| :--- | :--- | :--- |
| `Qwen3.5-9B-UD-Q4_K_XL` | 32k+MTP; 65k no MTP | VRAM kill mid-HE both; claw-full 0.1333 only |

## Operational lessons

1. **Dense coding can VRAM-kill at agentic ctx.** Ornith UD claw-full OK @ 65k; coding @ 65k hit 7950 MB → lock coding Baseline to **32k** on this rig. Qwen3.5-9B: even **65k no MTP** and **32k+MTP** kill mid coding-10.
2. **LCB cache must be a real file on Windows.** Symlink → `WinError 1314`. Harness now `shutil.copy2`.
3. **Sandbox timeout 30s** on generated-code subprocess (infinite loops hang otherwise).
4. **Agentic ≠ coding.** Laguna claw-full **0.6667** / coding **0.195**. Gemma-26B coding **0.59** / claw **0.13**.
5. **Quants are separate Trials.** Ornith-35B Q3 vs Q4 both measured — Q4 wins quality.
6. CLI: edit `config.py`, then  
   `.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 …"`  
   LCB-only remeasure: `.\venv\Scripts\python.exe scripts\lcb_only.py`

## Prefer by job (this rig)

| Job | Prefer |
| :--- | :--- |
| Agentic / tools | **POCKET-35B** / `laguna-xs` → `lfm2.5-1.2b` / `ornith-9b` |
| Direct coding preflight | Mythos → **POCKET-35B** → Ornith-9B/35B-Q4 @ fit ctx → LFM only if need speed |
| Balanced (agentic + coding) | **POCKET-35B** (claw 0.67 + coding 0.62) |
| Day supervised | **Ornith-9B-MTP** ([pareto](pareto-leaderboard.md) / ADR 0008); UD if `DAY_IQ_RATIO=0.8` |

## See also

* [pareto-leaderboard.md](pareto-leaderboard.md) — global frontier + Day/Night  
* [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI  
* [claw-eval-leaderboard.md](claw-eval-leaderboard.md) — agentic ranks  
* Sessions: [coding-10](../sessions/2026-07-24-coding-10-claw-leaders.md), [2026-07-27](../sessions/2026-07-27-incomplete-vectors-pareto.md)
