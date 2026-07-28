# 2026-07-27 — Incomplete Objective Vectors + Pareto + Day rule (ADR 0008)

## Goal

1. Build global Pareto ranking from `results.tsv` (ctx × TPS × agentic × coding).
2. Fix bad Day picks (max TPS → LFM 8B; speed-band → LFM 1.2B half-IQ).
3. Land **ADR 0008**: Day = IQ ε-band then max TPS; Night maximin unchanged.
4. Complete incomplete Objective Vectors; document failures.
5. Treat distinct quants as separate Trials (Q3 ≠ Q4).

## Hardware

RTX 4060 **8 GB**, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA.

## Decisions

* **Day ≠ max TPS** (ADR 0006 failure) → ADR 0007 speed-band (commit `40322f4`) → still half-IQ → **ADR 0008**: `min ≥ DAY_IQ_RATIO × IQ_best` then max TPS (default ratio **0.75**).
* **Night unchanged:** `CTX ≥ NIGHT_CTX_FLOOR` then max `min(agentic, coding)`.
* **Quantizations are not duplicates.** Ornith-35B Q3_K_XL and Q4_K_XL both get full pipelines.
* **Qwen3.5-9B coding** closed as **failure** after VRAM kills (no further retries on this rig).

## Commands (reproducible)

```powershell
# config.py Baseline then:
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "…"
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "…"
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "…"
```

Sequential pipeline example (Ornith-35B Q3): validation → claw-full → coding-10, one process at a time.

## Findings

### Incomplete → complete

| Model | Fingerprint | agentic | coding | TPS (claw / coding) | VRAM |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `Ornith-1.0-35B-UD-Q4_K_XL` | 65k, n-cpu-moe 40, TEMP 0.4 | 0.6000 | **0.5800** | 25.7 / 32.0 | 5.1 GB |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` | 65k, n-cpu-moe 30, mtp2, TEMP 1.0 | 0.1333 | **0.5900** | 29.2 / 33.5 | 5.2 GB |
| `Ornith-1.0-9B-MTP-Q4_K_M` | 32k, MTP n=4, TEMP 0.4 | 0.4667 | **0.5800** | 63.7 / 86.7 | 7.4 GB |

### Ornith-35B Q3_K_XL (full pipeline, separate quant)

| Step | Score | TPS | Peak VRAM |
| :--- | ---: | ---: | ---: |
| validation (claw-quick) | 0.6000 | 25.9 | 4.4 GB |
| claw-full | **0.4667** | 25.2 | 4.5 GB |
| coding-10 | **0.5550** | 29.5 | 4.6 GB |

vs Q4_K_XL: worse agentic (0.47 vs 0.60), slightly worse coding (0.555 vs 0.580), lower VRAM / smaller file. **Prefer Q4 for quality.**

### Qwen3.5-9B coding — rejected

| Attempt | Outcome |
| :--- | :--- |
| 32k + MTP n=4 | VRAM kill **7913 > 7900** mid HumanEval |
| 65k, no MTP | VRAM kill **7931 > 7900** mid HumanEval (preflight est 7584 MB) |

Claw-full @ 32k+MTP still fits (~7.5 GB) because tool calls are short; long codegen does not. Agent **0.1333**. Coding axis incomplete (VRAM rejected).

### Pareto lenses (ADR 0008)

* **Night:** POCKET-35B (`min` 0.615).
* **Day @ 0.75:** Ornith-9B-MTP (~64 t/s, `min` ≈ 0.47). LFM 1.2B out (coding min 0.35).
* **Day @ 0.8:** Ornith-9B-UD (~42 t/s, `min` ≈ 0.57).

See [pareto-leaderboard.md](../discovery/pareto-leaderboard.md), [pareto-selection.md](../discovery/pareto-selection.md).

## Errors

* Day = max TPS selected LFM 8B → ADR 0007.
* Day = speed-band then IQ selected LFM 1.2B (~half Night IQ) → ADR 0008 (IQ ε-band then max TPS).
* Dense Qwen3.5-9B: historical “128k ran” was **without MTP** / light Autoloop load; not comparable to coding-10.
* Preflight VRAM underestimate on dense 65k coding (7584 est → 7931 peak).

## Docs updated

* ADR 0007 then **0008** + CONTEXT / AGENTS Day wording  
* Cards: `ornith-1.0-35b`, `ornith-1.0-9b`, `gemma-4-26b-a4b`, `qwen3.5-9b`  
* Leaderboards: claw, coding, **pareto** + selection method note  

## Follow-ups

* Wire `DAY_IQ_RATIO` into any future automated Day picker (docs-first today).
