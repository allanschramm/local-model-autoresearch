# Session 2026-07-24 — Coding-10 queue (Laguna / LFM 1.2B / Ornith-9B)

## Goal

Run direct-coding preflight (exactly 10 tasks × HE+ / MBPP+ / LCB / BigCode Hard) on the three models preferred after claw-full ranking: Laguna-XS, LFM2.5-1.2B, Ornith-9B UD.

## Hardware

discrete 8 GB-class NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream `llama.cpp` CUDA.

## Commands

```powershell
# Baseline in autoresearch/core/config.py first, then:
$env:PYTHONUTF8=1; $env:PYTHONUNBUFFERED=1
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 …"
```

## Baselines

| Model | ctx | KV | Notes |
| :--- | ---: | :--- | :--- |
| `Laguna-XS-2.1-Q3_K_XL` | 65k | q4_0 | `N_CPU_MOE=40` |
| `LFM2.5-1.2B-Instruct-Q8_0` | 65k | f16 | dense |
| `Ornith-1.0-9B-UD-Q4_K_XL` | **32k** | q4_0 | 65k VRAM-kill mid-coding |

## First-pass scores (LCB broken — symlink WinError 1314)

| Model | coding | LCB | HE | MBPP | BC | bench_tg | VRAM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laguna | 0.125 | **0**† | 0.2 | 0.3 | 0.0 | 35.2 | 4.5 GB |
| LFM 1.2B | 0.315 | **0**† | 0.5 | 0.7 | 0.1 | 167.6 | 4.0 GB |
| Ornith 32k | 0.430 | **0**† | 0.9 | 0.7 | 0.2 | 38.4 | 7.4 GB |

† LCB download failed → recorded 0. Not model skill.

## Errors

1. **Ornith @ 65k:** `VRAM_LIMIT EXCEEDED used=7950MB > limit=7900MB` during HumanEval → fail row. Retry @ **32k** OK.
2. **LCB all three:** `WinError 1314` symlink. Fixed later via copy2 + [LCB patch session](./2026-07-24-lcb-patch-gambiarra.md).

## After LCB patch (same HE/MBPP/BC; LCB re-measured)

| Model | LCB | coding patched |
| :--- | ---: | ---: |
| Laguna | 0.20 | **0.195** |
| LFM 1.2B | 0.10 | **0.350** |
| Ornith 32k | 0.40 | **0.570** |

`results.tsv` trials patched in place with `lcb_patch=2026-07-24` in description.

## Decisions

* Laguna = agentic king, **weak** coding (0.195).
* LFM 1.2B = speed + claw 0.60; coding mid (0.350).
* Ornith-9B = best of this queue on coding (**0.570**); still below historical Mythos **0.640** @ 131k.
* Coding alias for Ornith dense on 8 GB: prefer **ctx ≤32k** (agentic alias stays 65k).

## See also

* [lcb-patch-gambiarra](./2026-07-24-lcb-patch-gambiarra.md)
* [claw-eval-leaderboard](../discovery/claw-eval-leaderboard.md)
* [coding-leaderboard](../discovery/coding-leaderboard.md)
