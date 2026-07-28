# 2026-07-26 — Bonsai coding-10 vs POCKET

## Goal

Fair coding-10 on `Bonsai-27B-Q1_0` at the same ctx/sampler recipe as POCKET-35B Q3_K_M, to close the head-to-head Objective Vector.

## Hardware

RTX 4060 8 GB, `VRAM_LIMIT_MB=7900`, upstream CUDA.

## Setup

```python
MODEL = 'Bonsai-27B-Q1_0.gguf'
CTX_SIZE = 65536
KV_CACHE_K = KV_CACHE_V = 'q4_0'
N_CPU_MOE = None  # dense
THREADS_BATCH = 12
CONT_BATCHING = False
SPEC_TYPE = None
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
```

## Commands

```powershell
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 bonsai-q1_0 vs pocket"
```

## Findings

| Model | coding | HE | MBPP | LCB | BC | tg | VRAM | claw-full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **POCKET-35B Q3_K_M** | **0.6150** | 0.80 | 0.90 | 0.50 | 0.10 | 35.0 | 3.7 GB | **0.6667** |
| Bonsai-27B Q1_0 | 0.4550 | 0.40 | 0.80 | 0.40 | 0.10 | **41.7** | 7.5 GB | 0.4667 |

- POCKET +35% coding, +43% claw-full, ~half the VRAM.
- Bonsai +19% TPS only.
- Same BC floor (0.10). Gap is mostly HE (0.80 vs 0.40).

Harness `DISCARD` on Bonsai coding vs prior claw KEEP — ignore; coding axis separate.

## Decisions

- Prefer **POCKET-35B** for agentic + coding on this rig (higher claw + coding; lower peak VRAM).
- Bonsai remains a higher-TPS / lower-quality point on the same axes.
