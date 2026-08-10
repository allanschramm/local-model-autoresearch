# 2026-07-26 — POCKET-26B Q4_K_M full pipeline

## Goal

Champion path for `POCKET-26B-Q4_K_M.gguf` (Gemma4-26B-A4B Korean-tuned quant): validation → claw-full → coding-10.

## Hardware

discrete 8 GB-class NVIDIA, `VRAM_LIMIT_MB=7900`, upstream CUDA.

## Setup

```python
MODEL = 'POCKET-26B-Q4_K_M.gguf'
CTX_SIZE = 65536
N_CPU_MOE = None  # → 30
TEMP = 1.0
TOP_P = 0.95
TOP_K = 64  # from GGUF sampling
SPEC_TYPE = None
```

## Commands

```powershell
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate pocket-26b-q4_k_m pipeline"
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --no-agentic-quick --desc "claw-full pocket-26b-q4_k_m"
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 pocket-26b-q4_k_m"
```

## Findings

| Stage | Result |
|---|---|
| Validation | claw-quick **0.4000**, tg **21.8**, VRAM **4.5 GB** |
| Claw full | **0.2000** (3/15), tg **20.5**, VRAM **4.5 GB** |
| Coding-10 | **0.4900** (HE 0.60 / MBPP 0.60 / LCB 0.50 / BC 0.10), tg 20.4, VRAM 4.7 GB |

### vs siblings on the operator host

| Model | claw-full | coding | tg |
|---|---:|---:|---:|
| **POCKET-35B Q3_K_M** | **0.6667** | **0.615** | ~36 |
| POCKET-26B Q4_K_M | 0.2000 | 0.490 | ~21 |
| gemma-4-26b-a4b (Unsloth + MTP) | 0.1333 | — | ~29 |

Korean-tuned Gemma POCKET-26B beats Unsloth sibling on agentic slightly, still far behind POCKET-35B. Coding mid-pack.

## Decisions

- Prefer **POCKET-35B** over POCKET-26B for agentic + coding (scores above).
