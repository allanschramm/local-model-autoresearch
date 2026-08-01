# DAY candidates: full 131k pipelines

## Goal

Measure three DAY candidates with throughput, coding-10, and Claw-Eval full on the same 131072-context fingerprint.

## Hardware

- Discrete NVIDIA GPU with 8 GB physical VRAM.
- Host-memory preflight budget: 27790 MB.
- Official `llama.cpp` CUDA build.

## Setup

- `CTX_SIZE=131072`; KV K/V `q4_0`; batch / ubatch `1024 / 256`; threads `8 / 8`; flash attention on.
- Sampler: temperature `0.6`, top-p `0.95`, top-k `0`, min-p `0`, repeat `1.0`, presence `0`, frequency unset.
- Granite's official cards and generation configs published no sampler values; the operator explicitly authorized this universal fallback.

## Commands

```powershell
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc <trial-description>
```

The mutable Baseline was written to `autoresearch/core/config.py` before each invocation. Coding used exactly 10 tasks per dataset.

## Findings

| GGUF basename | Context | Bench tg | Combined TPS | Peak VRAM | HE | MBPP | LCB | BigCode | Coding | Claw full | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | 131072 | 79.2 | 90.6 | 5.6 GB | 0.90 | 0.80 | 0.30 | 0.10 | 0.5450 | 0.2667 | OK / keep |
| `granite-4.0-h-tiny-Q4_K_M.gguf` | 131072 | 51.2 | 79.9 | 3.6 GB | 0.40 | 0.90 | 0.40 | 0.00 | 0.4650 | 0.8000 | OK / keep |
| `granite-4.1-3b-Q4_K_M.gguf` | 131072 | 92.0 | 66.9 | 7.2 GB | 0.10 | 0.70 | 0.30 | 0.10 | 0.3200 | 0.6667 | OK / keep |

Granite 4.0 H Tiny resolved `N_CPU_MOE=None` to `--n-cpu-moe 40`. Granite 4.1 3B and Nemotron were dense and remained within physical VRAM.

## Errors

- Nemotron's first Claw full attempt failed when the web mock inherited the Windows legacy code page. The harness now starts mock services with `PYTHONUTF8=1`; focused runner tests passed `7/7`, and the retry completed.
- Granite 4.0's first full attempt ended after one coding request with `[Errno 22] Invalid argument`. The server was healthy; the launcher had closed the output handle. Redirecting stdout/stderr at process creation produced a complete retry without changing the fingerprint.

## Decisions

- Preserve failed infrastructure rows alongside successful measurements.
- Use OS process completion events for long detached Trials, then inspect logs once after exit.
- Recompute DAY selection only through `scripts/rank_results.py` after its full-Fingerprint merge bug is fixed.
