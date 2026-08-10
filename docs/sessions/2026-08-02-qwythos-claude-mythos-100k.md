# Qwythos Claude-Mythos at 100k

## Goal

Complete the throughput, coding-10, and Claw full Objective Vector for `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` at 100000 context for the Qwythos comparison.

## Hardware

- NVIDIA 8 GB-class discrete NVIDIA with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

## Setup

- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP disabled, per model-card recommendation.
- Batch/ubatch: `16/8`.
- Sampler: temperature `0.6`, top-p `0.95`, top-k `20`, repeat penalty `1.05`.

## Commands

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc full-qwythos-claude-mythos-tqp030-100k-turbo2-b16
```

## Findings

Trial `94b858fd-7c80-4627-b10f-125d39f19de3` completed successfully:

| Metric | Result |
|---|---:|
| Backend bench | 42.1 t/s |
| Coding generation | 47.9 t/s |
| LiveCodeBench | 0.5000 |
| HumanEval+ | 0.4000 |
| MBPP+ | 0.7000 |
| BigCodeBench Hard | 0.0000 |
| Coding composite | 0.4500 |
| Claw full | 0.4000 (6/15) |
| Peak VRAM | 7.3 GB |

## Errors

- Some individual Claw tasks returned HTTP 500 after malformed or unsupported model requests; the adapter recorded those tasks as failures and completed all 15 tasks.

## Decisions

- Preserve this complete 100k point for comparison with both Qwythos v2 artifacts.
- Normal v2 and Claude-Mythos are batch-matched at `16/8`; their measured TPS is directly comparable under this setup.
- The complete MTP point uses `32/16`. Its strict `16/8` Trial crossed the physical-VRAM gate, so report MTP's higher TPS as a best-feasible result, not a batch-matched speedup.
