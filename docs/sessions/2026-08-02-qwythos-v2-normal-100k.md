# Qwythos v2 normal at 100k

## Goal

Complete the throughput, coding-10, and Claw full Objective Vector for the no-MTP Qwythos v2 artifact at 100000 context.

## Hardware

- NVIDIA RTX 4060 with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

## Setup

- Model: `Qwythos-9B-v2-Q4_K_M.gguf`.
- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP disabled.
- Batch/ubatch: `16/8`.
- Sampler: temperature `0.6`, top-p `0.95`, top-k `20`, repeat penalty `1.05`.

## Commands

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc full-qwythos-v2-normal-tqp030-100k-turbo2-b16
```

## Findings

Trial `5c41394f-032c-4fc3-b8d5-e0db1ef3a36a` completed successfully:

| Metric | Result |
|---|---:|
| Backend bench | 41.6 t/s |
| Coding generation | 47.3 t/s |
| LiveCodeBench | 0.4000 |
| HumanEval+ | 0.7000 |
| MBPP+ | 0.7000 |
| BigCodeBench Hard | 0.0000 |
| Coding composite | 0.4900 |
| Claw full | 0.4667 (7/15) |
| Peak VRAM | 7.2 GB |

## Errors

- Batch/ubatch `32/16` crossed the physical-VRAM gate at 7902 MB in the preceding Trial.

## Decisions

- Keep `16/8` as the complete no-MTP 100k point.
- Do not merge this row with the embedded-MTP point; they are distinct model basenames and Fingerprints.
- A strict MTP speed A/B must use matching batch/ubatch in addition to matching context, sampler, KV, and runtime.
