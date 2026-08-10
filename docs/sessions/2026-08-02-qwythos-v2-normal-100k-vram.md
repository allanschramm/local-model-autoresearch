# Qwythos v2 normal at 100k: VRAM rejection

## Goal

Complete a no-MTP Qwythos v2 Objective Vector at the same context, runtime, KV, sampler, and initial batch settings used by the embedded-MTP comparison point.

## Hardware

- NVIDIA 8 GB-class discrete NVIDIA with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

## Setup

- Model: `Qwythos-9B-v2-Q4_K_M.gguf`.
- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP disabled.
- Batch/ubatch: `32/16`.
- Sampler matched the Qwythos v2 MTP Trial.

## Commands

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc full-qwythos-v2-normal-tqp030-100k-turbo2-b32
```

## Findings

Trial `21a5b115-5620-46a1-803f-25cea29d0389` passed preflight and backend bench at 41.2 t/s. During HumanEval, measured VRAM reached 7902 MB and the dense-model hard gate stopped the server.

The partial benchmark scores after server termination are invalid and must not be used for model comparison.

## Errors

```text
[VRAM] LIMIT EXCEEDED used=7902MB > limit=7900MB
Evaluation failed: FAIL: VRAM_LIMIT_EXCEEDED
```

## Decisions

- Preserve the rejected row in `results.tsv`.
- Retry with batch/ubatch `16/8`, changing no other model, runtime, context, KV, or sampler setting.
