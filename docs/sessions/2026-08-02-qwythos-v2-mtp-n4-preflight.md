# Qwythos v2 MTP n=4 preflight at 100k

## Goal

Test whether increasing embedded MTP from n=2 to n=4 can enter a complete 100000-context Trial without crossing the physical-VRAM gate.

## Hardware

- NVIDIA 8 GB-class discrete NVIDIA with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

## Setup

- Model: `Qwythos-9B-v2-MTP-Q4_K_M.gguf`.
- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP: `draft-mtp`, `SPEC_DRAFT_N_MAX=4`.
- Batch/ubatch: `32/16`.
- All other engine and sampler values matched the successful n=2 Trial.

## Commands

The mutable Baseline was changed in `autoresearch/core/config.py` before invoking the standard full-pipeline command:

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc full-qwythos-v2-mtp-tqp030-100k-turbo2-draft-mtp4-b32
```

## Findings

Trial `99e8c650-9085-4c45-886d-7128453d45cc` was rejected before model startup:

```text
[vram-preflight] est=8248MB limit=7900MB ok=False
Evaluation failed: FAIL: VRAM_PREFLIGHT est=8248MB > limit=7900MB
```

No throughput or quality axis ran for n=4.

## Errors

- None beyond the expected physical-VRAM preflight rejection.

## Decisions

- Keep n=2 as the active 100k MTP Baseline.
- Skip n=6 because n=4 already exceeds the physical-VRAM limit.
