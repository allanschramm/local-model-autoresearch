# Qwythos v2 MTP matching-batch VRAM rejection

## Goal

Run a strict MTP versus no-MTP comparison at 100000 context using the no-MTP point's batch/ubatch `16/8`.

## Hardware

- NVIDIA 8 GB-class discrete NVIDIA with 8 GB physical VRAM.
- Physical-VRAM limit: 7900 MB.

## Setup

- Model: `Qwythos-9B-v2-MTP-Q4_K_M.gguf`.
- Runtime: TurboQuant+ `tqp-v0.3.0`.
- Context: 100000.
- KV: `turbo2/turbo2`.
- MTP: `draft-mtp`, n=2.
- Batch/ubatch: `16/8`.
- Sampler matched the complete no-MTP point.

## Commands

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = (Resolve-Path 'llama.cpp-releases\turboquant\tqp-v0.3.0').Path
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc full-qwythos-v2-mtp-tqp030-100k-turbo2-draft-mtp2-b16
```

## Findings

Trial `fc6edf7e-d8cd-4502-97ff-b21a50e1aaab` passed preflight and produced a backend bench of 53.4 t/s. During the second HumanEval task, measured VRAM reached 7923 MB and the hard gate killed the server.

Partial coding and agentic values after server termination are invalid.

## Errors

```text
[VRAM] LIMIT EXCEEDED used=7923MB > limit=7900MB
Evaluation failed: FAIL: VRAM_LIMIT_EXCEEDED
```

## Decisions

- Do not use this rejected Trial for the MTP quality comparison.
- Keep the complete MTP n=2 batch/ubatch `32/16` point as the active MTP Baseline.
- Treat the strict matching-batch A/B as unavailable until the rig has enough repeatable physical-VRAM headroom.
