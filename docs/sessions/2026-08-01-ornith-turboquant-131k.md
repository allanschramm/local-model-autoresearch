# Ornith on TurboQuant+ at 131k

## Goal

Run the complete Ornith pipeline at 131072 context without crossing the 7900 MB physical-VRAM gate, changing only runtime and sampler Baseline settings.

## Hardware

- NVIDIA 8 GB-class discrete NVIDIA, 8 GB physical VRAM
- VRAM gate: 7900 MB

## Setup

- Model: `Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Runtime: TurboQuant+ `tqp-v0.3.0`, Windows x64 CUDA 12.4 release
- Context: 131072
- KV: Turbo2/Turbo2
- Batch/ubatch: 64/32
- Sampler: temperature 0.4, top-p 0.95, top-k 20, min-p 0, repeat penalty 1

## Commands

The mutable Baseline was set in `autoresearch/core/config.py` before each Trial. The complete pipeline used:

```powershell
$env:AUTORESEARCH_LLAMA_CPP_ROOT = '<repo>\llama.cpp-releases\turboquant\tqp-v0.3.0'
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc 'full-ornith-9b-tqp030-131k-turbo2-nospec-b64'
```

## Findings

| Trial ID | Speculation | Bench | Peak VRAM | Result |
|---|---:|---:|---:|---|
| `8d1db64d-8df5-42e8-ba06-bf78a9f76ef2` | MTP n=2 | 62.3 t/s | 8090 MB | VRAM gate |
| `8881827c-de13-4d66-b7e1-201ae58d435f` | MTP n=1 | not recorded | — | native exit `0xC0000409` |
| `95a8213b-4fd0-4b06-8d16-fcb7b782dced` | disabled | 43.3 t/s | 7.2 GB | complete pipeline OK |

The successful Objective Vector measured combined 53.8 TPS, coding 0.5550 (HumanEval+ 0.8, MBPP+ 0.8, LiveCodeBench 0.4, BigCodeBench 0.1), and Claw full 0.3333 (5/15). Runtime was 1823 seconds.

## Errors

- MTP n=2 exceeded the gate by 190 MB.
- MTP n=1 is not a viable fallback in this release/model combination because `llama-cli` exits with `0xC0000409`.

## Decisions

- Keep the successful local Baseline at 131072, Turbo2/Turbo2, batch/ubatch 64/32, with speculative decoding disabled.
- The embedded-MTP GGUF can serve 131k safely, but MTP itself is not active in the successful Fingerprint.
