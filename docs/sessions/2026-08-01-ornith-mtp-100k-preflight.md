# Ornith 9B MTP at the 100k context floor

## Goal

Run the full throughput, coding-10, and Claw full pipeline for the embedded-MTP Ornith 9B artifact at the operator's minimum 100000-token context.

## Hardware

- Discrete NVIDIA GPU with 8 GB physical VRAM.
- Physical-VRAM Trial limit: 7900 MB.

## Setup

- GGUF: `Ornith-1.0-9B-MTP-Q4_K_M.gguf`
- Context / KV: `100000 / q4_0`
- MTP: `draft-mtp`, draft max `4`
- Batch / ubatch: `256 / 128`; threads `6 / 8`; mmap disabled
- Model-card sampler: temperature `0.4`, top-p `0.95`, top-k `20`, min-p `0`, repeat `1.0`

## Commands

```powershell
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --agentic-full --desc full-ornith-9b-mtp-100k-q4kv-n4
```

## Findings

The preflight estimated `8000 MB`, above the configured `7900 MB` physical-VRAM limit. The harness returned `MODEL_REJECTED` before starting `llama-cli` or `llama-server`. Trial ID: `05198702-332a-436f-9762-218b7997977f`.

## Errors

No infrastructure or model-runtime error occurred. This was an intentional hard-gate rejection.

## Decisions

- Do not lower context below the operator's 100000-token floor.
- Do not raise this fingerprint to 114688 or 131072.
- Do not substitute unsupported `turbo3`; this upstream build supports `q4_0` for the selected compressed KV path.
