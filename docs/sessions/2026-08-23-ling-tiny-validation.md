# 2026-08-23 — Ling-3.0-tiny Q4_K_M Validation @65536 — PASS 0.80 (8 GB-class)

## Goal
Validate missed candidate `bartowski/Ling-3.0-tiny-GGUF` Q4_K_M 4.9G at 65536 q4_0 — inclusionAI MoE (block_count 24, `n-cpu-moe 24` auto), created 2026-08-18, found via bartowski author-feed poll after NEW-window-exhausted declaration.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 1687/7676 ok, host 5063/27790 ok
- MoE streaming `n-cpu-moe 24`, dense layers full GPU

## Setup
- File: `models/bartowski/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf` (hf 118s, Q4_K_M 4.9G)
- Baseline: `MODEL Ling-3.0-tiny-Q4_K_M.gguf / CTX 65536 / q4_0 / TEMP 0.6 TOP_P 0.95 TOP_K 20 / N_CPU_MOE auto`
- Why suitable: real base (inclusionAI), first-party quantizer, MIT, fits 8GB class, text-generation — unlike sub-2B micro-controls

## Commands
```powershell
hf download bartowski/Ling-3.0-tiny-GGUF Ling-3.0-tiny-Q4_K_M.gguf --local-dir models/bartowski/Ling-3.0-tiny-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Ling-3.0-tiny Q4_K_M @65536 q4_0"
```

## Findings
- **Bench:** `53.5 t/s` (capped 4096) — **PASS** (>20, **>50 Day floor**)
- **Agentic quick 5 tasks:** `4/5 0.8000` in 242s — T002 0.50, T004 0.80, T006 1.00, T008 0.15 length_stops 1, T010 1.00. Log `agentic-20260823-112233-Ling-3.0-tiny-Q4_K_M.json`. Peak **2.5 GB** VRAM — lowest of any validated model.
- **Status:** `incomplete` (validation profile) — score 0.8000
- **TSV:** `ed7b6627-0065-4191-8e68-0b79c64d91db` @ `f202b7b` `validation` `incomplete` `0.8000` `53.5` `53.5` `q4_0 65536` — `Ling-3.0-tiny-Q4_K_M.gguf`

## Errors / Corrections
- None — loads clean on b10549.

## Decisions
- Validation **PASS** → full trial per standing instruction (`--agentic-full --include-coding`) running.

## Open questions
- **TBD:** Full vector on same Fingerprint.

## References
- HF: `bartowski/Ling-3.0-tiny-GGUF` (dry-run: Q4_K_M 4.9G, Q5_K_M 5.7G; Flash variant 30G+ too large)
- Sessions: `2026-08-23-new-models-api-exhaustive.md`, `2026-08-23-qwen38-4b-distill-full-trial.md` (base winner 0.64/0.87)

## Verification
- Measured: bench 53.5, agentic JSON 0.8000, NVML 2.5G, TSV row.
