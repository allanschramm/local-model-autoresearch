# 2026-08-23 — Qwen3.8-4B-Heretic Q4_K_M Validation @131072 — PASS 1.0 (8 GB-class)

## Goal
Validate NEW `yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF` model-Q4_K_M 2.8G at 131072 q4_0 — heretic abliterated variant of winning Qwen distill (created 2026-08-23T06:23).

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 5326/7676 ok
- `block_count 33` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF/model-Q4_K_M.gguf` (hf 70s, 2.8G, mmproj 672M ignored)
- Baseline: `MODEL model-Q4_K_M.gguf / CTX 131072 / q4_0 / 0.6/0.95/20` — same arch as `empero-ai` distill (Qwen3.5-4B Gated DeltaNet)
- Quant comparable to base Q4_K_M (advisory: Q4 vs Q2 confound avoided — largest fit Q4 chosen)

## Commands
```powershell
hf download yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF model-Q4_K_M.gguf --local-dir models/yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Qwen3.8-4B-Heretic Q4_K_M @131072 q4_0"
```

## Findings
- **Bench:** `74.8 t/s` (capped 4096) — **PASS** (>20, >50 Day), essentially identical to base `empero-ai` 74.9
- **Agentic quick 5 tasks:** `5/5 1.0000` in 162s — T002 0.50, T004 0.65, T006 1.00, T008 1.00, T010 1.00 — vs base T004 0.80, similar. Log `agentic-20260823-053507-model-Q4_K_M.json`. Peak **5.5G** VRAM (same as base 5.5G)
- **Status:** `incomplete` (validation profile) — coding 0.0, score 1.0000
- **TSV:** `f73d0e50-0ce5-4e02-8062-6b2862c0a893` @ `a40e2ba` `validation` `incomplete` `1.0000` `74.8` `74.8` `q4_0 131072 8/8 512/128 True/on/False` — `model-Q4_K_M.gguf` (yachen4ever)
- **Note:** Heretic abliteration does not degrade bench/agentic quick vs base.

## Errors / Corrections
- None — load succeeds.

## Decisions
- Validation **PASS** — run full trial (`--agentic-full --include-coding`) on same Fingerprint to test whether heretic improves `min(coding,agentic)` vs base `0.64/0.87`.

## Open questions
- **TBD:** Full vector on same Fingerprint.

## References
- HF: `yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF` (dry-run 2026-08-23: Q4 2.8G + mmproj)
- Sessions: `2026-08-23-qwen38-4b-distill-full-trial.md` (base on_front 0.64/0.87), `2026-08-23-new-models-api-exhaustive.md` (found at skip=20)
- Logs: `llama-server-20260823-053222-model-Q4_K_M.log`

## Verification
- Measured: bench 74.8, agentic JSON 1.0, NVML 5.5G, TSV row.
