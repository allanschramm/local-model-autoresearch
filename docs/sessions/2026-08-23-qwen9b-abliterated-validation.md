# 2026-08-23 — Qwen3.8-9B Abliterated IQ4_XS @65536 — PASS 1.0 (8 GB-class)

## Goal
Validate NEW `nuofang/Qwen3.8-9B-abliterated-25-GGUF` IQ4_XS-no-mtp 5.2G at 65536 q4_0 — abliterated 9B variant (created 2026-08-23T04:44).

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 6403/7676 ok
- `block_count 32` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/nuofang/Qwen3.8-9B-abliterated-25-GGUF/Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf` (hf 109s, 5.2G + imatrix, mmproj 921M)
- Baseline: `MODEL Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf / CTX 65536 / q4_0 / 0.6/0.95/20`
- Found via `filter=gguf&skip=40` (04:44) after exhaustive sweep — next small text-gen after Heretic 4B

## Commands
```powershell
hf download nuofang/Qwen3.8-9B-abliterated-25-GGUF Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf --local-dir models/nuofang/Qwen3.8-9B-abliterated-25-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Qwen3.8-9B abliterated IQ4_XS @65536 q4_0"
```

## Findings
- **Bench:** `48.6 t/s` (capped 4096) — **PASS** (>20, <50 Day floor)
- **Agentic quick 5 tasks:** `5/5 1.0000` in 229s — T002 0.50, T004 1.00, T006 1.00, T008 1.00, T010 1.00 — vs Heretic 0.65 on T004, so slightly better. Log `agentic-20260823-062056-Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.json`. Peak **6.4 GB** VRAM (highest among 9B validations due to IQ4_XS + KV).
- **Status:** `incomplete` (validation profile) — coding 0.0, score 1.0000
- **TSV:** `6de4dfe1-98a8-4994-b270-3a1f56f8a746` @ `d297eb0` `validation` `incomplete` `1.0000` `48.6` `48.6` `q4_0 65536 8/8 512/128 True/on/False` — `Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf`
- **Note:** 9B IQ4_XS 5.2G + KV 6.4G peak fits, but 48.6 TPS < Day 50 and < Qwen 4B 74.9 — slower per token due to 9B vs 4B.

## Errors / Corrections
- None — load succeeds.

## Decisions
- Validation **PASS on IQ** — run full trial (`--agentic-full --include-coding`) on same Fingerprint to complete vector (expected `on_front` at 65K vs Smol 0.365).

## Open questions
- **TBD:** Full vector (Claw-full + coding-10) on same Fingerprint.

## References
- HF: `nuofang/Qwen3.8-9B-abliterated-25-GGUF` (dry-run 2026-08-23: IQ4_XS 5.2G)
- API: `filter=gguf&skip=40` 04:44 hit
- Logs: `llama-server-20260823-061703-Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.log`
- Sessions: `2026-08-23-qwen-heretic-full-trial.md` (0.6667/0.64)

## Verification
- Measured: bench 48.6, agentic JSON 1.0, NVML 6.4G, TSV row.
