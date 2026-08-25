# 2026-08-23 — Ornith-1.5-35B Heretic MTP APEX I-Mini @65536 q4_0 Validation — PASS 0.8 (8 GB-class)

## Goal
Validate NEW `ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF` I-Mini 14.3G (created 2026-08-23T06:28) at 65536 q4_0 on 8 GB-class rig (MoE 41/41 offload).

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 4846/7676 ok, host 14647/27790 ok
- `block_count 41` auto `n-cpu-moe 41`, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf` (hf 294s, 14.3G)
- Baseline: `MODEL Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf / CTX 65536 / q4_0 / 0.6/0.95/20 / NO_MMAP False`
- This is the newest text-gen GGUF in API window after Smol (06:28 vs 06:25)

## Commands
```powershell
hf download ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf --local-dir models/ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Ornith-1.5-35B-Heretic-MTP-APEX Mini @65536 q4_0"
```

## Findings
- **Bench:** `34.7 t/s` (capped 4096 bench ctx) — **PASS** (>20, >50? No — 34.7 <50 Day floor, so Day not eligible but Night eligible at 65K)
- **Agentic quick 5 tasks:** `4/5 0.8000` in 441s — **PASS:** T002 0.50, T004 1.00, T010 1.00, T008 0.80 (length_stops 1), **FAIL:** T006 0.20 (length_stops 1, 75 chars, no draft). Log `agentic-20260823-034854-Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.json`. Peak **3.0 GB** VRAM (lowest yet — MoE offload 41 active).
- **Status:** `incomplete` (validation profile) — coding 0.0, score 0.8000
- **TSV:** `b00aba6f-d5ab-4847-8f15-ae24b89fdf93` @ `6a7bbb3` `validation` `incomplete` `0.8000` `34.7` `34.7` `q4_0 65536 8/8 512/128 True/on/False` — MoE streaming `n-cpu-moe 41`.
- **Note:** T006 length_stops suggests `max_tokens` cap hit on long draft (same as 2026-08-20 4096 rerun). T008 also length_stops but still PASS.

## Errors / Corrections
- None — load succeeds (vs maple unknown arch, vs Ornith FC blk.32 missing). This is the first 35B that loads on b10549 among the NEW 35B variants.

## Decisions
- Validation **PASS on IQ** (0.8) but **TPS 34.7 <50 Day floor** — still run full trial per instruction (`--agentic-full --include-coding`) to complete vector for Night/support. Expect dominated on Day but possible Night `on_front` if coding strong.
- Keep the 14.3G file.

## Open questions
- **TBD:** Full vector (Claw-full 15 + coding-10) on same Fingerprint — falsifiable via next run.

## References
- HF: `ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF` (dry-run 2026-08-23: Mini 14.3G, Compact 17.3G, Quality 23.5G, Balanced 26.1G)
- API: `filter=text-generation&sort=createdAt` top hit 06:28
- Logs: `llama-server-20260823-034130-Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.log`
- Sessions: `2026-08-23-new-models-api-exhaustive.md`

## Verification
- Measured: bench 34.7, agentic JSON 0.8000, NVML 3.0G, TSV row.
