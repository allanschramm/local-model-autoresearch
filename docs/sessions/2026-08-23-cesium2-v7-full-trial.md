# 2026-08-23 — Cesium2-v7 Q8_0 @65536 Full Trial — MODEL_REJECTED coding 0.0 (8 GB-class)

## Goal
Complete Objective Vector for `ram1234598766/Cesium2-v7-GGUF` Q8_0 1.6G at 65536 q4_0 — small text-gen from `skip=40` 04:41.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 2121/7676 ok
- `block_count 28` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/ram1234598766/Cesium2-v7-GGUF/cesium2-v7-q8_0.gguf` (hf 55s, Q8 1.6G)
- Baseline: `MODEL cesium2-v7-q8_0.gguf / CTX 65536 / q4_0 / 0.6/0.95/20` — same as validation `bf2b6d2`

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial Cesium2-v7 Q8 @65536 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `109.0 t/s` (capped 4096) — same as validation 112.0
- **Coding:** HE 0/10 0.00 @141.1, MBPP 0/10 0.00 @140.9, LCB 0/10 0.00 @155.8, BigCode 0/10 0.00 @168.0 → **0.0000** combined — **all FAIL**, 7-8 no code extracted. TSV `rejected` `MODEL_REJECTED` `Coding preflight failed`. No agentic run (coding gate fails first).
- **Agentic quick from validation:** `0/5 0.0000` in 154s — same 0.00 pattern (length_stops + HTTP 500). Full trial never reached agentic due to coding gate.
- **Status:** `rejected` — `MODEL_REJECTED` — coding 0.0 is hard reject (vs Qwen 0.475, Smol 0.365).
- **TSV:** `8fa08f69-5cf6-4960-bd24-d3fe912ce299` @ `bf2b6d2` `agentic-full` `rejected` `0.0000` `109.0` `q4_0 65536` — `rejected` not `dominated`/`on_front`.
- **Time:** 451s (~7.5 min) — failed at coding preflight, no agentic 15.

## Errors / Corrections
- Coding 0/10 on all 40 tasks — model not instruction-tuned for coding (no code extracted, peg-native format error in quick).

## Decisions
- **Mark `Cesium2-v7` as `rejected` — do not retry** — 0.00 coding is zero-value (like maple/Ornith FC). Keep file for now but flag for purge (1.6G tiny, D: still 19.7G >10G, but next download will need space).
- Baseline stays Cesium for this trial only — revert to winning `Qwen3.8-4B-Distill` Q4_K_M @131072 for next.

## Open questions
- None — 0.00 coding is terminal for this rig's coding profile.

## References
- HF: `ram1234598766/Cesium2-v7-GGUF` (Q8 1.6G dry-run 2026-08-23)
- Sessions: `2026-08-23-cesium2-v7-validation.md` (0.00 quick), `2026-08-23-qwen9b-abliterated-full-trial.md` (dominated 0.475)
- Logs: `llama-server-20260823-071610-cesium2-v7-q8_0.log`
- TSV: `results.tsv` `8fa08f69...` `rejected`

## Verification
- Measured: bench 109.0, coding per-task 0/10, TSV `rejected`.
