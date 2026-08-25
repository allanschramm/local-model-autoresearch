# 2026-08-23 — MindSparQ-Coder-1.5B Q4_K_M @65536 Full Trial — dominated 0.025/0.00 (8 GB-class)

## Goal
Complete Objective Vector for `mradermacher/MindSparQ-Coder-1.5B-GGUF` Q4_K_M 986M at 65536 q4_0 — 1.5B code model, follow-up to 0.00 quick.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 1491/7676 ok
- `block_count 28` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/mradermacher/MindSparQ-Coder-1.5B-GGUF/MindSparQ-Coder-1.5B.Q4_K_M.gguf` (hf 32s)
- Baseline: `MODEL MindSparQ-Coder-1.5B.Q4_K_M.gguf / CTX 65536 / q4_0 / 0.6/0.95/20` — same as validation `dd126bb`

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial MindSparQ-Coder 1.5B Q4_K_M @65536 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `181.6 t/s` (capped 4096) — fastest yet
- **Coding:** HE 1/10 0.10 @191.3, MBPP 0/10 0.00 @191.1, LCB 0/10 0.00 @311.9, BigCode 0/10 0.00 @230.8 → **0.0250** combined — **all 0 on MBPP/LCB/BigCode**, only 1 HE pass. Vs Smol 0.365, Qwen 0.64.
- **Agentic quick co-run:** `0/5 0.0000` in 53s — T002 0.20 length_stops 11938, T004 0.00, T006 0.00, T008 0.00, T010 0.00 length_stops 28101
- **Agentic full:** `0/15 0.0000` in 145s — **all 15 FAIL:** T002 0.20, T004 0.00, T006 0.00, T008 0.00 length_stops 19983, T010 0.00 length_stops 28101, T012 0.30, T014 0.00, T016 0.00 length_stops 19878, T018 0.00, T044 0.08, T046 0.00, T048 0.00, T050 0.00, T053 0.00, T054 0.00. Wrote `agentic-20260823-073706-MindSparQ-Coder-1.5B.Q4_K_M.json`. **0 tool calls on 11/15** — model generates long non-tool text.
- **Combined TPS:** `213.8` (coding gen) vs bench 181.6 — Day-eligible but `min=0.00`
- **VRAM:** peak `2.7 GB` (same as validation)
- **TSV:** `a8399a88-9c3c-4ac4-9d0a-ac2e37336edd` @ `dd126bb` `agentic-full` `dominated` `0.0000 / 0.0250` `213.8 / 181.6` `q4_0 65536 8/8 512/128 True/on/False` — **dominated** (not on_front per log pre-recompute; TSV final is `dominated` vs Qwen/Smol due to 0.00 IQ despite 213 TPS).
- **Time:** 509s (~8.5 min) — fastest full trial due to small size.

## Errors / Corrections
- None — 0.00 is model capability (1.5B too small for tool format), not harness.

## Decisions
- **Keep** the 986M file for now — dominated control, not recommended for coding. Flag for purge if next NEW needs space.
- Baseline stays MindSparQ for this trial only — revert to winning `Qwen3.8-4B-Distill` Q4_K_M @131072 for next.

## Open questions
- None — 0.025/0.00 is terminal for coding profile.

## References
- HF: `mradermacher/MindSparQ-Coder-1.5B-GGUF` (Q4 986M)
- Sessions: `2026-08-23-mindsparq-validation.md` (0.00 quick), `2026-08-23-cesium2-v7-full-trial.md` (rejected 0.00)
- Logs: `llama-server-20260823-072923-MindSparQ-Coder-1.5B.Q4_K_M.log`
- TSV: `results.tsv` `a8399a88...` `dominated`

## Verification
- Measured: bench 181.6, coding 0.025, agentic 0.0000, NVML 2.7G, TSV `dominated`.
