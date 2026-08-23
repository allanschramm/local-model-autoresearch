# 2026-08-23 — Ornith-1.5-35B Heretic MTP APEX I-Mini @65536 q4_0 Full Trial — on_front (8 GB-class)

## Goal
Complete Objective Vector for NEW `ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF` I-Mini 14.3G at 65536 q4_0 on same Fingerprint as validation — Claw-full + coding-10.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 4846/7676 ok, host 14647/27790 ok
- `block_count 41` auto `n-cpu-moe 41`, MoE streaming, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf` (hf 294s)
- Baseline: `MODEL Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf / CTX 65536 / q4_0 / 0.6/0.95/20` — same as validation `ecfa8df`
- Created 2026-08-23T06:28, newest text-gen GGUF in window (agentic-coding, MTP, APEX)

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial Ornith-1.5-35B-Heretic-MTP-APEX Mini @65536 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `34.9 t/s` (capped 4096) — <50 Day floor, Night eligible
- **Coding:** HE 6/10 0.60 @45.3, MBPP 9/10 0.90 @40.7, LCB 4/10 0.40 @45.0, BigCode 1/10 0.10 @41.2 → **0.5300** combined. Log `llama-server-20260823-035208-Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.log`.
- **Agentic quick co-run:** `5/5 1.0000` in 420s — T002 0.50, T004 1.00, T006 1.00, T008 0.80 length_stops 1, T010 1.00 — better than validation 0.80 (T006 now PASS)
- **Agentic full:** `13/15 0.8667` in 4097s — **passes:** T002 1.00, T004 1.00, T006 1.00, T008 0.80 length_stops 1, T010 1.00, T012 1.00, T014 0.70, T016 0.50, T018 1.00, T044 1.00, T046 1.00, T048 1.00, T050 1.00 — **fails:** T053 0.20 (golden share only), T054 0.00. Wrote `agentic-20260823-052551-Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.json`.
- **Combined TPS:** `42.9` (coding gen), bench 34.9
- **VRAM:** peak `3.1 GB` (MoE offload 41, lowest among 35B trials)
- **TSV:** `19046420-e535-49bf-a196-684fe3151602` @ `ecfa8df` `agentic-full` `on_front` `0.8667 / 0.5300` `42.9 / 34.9` `q4_0 65536 8/8 512/128 True/on/False` — **on_front** (Night ctx 65K, not dominated: Qwen 0.64/0.87 at 131K wins on IQ but 65K vs 131K is different ctx axis).
- **Pareto:** Joins front at 65K (Night). For Night profile (max `min` among ≥65536), `min=0.53` trails Qwen `0.64` at 131K — Qwen remains Night pick unless coding improves.

## Errors / Corrections
- T008 length_stops (12772 chars) but still 0.80 — same as validation. T053/T054 finance web_real cluster fails (same as Qwen/Smol).

## Decisions
- **Keep** 14.3G Mini on disk for now (D: ~20G free after download, still >10G guard). Its `on_front` at 65K is valid but not Night winner vs Qwen 131K.
- Baseline stays `Ornith-1.5-35B Heretic Mini @65536` for this trial only — next `autoloop` or validation should revert to Qwen winning point unless testing this family at larger ctx.

## Open questions
- **TBD:** Same model at `CTX 131072` or with `SPEC_TYPE draft-mtp` — would it beat Qwen on `min` while staying >20 TPS? Falsifiable via same coding+agentic on larger ctx.

## References
- HF: `ursb01/Ornith-1.5-35B-A3B-Heretic-MTP-APEX-GGUF` (Mini 14.3G)
- Sessions: `2026-08-23-ornith-35b-heretic-validation.md` (0.80), `2026-08-23-qwen38-4b-distill-full-trial.md` (Qwen on_front 0.64/0.87)
- Logs: `llama-server-20260823-035208-...` , `agentic-20260823-052551-...`
- TSV: `results.tsv` `19046420...`

## Verification
- Measured: bench 34.9, coding per-task, agentic JSON 0.8667, NVML 3.1G, TSV `on_front`.
