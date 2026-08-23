# 2026-08-23 — Qwen3.8-9B Abliterated IQ4_XS @65536 Full Trial — dominated (8 GB-class)

## Goal
Complete Objective Vector for NEW `nuofang/Qwen3.8-9B-abliterated-25-GGUF` IQ4_XS 5.2G at 65536 q4_0 — 9B abliterated variant.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 6403/7676 ok
- `block_count 32` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/nuofang/Qwen3.8-9B-abliterated-25-GGUF/Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf` (hf 109s)
- Baseline: `MODEL Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.gguf / CTX 65536 / q4_0 / 0.6/0.95/20` — same as validation `f0e704f`

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial Qwen3.8-9B abliterated IQ4_XS @65536 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `48.7 t/s` (capped 4096) — <50 Day floor, same as validation 48.6
- **Coding:** HE 4/10 0.40 @66.0, MBPP 8/10 0.80 @58.7, LCB 5/10 0.50 @65.9, BigCode 0/10 0.00 @61.4 → **0.4750** combined — trails Qwen 4B distill 0.64 (-26%) and Heretic 0.64 equal, ahead of Smol 0.365.
- **Agentic quick co-run:** `5/5 1.0000` in 261s — T002 0.50, T004 1.00, T006 1.00, T008 1.00, T010 1.00
- **Agentic full:** `13/15 0.8667` in 1887s — **passes:** T002 0.50, T004 1.00, T006 1.00, T008 1.00, T010 1.00, T012 1.00, T014 0.70, T016 0.50, T018 1.00, T044 1.00, T046 1.00 (13218 length_stops 1 but PASS), T048 1.00, T050 1.00 — **fails:** T053 0.20, T054 0.00 (HTTP 400 66919 >65536 ctx exceed). Wrote `agentic-20260823-070846-Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.json`. Same 0.8667 as Qwen distill/heretic 35B Mini — finance web_real cluster again.
- **Combined TPS:** `63.4` (bench 48.7)
- **VRAM:** peak `6.5 GB` (largest among 9B validations due to IQ4_XS)
- **TSV:** `84a4fda8-f758-481b-a94e-ecd9e1964b3f` @ `f0e704f` `agentic-full` `dominated` `0.8667 / 0.4750` `63.4 / 48.7` `q4_0 65536 8/8 512/128 True/on/False` — **dominated** (log said `on_front` before global recompute, but TSV final is `dominated` vs Qwen 4B distill 131K: coding 0.475 <0.64, TPS 63.4 <94.2, same agentic).
- **Pareto:** Dominated by `Qwen3.8-4B-Distill` @131K — not Day/Night pick.
- **Time:** 2840s (~47 min)

## Errors / Corrections
- T054 ctx exceed 66919 >65536 at 65K — same as Heretic at 131K but now at 65K for 9B — 9B's longer reasoning trace exceeds 65K.

## Decisions
- **Keep** 5.2G file for now (D: ~17G → ~12G after download, still >10G guard) — dominated but documented as NEW abliterated 9B not superior to 4B distill.
- Baseline stays Qwen 9B for this trial only — revert to winning `Qwen3.8-4B-Distill` Q4_K_M @131072 for next.

## Open questions
- **TBD:** Same 9B at `CTX 131072` — would it avoid T054 exceed and improve coding? But VRAM at 131K would be >7G, likely near limit.

## References
- HF: `nuofang/Qwen3.8-9B-abliterated-25-GGUF` (IQ4_XS 5.2G, 04:44)
- Sessions: `2026-08-23-qwen9b-abliterated-validation.md` (1.0), `2026-08-23-qwen38-4b-distill-full-trial.md` (base winner)
- Logs: `llama-server-20260823-062254-Qwen3.8-9B-abliterated-25-IQ4_XS-no-mtp.log`
- TSV: `results.tsv` `84a4fda8...` `dominated`

## Verification
- Measured: bench 48.7, coding per-task, agentic JSON 0.8667, NVML 6.5G, TSV `dominated`.
