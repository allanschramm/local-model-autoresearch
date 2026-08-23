# 2026-08-23 — Ling-3.0-tiny Q4_K_M @65536 Full Trial — dominated (8 GB-class)

## Goal
Complete Objective Vector for missed candidate `bartowski/Ling-3.0-tiny-GGUF` Q4_K_M 4.9G at 65536 q4_0 — MoE (`n-cpu-moe 24` auto), found via bartowski author-feed poll.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 1687/7676 ok
- `block_count 24` MoE, dense layers full GPU, expert streaming

## Setup
- File: `models/bartowski/Ling-3.0-tiny-GGUF/Ling-3.0-tiny-Q4_K_M.gguf` (hf 118s)
- Baseline: `MODEL Ling-3.0-tiny-Q4_K_M.gguf / CTX 65536 / q4_0 / 0.6/0.95/20` — same as validation `f853a92`

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial Ling-3.0-tiny Q4_K_M @65536 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `52.8 t/s` — Day-eligible (≥50), same as validation 53.5
- **Coding:** HE 3/10 0.30 @58.7, MBPP 7/10 0.70 @60.5, LCB 4/10 0.40 @70.0, BigCode 0/10 0.00 @59.6 → **0.3900** combined — between Smol 0.365 and Qwen distill 0.64
- **Agentic quick co-run:** `5/5 1.0000` in 225s (T008 recovered vs validation 0.15 → 1.00)
- **Agentic full:** `13/15 0.8667` in 2070s — same score as Qwen distill and Ornith 35B Mini; fails T053/T054 finance web_real cluster only. Wrote `agentic-20260823-122216-Ling-3.0-tiny-Q4_K_M.json`
- **Combined TPS:** `62.0`, bench 52.8 — Day-eligible
- **VRAM:** peak `2.5 GB` — lowest of all full trials (MoE offload efficient)
- **TSV:** `e2a298fb-526c-47ec-96ab-29d03b4759d9` @ `f202b7b` `agentic-full` `dominated` `0.8667 / 0.3900` `62.0 / 52.8` `q4_0 65536` — log printed `on_front` pre-recompute, final TSV `dominated` by `Qwen3.8-4B-Distill` @131K (coding 0.39 <0.64 at same agentic 0.87)
- **Time:** 3552s (~59 min)

## Errors / Corrections
- None — clean run.

## Decisions
- **Keep** 4.9G file (D: 18.4G → ~13.5G after, still >10G guard) — dominated on coding but valid VRAM-efficient point; useful as low-VRAM fallback.
- Baseline stays Ling for this trial only — reverted to winning `Qwen3.8-4B-Distill` @131072 for next operator action.

## Open questions
- **TBD:** Ling tiny at `CTX 131072` with turbo2 KV — would coding hold while ctx doubles? VRAM headroom exists (2.5/7.7G used).

## References
- HF: `bartowski/Ling-3.0-tiny-GGUF` (Q4_K_M 4.9G, created 2026-08-18)
- Sessions: `2026-08-23-ling-tiny-validation.md` (0.80 quick), `2026-08-23-qwen38-4b-distill-full-trial.md` (base winner)
- Logs: `llama-server-20260823-112426-Ling-3.0-tiny-Q4_K_M.log`
- TSV: `results.tsv` `e2a298fb...` `dominated`

## Verification
- Measured: bench 52.8, coding per-task, agentic JSON 0.8667, NVML 2.5G, TSV `dominated`.
