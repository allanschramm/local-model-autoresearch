# 2026-08-23 — Qwen3.8-4B-Heretic Q4_K_M Full Trial @131072 — on_front 0.64/0.67 (8 GB-class)

## Goal
Complete Objective Vector for NEW `yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF` model-Q4_K_M 2.8G at 131072 q4_0 — heretic variant of winning distill.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 5326/7676 ok
- `block_count 33` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF/model-Q4_K_M.gguf` (hf 70s, 2.8G)
- Baseline: `MODEL model-Q4_K_M.gguf / CTX 131072 / q4_0 / 0.6/0.95/20` — same as validation `8050d09`
- Heretic abliterated Q4_K_M comparable to base Q4 (no low-bit confound)

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial Qwen3.8-4B-Heretic Q4_K_M @131072 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `74.9 t/s` (capped 4096) — identical to base 74.9
- **Coding:** HE 9/10 0.90 @96.5, MBPP 9/10 0.90 @91.2, LCB 5/10 0.50 @96.9, BigCode 1/10 0.10 @90.4 → **0.6400** combined — identical to base 0.64
- **Agentic quick co-run:** `5/5 1.0000` in 145s — T002 0.50, T004 0.80, T006 1.00, T008 1.00, T010 1.00
- **Agentic full:** `10/15 0.6667` in 1559s — **passes:** T002 0.50, T004 1.00, T006 1.00, T008 1.00, T010 1.00, T012 0.75, T014 0.70, T016 0.50, T018 0.85, T044 1.00 — **fails:** T046 0.00 (HTTP 400 138086 >131072 ctx exceed, `n_ctx 131072`), T048 0.00, T050 0.00, T053 0.00, T054 0.00. Wrote `agentic-20260823-061219-model-Q4_K_M.json`. **Base distill was 13/15 0.8667 — heretic drops 3 tasks (T046/T048/T050) to 0.00 due to context blow-up (same finance web_real cluster).**
- **Combined TPS:** `94.3` (coding gen), bench 74.9
- **VRAM:** peak `5.5 GB` (same as base)
- **TSV:** `16c1322c-3826-477c-843f-79c368bf1408` @ `8050d09` `agentic-full` `on_front` `0.6667 / 0.6400` `94.3 / 74.9` `q4_0 131072 8/8 512/128 True/on/False` — **on_front** via same `min=0.64` as base but agentic 0.6667 < 0.8667, so **dominated on agentic axis** — base remains Day/Night pick (`min 0.64` equal but agentic tie-break goes to 0.8667).
- **Time:** 2200s (~37 min)

## Errors / Corrections
- T046 context exceed 138086 >131072 — same `REASONING_PRESERVE` inflation seen on 35B but now on 4B heretic at 131K — not a harness bug, but heretic's longer reasoning trace exceeds ctx.

## Decisions
- **Keep** the heretic 2.8G — its vector is `on_front` but **not winning** vs base distill. Document as abliterated variant not superior for coding.
- Baseline stays heretic for this trial only — revert to base `empero-ai` distill `on_front` for any hill-climb.

## Open questions
- **TBD:** Heretic at `CTX 65536` or with `REASONING_PRESERVE False` — would it avoid T046 exceed and recover 0.8667?

## References
- HF: `yachen4ever/Qwen3.8-4B-Distill-Heretic-Abliterated-GGUF` (Q4 2.8G, 06:23)
- Sessions: `2026-08-23-qwen-heretic-validation.md` (1.0), `2026-08-23-qwen38-4b-distill-full-trial.md` (base 0.8667/0.64 winner)
- Logs: `llama-server-20260823-053651-model-Q4_K_M.log`, `agentic-20260823-061219-model-Q4_K_M.json`
- TSV: `results.tsv` `16c1322c...`

## Verification
- Measured: bench 74.9, coding per-task, agentic JSON 0.6667, NVML 5.5G, TSV `on_front`.
