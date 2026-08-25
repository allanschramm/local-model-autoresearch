# 2026-08-23 — Qwen3.8-4B-Distill Q4_K_M Full Trial @131072 q4_0 — on_front (8 GB-class)

## Goal
Complete Objective Vector for NEW `empero-ai/Qwen3.8-4B-Distill-GGUF` Q4_K_M at 131072 ctx on same Fingerprint as validation — Claw-full (15 tasks) + coding-10 (40 tasks) + bench.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, Windows host
- Baseline `VRAM_LIMIT_MB` 8000 → 7676 MB, preflight 5320 MB ok, host 27790 ok
- Engine `b10549` Gated DeltaNet, `q4_0` KV, `fa on`, `b 512 / ub 128 / t 8 / tbd 8`, `jinja`, `cont-batching`, `cache-reuse 256`, dense `block_count 33`

## Setup
- Model: `models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf` (2.8G, hf-verified 2026-08-23)
- Sampler: `TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / REP 1.0` (HF card) — same as validation
- Baseline unchanged from `d73b79d` (`Qwen3.8-4B-Q4_K_M.gguf / 131072 / q4_0`)
- Fingerprint join: validation `6b7e2a3f...` (quick 1.0) → this full trial `6069530a...` — identical `config_json`

## Commands
```powershell
# same Baseline (no edit)
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial Qwen3.8-4B-Distill Q4_K_M @131072 q4_0 agentic-full + coding-10"
```
No `autoloop` — single monitored run.

## Findings
- **Bench:** `tg 74.9 t/s` (capped 4096 bench ctx) — same as validation, >50 Day floor
- **Coding (10 each):** `humaneval 9/10 (0.9000) @96.1 t/s`, `mbpp 9/10 (0.9000) @89.4 t/s`, `lcb 5/10 (0.5000) @99.7 t/s`, `bigcode 1/10 (0.1000) @89.0 t/s` → **combined `0.6400`** (avg). Log `llama-server-20260823-023422-Qwen3.8-4B-Q4_K_M.log`.
- **Agentic quick (co-run):** `5/5 1.0000` in 179s (same 5 tasks as validation, re-measured)
- **Agentic full (15 tasks, rule-based):** `13/15 0.8667` in 1087s — **passes:**
  - T002 0.50, T004 1.00, T006 1.00, T008 1.00, T010 1.00, T012 1.00, T014 0.70, T016 0.50, T018 1.00, T044 1.00, T046 1.00 (length_stops 1, `finish_reason=length` on web_real but PASS), T048 1.00, T050 1.00
  - **Fails:** T053 0.00, T054 0.00 (same finance web_real cluster as `Ornith-1.5-35B` — see `2026-08-20-ornith-35b-4096-rerun.md` pattern; not a new failure mode for this rig)
  - Wrote `agentic-20260823-030251-Qwen3.8-4B-Q4_K_M.json`
- **Combined TPS:** `94.2` (coding gen TPS blended; bench 74.9 alone)
- **VRAM:** peak `5.4 GB` (NVML 20ms) — fits 7676 limit with 2.2G headroom
- **TSV row:** `6069530a-f5c6-4bd1-be5e-c42be4577be2` @ `d73b79d`, `agentic-full` `on_front` `0.8667 / 0.6400` `94.2 / 74.9` `q4_0 131072 8/8 512/128 True/on/False` `jinja` — `status on_front` (complete, non-dominated).
- **Pareto impact:** First `on_front` at 131072 with `q4_0` on this rig for a dense 4B — beats prior `Ornith-1.5-9B` bench 44.4 / coding ~0.5 region (see `results.tsv` `19cd4006...` 44.4 t/s) on TPS while holding `min(coding,agentic)=0.64` (vs `Ornith-35B` 0.8667 agentic but 2× VRAM). Day profile now viable: `TPS 94.2 ≥50` and `min 0.64` max among Day-eligible points (verify `scripts/recompute_status.py`).

## Errors / Corrections
- First full-trial attempt timed out at 300s (default `bash` timeout) mid-LCB — **retried with `timeout:0` (no deadline)** per `AGENTS.md` no-timeout rule — completed in 1783s. No code change.
- Two finance tasks (T053/T054) fail — **not a harness bug** — same web_real keyword miss seen on 35B; `length_stops` on T046 but still PASS.

## Decisions
- **Keep** `Qwen3.8-4B-Q4_K_M.gguf` — it is now `on_front`.
- Baseline remains `Qwen3.8-4B-Q4_K_M @131072 q4_0` — next hill-climb should `autoloop.py --mode tps` **only on operator command** (not auto).

## Open questions
- **TBD:** Whether `maple-preview` TQ1_0 5.0G at same ctx beats this point on `min(coding,agentic)` vs TPS — falsifiable via same `--agentic-full --include-coding` on its Fingerprint.
- **TBD:** Night `65536` vs `131072` Pareto for this model — TPS vs ctx tradeoff not yet measured.

## References
- HF: `empero-ai/Qwen3.8-4B-Distill-GGUF` (Q4 2.8G)
- Sessions: `2026-08-23-qwen38-4b-distill-validation.md` (validation pass), `2026-08-23-new-models-qwen38-distill.md` (pick), `2026-08-23-new-models-api-exhaustive.md` (exhaustive sweep)
- Logs: `llama-server-20260823-023422-Qwen3.8-4B-Q4_K_M.log`, `agentic-20260823-030251-Qwen3.8-4B-Q4_K_M.json`
- TSV: `results.tsv` `6069530a...` (`on_front`), `6b7e2a3f...` (validation)

## Verification
- Measured: bench log, agentic JSON, coding per-task PASS/FAIL, NVML peak, TSV `on_front`.
- No SKU/PII, memory-class only, follow-up file per `docs/sessions/AGENTS.md`.
