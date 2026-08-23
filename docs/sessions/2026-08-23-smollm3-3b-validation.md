# 2026-08-23 — SmolLM3-3B Q4_K_M Validation @131072 q4_0 (8 GB-class)

## Goal
Validate lightweight control `unsloth/SmolLM3-3B-GGUF` Q4_K_M 1.9G at 131072 q4_0 (128K YARN) on 8 GB-class rig.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 3417/7676 ok, host 3417/27790 ok
- `block_count 36` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/unsloth/SmolLM3-3B-GGUF/SmolLM3-3B-Q4_K_M.gguf` (hf download 48s)
- Baseline: `MODEL SmolLM3-3B-Q4_K_M.gguf / CTX 131072 / q4_0 / TEMP 0.6 TOP_P 0.95 TOP_K 20`
- Same sampler as Qwen distill for fair comparison

## Commands
```powershell
hf download unsloth/SmolLM3-3B-GGUF SmolLM3-3B-Q4_K_M.gguf --local-dir models/unsloth/SmolLM3-3B-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate SmolLM3-3B Q4_K_M @131072 q4_0"
```

## Findings
- **Bench:** `110.0 t/s` (fastest yet — 1B fewer params than Qwen), threshold 20 — **PASS**, surpasses Day 50
- **Agentic quick (5 tasks):** `2/5 0.4000` in 56s — **weak:** T002 0.20 (tool fail), T004 0.35 (event fail), T006 0.00 (no tool calls), T008 0.65, T010 0.80 — only dedup/lookup passed. Log `agentic-20260823-031037-SmolLM3-3B-Q4_K_M.json`.
- **VRAM:** peak `5.8 GB` (similar to Qwen 5.5 despite smaller model — KV dominates @131K)
- **Status:** `incomplete` (validation profile) — coding 0.0, score 0.4000
- **TSV:** `4b1b7558-fa51-4657-b8d8-af48e5409b31` @ `1b07bd1` `validation` `incomplete` `0.4000` `110.0` `110.0` `q4_0 131072 8/8 512/128 True/on/False`
- **Interpretation:** Fast TPS (+49% over Qwen 74.9) confirms size scaling, but **agentic 0.4000 is weakest yet** — predicts LCB ~15% no-thinking / 30% thinking per card, far below Qwen 0.64. Not a Day contender despite speed.

## Errors / Corrections
- No crash — model loads. Weak agentic is **capability**, not harness bug (0 tool calls on T006). No `MODEL_REJECTED`.

## Decisions
- Validation **technically PASS on bench** (`incomplete` not `rejected`) but **weak agentic** — still run full trial per instruction to complete vector (will be dominated), then keep or purge based on D: guard.

## Open questions
- **TBD:** Full vector (Claw-full + coding-10) on same Fingerprint — expected dominated vs Qwen `on_front`.

## References
- HF: `unsloth/SmolLM3-3B-GGUF` (Q4 1.9G, hf --dry-run 2026-08-23)
- Sessions: `2026-08-23-qwen38-4b-distill-full-trial.md` (`on_front` 0.64/0.87), `2026-08-23-new-models-qwen38-distill.md` (Smol as control)
- Logs: `llama-server-20260823-030938-SmolLM3-3B-Q4_K_M.log`

## Verification
- Measured: bench 110.0, agentic JSON 0.4000, NVML 5.8G, TSV row.
