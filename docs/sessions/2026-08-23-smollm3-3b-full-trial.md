# 2026-08-23 — SmolLM3-3B Q4_K_M Full Trial @131072 q4_0 — on_front (8 GB-class)

## Goal
Complete Objective Vector for lightweight control `unsloth/SmolLM3-3B-GGUF` Q4_K_M 1.9G at 131072 q4_0 — Claw-full + coding-10.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 3417/7676 ok
- `block_count 36` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/unsloth/SmolLM3-3B-GGUF/SmolLM3-3B-Q4_K_M.gguf` (hf 48s)
- Baseline: `MODEL SmolLM3-3B-Q4_K_M.gguf / CTX 131072 / q4_0 / 0.6/0.95/20` — same as validation `f4526e0`

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --include-coding --desc "trial SmolLM3-3B Q4_K_M @131072 q4_0 agentic-full + coding-10"
```

## Findings
- **Bench:** `109.9 t/s` (vs Qwen 74.9, +47%) — fastest bench yet
- **Coding:** HE 3/10 0.30 @147.2, MBPP 6/10 0.60 @133.4, LCB 4/10 0.40 @146.2, BigCode 0/10 0.00 @131.2 → **0.3650** combined. Below Qwen 0.64 (-43%).
- **Agentic quick (co-run):** `1/5 0.2000` in 52s — T008 0.65 only, others 0/0 tool calls (worse than validation 0.40 due to nondet).
- **Agentic full:** `8/15 0.5333` in 205s — **passes:** T008 0.65, T010 0.80, T012 0.85, T016 0.50, T044 0.63, T046 0.85, T048 1.00, T050 0.78 — **fails:** T002 0.20, T004 0.35, T006 0.00, T014 0.00, T018 0.25, T053 0.00, T054 0.00. Weaker than Qwen 0.8667 (-38%) and even validation 0.40. Wrote `agentic-20260823-032548-SmolLM3-3B-Q4_K_M.json`.
- **Combined TPS:** `138.5` (coding gen) vs bench 109.9 — Day-eligible (≥50) but `min(coding,agentic)=0.365` trails Qwen 0.64
- **VRAM:** peak `5.9 GB` (higher than Qwen 5.4 despite 1.9G file — KV 131K dominates; +0.4G vs Qwen)
- **TSV:** `c5ae6cbd-1733-4609-92f0-7990058aa510` @ `f4526e0` `agentic-full` `on_front` `0.5333 / 0.3650` `138.5 / 109.9` `q4_0 131072 8/8 512/128 True/on/False` — `on_front` via TPS axis (not dominated: Smol 138.5 > Qwen 94.2 on TPS, but losing on both IQ axes).
- **Pareto:** Smol joins front on speed, but **dominated on quality** — not a Day pick (Day = max `min` among ≥50 TPS, Qwen wins 0.64 vs 0.365). Keep as speed control only.

## Errors / Corrections
- No crash — weak tool use (0 calls on 4/15 tasks) is model capability, not harness. Agentic quick regression 0.40→0.20 shows nondet at low tool-use rate.

## Decisions
- **Keep** Smol GGUF for now (D: still ~40G >10G guard, 1.9G tiny). It is `on_front` on speed but **not recommended for coding** — document as speed baseline.
- Baseline stays `SmolLM3-3B` for this trial only — next trial should revert to Qwen distill for hill-climb or test new find.

## Open questions
- **TBD:** Whether any sub-2B Q4 at 131K can beat Qwen's `min=0.64` — likely not without larger params.

## References
- HF: `unsloth/SmolLM3-3B-GGUF` (Q4 1.9G)
- Sessions: `2026-08-23-smollm3-3b-validation.md` (bench 110 + 0.40), `2026-08-23-qwen38-4b-distill-full-trial.md` (`on_front` 0.64/0.87), `2026-08-23-new-models-qwen38-distill.md`
- Logs: `llama-server-20260823-031201-SmolLM3-3B-Q4_K_M.log`
- TSV: `results.tsv` `c5ae6cbd...`

## Verification
- Measured: bench 109.9, coding per-task, agentic JSON 0.5333, NVML 5.9G, TSV `on_front`.
