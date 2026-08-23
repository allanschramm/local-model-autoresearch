# 2026-08-23 — MindSparQ-Coder-1.5B Q4_K_M @65536 Validation — bench 182 but agentic 0.00 (8 GB-class)

## Goal
Validate NEW `mradermacher/MindSparQ-Coder-1.5B-GGUF` Q4_K_M 986M at 65536 q4_0 — 1.5B code/vibe-coding model (mradermacher re-quant, 03:31).

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 1491/7676 ok
- `block_count 28` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/mradermacher/MindSparQ-Coder-1.5B-GGUF/MindSparQ-Coder-1.5B.Q4_K_M.gguf` (hf 32s, Q4 986M)
- Baseline: `MODEL MindSparQ-Coder-1.5B.Q4_K_M.gguf / CTX 65536 / q4_0 / 0.6/0.95/20`
- Found via `filter=gguf&skip=60` 03:31 code-tagged GGUF (qwen2.5)

## Commands
```powershell
hf download mradermacher/MindSparQ-Coder-1.5B-GGUF MindSparQ-Coder-1.5B.Q4_K_M.gguf --local-dir models/mradermacher/MindSparQ-Coder-1.5B-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate MindSparQ-Coder 1.5B Q4_K_M @65536 q4_0"
```

## Findings
- **Bench:** `182.3 t/s` (capped 4096) — **PASS** (>20, >50 Day) — fastest bench yet (1.5B)
- **Agentic quick 5 tasks:** `0/5 0.0000` in 95s — **all FAIL:** T002 0.20 (0 tool), T004 0.00 length_stops 17543, T006 0.00 length_stops 12249, T008 0.00, T010 0.00 length_stops 13785. Log `agentic-20260823-072806-MindSparQ-Coder-1.5B.Q4_K_M.json`. Peak **2.7 GB** VRAM (lowest yet).
- **Status:** `incomplete` (validation profile) — coding 0.0, score 0.0000
- **TSV:** `57067b6b-778b-4d24-9277-c7b0fbc7c32a` @ `e4ec905` `validation` `incomplete` `0.0000` `182.3` `182.3` `q4_0 65536 8/8 512/128 True/on/False` — `MindSparQ-Coder-1.5B.Q4_K_M.gguf`
- **Interpretation:** Bench 182 vs agentic 0.00 — **speed ≠ capability** — 1.5B too small for tool use (0 calls on 4/5).

## Errors / Corrections
- Length_stops on 3/5 tasks (17K, 12K, 13K chars) — model generates long non-tool text, hits `max_tokens` cap without tool calls.

## Decisions
- Validation **bench PASS but agentic 0.00** — **still run full trial per instruction** (validation not `rejected`) to complete vector (expected dominated/0 coding).

## Open questions
- **TBD:** Full vector (Claw-full + coding-10) on same Fingerprint — expected 0 coding.

## References
- HF: `mradermacher/MindSparQ-Coder-1.5B-GGUF` (Q4 986M dry-run 2026-08-23)
- API: `filter=gguf&skip=60` 03:31
- Logs: `llama-server-20260823-072629-MindSparQ-Coder-1.5B.Q4_K_M.log`
- Sessions: `2026-08-23-cesium2-v7-full-trial.md` (rejected 0.00)

## Verification
- Measured: bench 182.3, agentic JSON 0.0000 (0/5), NVML 2.7G, TSV row.
