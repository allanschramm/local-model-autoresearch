# 2026-08-23 — Cesium2-v7 Q8_0 @65536 Validation — bench 112 but agentic 0.00 (8 GB-class)

## Goal
Validate NEW `ram1234598766/Cesium2-v7-GGUF` Q8_0 1.6G at 65536 q4_0 — small text-gen from `filter=gguf&skip=40` 04:41.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 2121/7676 ok
- `block_count 28` dense, `b512 ub128 t8 fa on jinja`

## Setup
- File: `models/ram1234598766/Cesium2-v7-GGUF/cesium2-v7-q8_0.gguf` (hf 55s, Q8 1.6G, Modelfile 109)
- Baseline: `MODEL cesium2-v7-q8_0.gguf / CTX 65536 / q4_0 / 0.6/0.95/20`
- Found via `filter=gguf&skip=40` 04:41 text-gen GGUF (next small after Qwen 9B)

## Commands
```powershell
hf download ram1234598766/Cesium2-v7-GGUF cesium2-v7-q8_0.gguf --local-dir models/ram1234598766/Cesium2-v7-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Cesium2-v7 Q8 @65536 q4_0"
```

## Findings
- **Bench:** `112.0 t/s` (capped 4096) — **PASS** (>20, >50 Day) — fastest bench yet (1.6G Q8)
- **Agentic quick 5 tasks:** `0/5 0.0000` in 154s — **all FAIL:** T002 0.20 length_stops 1 (11798 chars, no tool), T004 0.00 length_stops 1 (8370 chars), T006 0.00 **HTTP 500 `peg-native format` error** (model output doesn't match expected tool format), T008 0.00 (35 chars), T010 0.00 length_stops 1 (11205 chars). Log `agentic-20260823-071447-cesium2-v7-q8_0.json`. Peak **3.3 GB** VRAM.
- **Status:** `incomplete` (validation profile) — coding 0.0, score 0.0000
- **TSV:** `7ca0ae0b-deb4-4de7-9c8e-39ab4d1a84ae` @ `b6c2ae1` `validation` `incomplete` `0.0000` `112.0` `112.0` `q4_0 65536 8/8 512/128 True/on/False` — `cesium2-v7-q8_0.gguf`
- **Interpretation:** Bench speed is **not predictive of agentic** — 112 TPS but 0.00 tool use (vs Smol 0.40, Qwen 1.00). HTTP 500 indicates model not instruction-tuned for tool format.

## Errors / Corrections
- T006 peg-native format error — model produced output not matching expected tool JSON.

## Decisions
- Validation **bench PASS but agentic 0.00** — **still run full trial per instruction** (validation not `rejected`) to complete vector, but expect dominated (0.00 quick predicts ~0.2 full). Keep 1.6G for now (D: 21.3G → ~19.7G after download, still >10G).

## Open questions
- **TBD:** Full vector (Claw-full + coding-10) on same Fingerprint — expected dominated.

## References
- HF: `ram1234598766/Cesium2-v7-GGUF` (Q8 1.6G dry-run 2026-08-23)
- API: `filter=gguf&skip=40` 04:41
- Logs: `llama-server-20260823-071211-cesium2-v7-q8_0.log`
- Sessions: `2026-08-23-qwen9b-abliterated-full-trial.md` (dominated 0.475/0.87)

## Verification
- Measured: bench 112.0, agentic JSON 0.0000 (0/5), NVML 3.3G, TSV row, HTTP 500 log.
