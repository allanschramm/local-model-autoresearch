# 2026-08-23 — Qwen3.8-4B-Distill Q4_K_M Validation @131072 q4_0 (8 GB-class)

## Goal
Validate NEW `empero-ai/Qwen3.8-4B-Distill-GGUF` Q4_K_M at 131072 ctx on 8 GB-class discrete rig — bench + agentic quick (5 tasks) per `benchmark_search.py --validation`.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, Windows host, WSL2 available
- Baseline `VRAM_LIMIT_MB` 8000 → clamped 7676 MB (keepout 512), preflight 5320 MB ok
- Engine `llama.cpp-releases/upstream/b10549` (Gated DeltaNet), `b` 512 / `ub` 128 / `t` 8 / `fa on` / `q4_0` KV / `jinja` / `cont-batching` / `cache-reuse 256`
- D: free 48.3 GB before download (see Prep)

## Setup
- Model: `models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf` (hf download 2026-08-23, 2.8G, 68s)
- Baseline seeded from HF card `empero-ai/Qwen3.8-4B-Distill-GGUF` (Apache-2.0): `TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0`, `CTX 131072`, `KV q4_0`, `N_GPU_LAYERS 99`, `NO_MMAP False` (dense)
- Config archived in `autoresearch/core/config.py` (commit `967fc51` state): `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS` as above

## Commands (reproducible)
```powershell
hf download empero-ai/Qwen3.8-4B-Distill-GGUF Qwen3.8-4B-Q4_K_M.gguf --local-dir models/empero-ai/Qwen3.8-4B-Distill-GGUF
# Baseline edit applied before run (see config.py diff)
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Qwen3.8-4B-Distill Q4_K_M @131072 q4_0"
```
`autoloop.py` not used — operator-only.

## Findings
- **Bench (llama-cli, 3 reps, ctx capped 4096 for bench):** `tg 74.9 t/s` (peak 74.9, threshold 20.0) — **PASS**, surpasses Day `TPS_FLOOR 50` by +49%. Log `autoresearch/runners/logs/llama-server-20260823-022318-Qwen3.8-4B-Q4_K_M.log` (NVML 20ms sampling).
- **Agentic quick (5 tasks, rule-based scoring, no LLM judge):** `5/5 passed` `score=1.0000` in 136s — `T002 0.50 / T004 0.80 / T006 1.00 / T008 1.00 / T010 1.00`. Services on :9100-9103, log `agentic-20260823-022537-Qwen3.8-4B-Q4_K_M.json`.
- **VRAM:** peak `5.5 GB` (NVML), headroom ~2.1 GB vs 7676 MB limit — fits 131K with `q4_0` (hybrid 8/32 attention → ~¼ dense KV).
- **Status:** `incomplete` (validation profile) — coding axis intentionally `0.0000`; needs `--agentic-full` + `--include-coding` on same Fingerprint for complete Objective Vector.
- **TSV row:** `6b7e2a3f-c37b-43c3-a28e-aec7164bb56b` @ `967fc51`, `llama.cpp` `validation` `incomplete` `1.0000` `74.9` `74.9` `q4_0` `131072` `8/8` `512/128` `on` `False` `jinja` (see `results.tsv` tail).
- **Dense arch:** `block_count=33`, `n-cpu-moe=None` (dense), `--mmap --no-warmup --simple-io --single-turn`.

## Errors / Corrections
- None — first try bench + quick both pass. Prior 404 repo (`empero-ai/Qwen3.8-4B-GGUF`) corrected to `...-Distill-GGUF` before this run.
- No VRAM kill, no timeout, no `max_tokens` truncation (validation uses capped bench ctx).

## Decisions
- Validation **PASS** — proceed to full trial (`--agentic-full --include-coding`) on same Fingerprint (`CTX 131072`, `q4_0`, `TEMP 0.6`).
- Keep GGUF on disk — D: still 48G free (>10G guard).

## Open questions
- **TBD:** Full Objective Vector (Claw-full 15 tasks + coding-10) on same Fingerprint — falsifiable via next run's TSV rows with identical `config_json`.
- **TBD:** TPS at full ctx server (bench was capped 4096) vs 131K KV reservation — verify `bench_tg` vs `tps` columns after full trial.

## References
- HF repos: `empero-ai/Qwen3.8-4B-Distill-GGUF` (Q4 2.8G verified `hf --dry-run` 2026-08-23)
- Sessions: `docs/sessions/2026-08-23-new-models-qwen38-distill.md` (primary pick), `docs/sessions/2026-08-23-new-models-api-exhaustive.md` (API sweep)
- Logs: `autoresearch/runners/logs/llama-server-20260823-022318-Qwen3.8-4B-Q4_K_M.log`
- TSV: `results.tsv` tail `6b7e2a3f...`

## Verification
- Desk + measured: bench log, agentic JSON, NVML peak, TSV row.
- No SKU/hostname/PII, memory-class only, follow-up file per `docs/sessions/AGENTS.md`.
