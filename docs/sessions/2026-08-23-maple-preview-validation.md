# 2026-08-23 — Maple Preview TQ1_0 Validation @65536 q4_0 — MODEL_REJECTED (8 GB-class)

## Goal
Validate NEW `deepgrove/maple-preview-GGUF` TQ1_0-head-Q4_K (5.0G, hf-verified 2026-08-23, ternary MoE 20B-A1B) at 65536 q4_0 on 8 GB-class rig.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, Windows host, `b10549` Gated DeltaNet, `VRAM_LIMIT 8000→7676`, preflight 1741/7676 ok, host 5163/27790 ok
- `block_count 24` auto `n-cpu-moe 24`

## Setup
- File: `models/deepgrove/maple-preview-GGUF/maple-preview-TQ1_0-head-Q4_K.gguf` (hf download 116s)
- Baseline: `MODEL maple-preview-TQ1_0-head-Q4_K.gguf / CTX 65536 / q4_0 / b512 ub128 t8 / fa on / jinja / 0.6/0.95/20`
- Engine pin `b10549` v0.2.0 CUDA 13.3 (same as Qwen3.8 validation)

## Commands
```powershell
hf download deepgrove/maple-preview-GGUF maple-preview-TQ1_0-head-Q4_K.gguf --local-dir models/deepgrove/maple-preview-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate maple-preview TQ1_0 Q4_K @65536 q4_0"
```

## Findings
- **FAIL — MODEL_REJECTED** in 29s: `llama-cli` exit 1 — `unknown model architecture: 'maple'` (`llama_model_load_from_file_impl: failed to load model`, `common_fit_params` fit error). Full stderr in `autoresearch/runners/logs/llama-server-20260823-*.log` (b10549 does not implement `maple` arch).
- Preflight passed (1741 MB) but load fails before bench/server — no TPS/VRAM measured, no agentic run.
- TSV: last row `maple-preview-TQ1_0-head-Q4_K.gguf` `rejected` `MODEL_REJECTED` `unknown model architecture: 'maple'` (0.0 tps, 0.0 VRAM).
- **Implication:** HF `filter=gguf&sort=createdAt` correctly returned this NEW ternary MoE, but it is **not triable on pinned `b10549`** — requires newer/patched `llama.cpp` with `maple` arch (ternary TQ1/TQ2). Not a VRAM or ctx failure.

## Errors / Corrections
- Prior estimate 5.0G fits 8GB was correct per `hf --dry-run`, but arch support was **TBD** in `2026-08-23-new-models-qwen38-distill.md` — now resolved as **rejected** until engine update.
- No retry at other CTX/KV — load fails before context matters.

## Decisions
- **Mark `maple-preview` as `rejected` — do not retry** until `llama.cpp` adds `maple` arch (track upstream PRs/commits). Keep GGUF on disk for now (D: 42G free >10G guard) — delete only if space falls below guard or operator says purge.
- **Next model:** `unsloth/SmolLM3-3B-GGUF` Q4_K_M 1.9G @128K YARN — lightweight control, fits easily, LCB 30% thinking — per optimal order.

## Open questions
- **TBD:** Upstream `llama.cpp` support for `maple` ternary arch — which commit adds it? Revalidate TQ1_0/TQ2_0 when `b10xxx` ships it.

## References
- HF: `deepgrove/maple-preview-GGUF` (hf --dry-run 2026-08-23: TQ1_0 5.0G / TQ2_0 5.9G)
- Logs: `autoresearch/runners/logs/llama-server-20260823-*-maple*.log`
- Sessions: `2026-08-23-new-models-qwen38-distill.md` (secondary bet), `2026-08-23-new-models-api-exhaustive.md` (found in API sweep)

## Verification
- Measured: `llama-cli` exit 1, server log `unknown model architecture`, TSV `rejected`.
- No SKU/PII, memory-class only, follow-up per `docs/sessions/AGENTS.md`.
