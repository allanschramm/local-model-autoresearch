# 2026-08-23 — Ornith-1.5-9B-Function-Calling Q2_K Validation @131072 — MODEL_REJECTED (8 GB-class)

## Goal
Validate NEW `ermiaazarkhalili/Ornith-1.5-9B-Function-Calling-xLAM-Unsloth-GGUF` Q2_K 3.8G at 131072 q4_0 — newest GGUF in 2026-08-23 API window (created 06:25 UTC).

## Hardware
- `discrete_gpu`, 8 GB VRAM class, `b10549`, `VRAM_LIMIT 8000→7676`, preflight 6315/7676 ok
- `block_count 33` dense (auto)

## Setup
- File: `models/ermiaazarkhalili/Ornith-1.5-9B-Function-Calling-xLAM-Unsloth-GGUF/ornith-1.5-9b-function-calling-xlam-unsloth.q2_k.gguf` (hf 83s, .gitattributes 1.6K + Q2_K 3.8G only file)
- Baseline: `MODEL ornith-1.5-9b-function-calling-xlam-unsloth.q2_k.gguf / CTX 131072 / q4_0 / 0.6/0.95/20`
- HF API: `filter=gguf&sort=createdAt&limit=20` top hit 06:25 (text-gen via `conversational` tag)

## Commands
```powershell
hf download ermiaazarkhalili/Ornith-1.5-9B-Function-Calling-xLAM-Unsloth-GGUF ornith-1.5-9b-function-calling-xlam-unsloth.q2_k.gguf --local-dir models/ermiaazarkhalili/Ornith-1.5-9B-Function-Calling-xLAM-Unsloth-GGUF
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Ornith-1.5-9B-Function-Calling Q2_K @131072 q4_0"
```

## Findings
- **FAIL — MODEL_REJECTED** in 45s: `llama-cli` exit 1 — `check_tensor_dims: tensor 'blk.32.attn_norm.weight' not found` (`llama_model_load` failed, `common_fit_params` g to fit). Repo lists only one Q2_K file (3.8G) — likely truncated/quant mismatch (33 blocks expected, 32 found) or base mismatch with `b10549` Gated DeltaNet arch.
- No bench TPS, no agentic run, peak 0.0 — pre-server load failure.
- TSV: last row `ornith-1.5-9b-function-calling-xlam-unsloth.q2_k.gguf` `rejected` `MODEL_REJECTED` `tensor blk.32... not found` (0.0 tps).

## Errors / Corrections
- File appears incomplete — HF `dry-run` shows only Q2_K (no Q4/Q5/Q8), unusual for 9B (normal Q4 ~5G). Not a VRAM/ctx failure.

## Decisions
- **Mark rejected** — do not retry Q2_K until publisher fixes quant. Keep file for now (D: 38.6G >10G) but flag for purge if next download needs space.
- **Next:** Re-query HF API for next NEW small GGUF after 06:25 window — likely next mradermacher re-quant (0-signal) vs fallback to unsloth first-party (none NEW). Consider closing NEW scan.

## References
- HF: `ermiaazarkhalili/Ornith-1.5-9B-Function-Calling-xLAM-Unsloth-GGUF` (dry-run 2026-08-23: 1 file Q2_K 3.8G)
- API: `https://huggingface.co/api/models?filter=gguf&sort=createdAt&direction=-1&limit=20` (2026-08-23 06:25 hit)
- Logs: `llama-server-20260823-*.log` `blk.32` error
- Sessions: `2026-08-23-new-models-api-exhaustive.md` (exhaustive sweep), `2026-08-23-qwen38-4b-distill-full-trial.md` (winning point)

## Verification
- Measured: `llama-cli` exit 1, server log `check_tensor_dims`, TSV `rejected`.
