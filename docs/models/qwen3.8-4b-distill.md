# Qwen3.8-4B-Distill — Model Card

**Source repo:** https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF
**Base model:** full-parameter distillation of Qwen3.8-2.4T-A95B into the Qwen3.5-4B architecture (~45k curated teacher traces)
**License:** Apache-2.0
**Local file:** `models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf` (2.8 GB; Q5_K_M 3.2 / Q6_K 3.6 / Q8_0 4.6 / BF16 8.7 also in repo)
**Family:** Qwen 3.8 distill on Qwen 3.5 architecture (Alibaba / Empero AI)
**Quantization:** standard Q4_K_M

## Architecture (from GGUF via harness)

- **Dense** (`block_count` = **33**, `expert_count` = 1) — harness `is_moe_model` reads dense ⇒ no `N_CPU_MOE`.
- Hybrid attention: Gated DeltaNet layers with sparse full-attention interval (Qwen3.5 pattern, 8/32 full-attn per publisher card) ⇒ ~¼ dense KV footprint at equal ctx.
- Native context class 262K-equivalent; runs 131072 configured on this rig.

## Hardware requirements (discrete 8 GB-class)

- Publisher tier: Q4_K_M / Q5_K_M → 4–6 GB GPUs; fits entirely (`N_GPU_LAYERS 99`).
- Measured @131072 + `q4_0` KV + flash-attn: peak **5.5 GB**, preflight est 5320 MB vs 7676 limit. Comfortable headroom; no OOM across validation + claw-full + coding-10.

## Recommended settings (publisher card)

- TEMP 0.6 · TOP_P 0.95 · TOP_K 20 · MIN_P 0.0 · REPEAT_PENALTY 1.0
- Chat template embedded; requires `JINJA=True`.

## MTP

- **No MTP tensors in this GGUF** (plain distill; not the `-MTP-GGUF` variant). No embedded draft head; separate draft models fail on this arch family — do not add spec decoding without an explicit A/B.

## MoE split ("VITRIOL split")

N/A — dense model, all layers on GPU.

## Our config baseline (measured 2026-08-23)

```python
ENGINE_DEFAULTS = {
    'MODEL': 'Qwen3.8-4B-Q4_K_M.gguf',
    'CTX_SIZE': 131072,
    'KV_CACHE': 'q4_0', 'KV_CACHE_K': 'q4_0', 'KV_CACHE_V': 'q4_0',
    'BATCH_SIZE': 512, 'UBATCH_SIZE': 128, 'THREADS': 8, 'THREADS_BATCH': 8,
    'FLASH_ATTN': 'on', 'NO_MMAP': False, 'JINJA': True,
}
SAMPLER_DEFAULTS = {'TEMP': 0.6, 'TOP_P': 0.95, 'TOP_K': 20, 'MIN_P': 0.0}
```

Measured Objective Vector (complete, same Fingerprint):

| Axis | Value | Evidence |
| :--- | ---: | :--- |
| bench tg | 74.9 t/s | validation + claw-full rows agree |
| Combined TPS (coding gen) | 94.2 | coding row |
| agentic (claw-full 15) | **0.8667** (13/15) | T053/T054 finance web_real fail only |
| coding-10 | **0.6400** (HE 0.9 / MBPP 0.9 / LCB 0.5 / BC 0.1) | combined column |
| min(agentic, coding) | **0.6400** | Day+Night eligible |

Status `on_front`; current Baseline pick. TSV `6b7e2a3f…` (validation) / `6069530a…` (full).

## Sources / Verification

- HF card `empero-ai/Qwen3.8-4B-Distill-GGUF` — sizes, tiers, mixed eval deltas (mmlu +0.199 / gsm8k_cot −0.065 vs Qwen3.5-4B base), Apache-2.0. Extracted 2026-08-23.
- `hf --dry-run` file listing 2026-08-23 (sizes above).
- Local measured: `docs/sessions/2026-08-23-qwen38-4b-distill-validation.md`, `2026-08-23-qwen38-4b-distill-full-trial.md`.

## Open questions

- **TBD:** 262K native vs 131K YaRN practical ceiling on b10549 — verify from effective server log before raising `CTX_SIZE`.
