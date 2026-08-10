# LFM2.5-2.6B-Q8_0 — Model Card (Local)

**GGUF path:** `models/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf`  
**Family:** Liquid LFM2.5  
**Architecture type:** Dense hybrid (`lfm2`) — not MoE  
**Quantization:** `Q8_0`  

## Architecture (from GGUF metadata)

- `general.architecture`: `lfm2`
- `general.name`: `LiquidAI_LFM2.5 2.6B`
- `general.size_label`: `2.6B`
- `lfm2.context_length`: `128000`
- `lfm2.block_count`: `30`
- MTP / `nextn`: `none`

Harness: dense → `N_CPU_MOE` must stay `None` (no expert offload).

## Hardware requirements (discrete 8 GB-class NVIDIA)

Weights ~2.9 GB (Q8_0). Fits comfortably in 8GB VRAM with 65k context.

| ctx | KV | Fits? | Peak VRAM |
|---|---|---|---|
| 65k | f16 | **Yes** | **6.0 GB** |

## Recommended settings / config baseline

```python
MODEL = 'LFM2.5-2.6B-Q8_0.gguf'
CTX_SIZE = 65536
KV_CACHE_K = 'f16'
KV_CACHE_V = 'f16'
BATCH_SIZE = 512
UBATCH_SIZE = 256
THREADS = 8
THREADS_BATCH = 8
FLASH_ATTN = 'on'
N_CPU_MOE = None
TPS_FLOOR = 15.0
```

Recommended Sampler:
- TEMP: 0.6
- TOP_P: 0.95
- TOP_K: 20
- MIN_P: 0.0

## Measured (Objective Vector)

| Metric | Pre-fix (2026-08-05) | Post-harness remasure (2026-08-08) |
|---|---|---|
| **Trial Status** | `on_front` | **`on_front`** |
| **Agentic (Claw full)** | 0.3333 (5/15) | **0.8667** (13/15) |
| **Coding (10 tasks/ds)** | 0.4800 | **0.5050** |
|   HumanEval+ | 0.8000 | 0.8000 |
|   MBPP+ | 0.7000 | 0.8000 |
|   LiveCodeBench | 0.3000 | 0.3000 |
|   BigCodeBench | 0.0000 | 0.0000 |
| **Bench tg** | 78.7 t/s | 78.8 t/s |
| **Combined TPS** | 85.3 t/s | 87.1 t/s |
| **Peak VRAM** | 6.0 GB | 6.0 GB |

Evidence: `results.tsv` rows for `LFM2.5-2.6B-Q8_0.gguf`. Claw jump is the 2026-08-08 agentic harness fix (`reasoning_content` / `max_tokens≥2048`), not a model change — see [thinking-models-claw-harness.md](../discovery/thinking-models-claw-harness.md).
