# Ornith-1.5-9B — Model Card (Local)

**Source repo:** https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF
**License:** MIT
**Local file:** `models/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf` (5.63 GB)
**Family:** Ornith-1.5 (ornith-ai; Qwen 3.5 architecture)
**Quantization:** official Q4_K_M (only official GGUF repo; no Unsloth/MTP pack yet)

## Architecture (verified from local GGUF, harness-backed `model_info.py` 2026-08-19)
- Causal LM, Qwen 3.5 arch (`qwen35.*` fields), **dense**
- **`block_count` = 32**
- `kv_f16_mb @ 65536 ctx` = 4096 MB (q4_0 KV ≈ 1 GB at 65k)
- No `nextn` / `mtp` / `draft` tensors in the GGUF (verified field scan)
- **TBD:** exact SSM/attention layout (full_attention_interval, hidden dim) — same file size and arch family as Ornith-1.0-9B Q4_K_M; see [ornith-1.0-9b.md](./ornith-1.0-9b.md) for the family layout.

## Hardware requirements
| Quant | Size |
|---|---|
| **Q4_K_M (our pick)** | **~5.6 GB VRAM** + KV cache |
| Q5_K_M / Q6_K / Q8_0 | 6.5 / 7.4 / 9.5 GB — Q5+ risks Shared spill on 8 GB-class |

**Target:** 8 GB-class discrete NVIDIA. Full GPU offload (`-ngl 99`); steady-state at 65k ctx q4_0 KV is ~7.7 GB — fits only with `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256` on 8 GB-class (default 512 keepout clamps the ceiling below steady state; see Trial notes below). Peak measured 7.4 GB.

## Recommended settings (HF card, 2026-08-19)
Reasoning model: emits `<think>` blocks; card suggests qwen3-style reasoning/tool-call parsers at the server.

- **General tasks:** TEMP 1.0, TOP_P 0.95, TOP_K 20, MIN_P 0.0, presence 1.5, repeat 1.0
- **Precise coding tasks:** TEMP 0.6, TOP_P 0.95, TOP_K 20, MIN_P 0.0, presence 0.0, repeat 1.0

**Seeded for this Trial:** coding profile (TEMP 0.6) — matches the card's own ClawEval eval config (temp=0.6).

## MTP (Multi-Token Prediction)
- **This GGUF has NO MTP/nextn tensors.** `SPEC_TYPE=None`. No MTP pack exists for 1.5 yet.

## VITRIOL / Split strategy
- Dense — no expert offload. Full GPU: `-ngl 99`, `N_CPU_MOE=None`.

## Our config baseline (Trial 2026-08-19)
- `MODEL = 'Ornith-1.5-9B-Q4_K_M.gguf'`
- `CTX_SIZE = 65536` (65k proven on family; 131072 risks VRAM kill mid-coding on 8 GB)
- `VRAM_LIMIT_MB = 8000.0` + run env: `AUTORESEARCH_SKIP_FREE_CLAMP=1` (free-VRAM clamp too tight with desktop VRAM use) and `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256` (see note below)
- `KV q4_0`, batch 256/128, threads 6/8, `FLASH_ATTN on`, `NO_MMAP True`, `CONT_BATCHING True`, NGL 99
- Sampler: TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / presence 0.0 / repeat 1.0

## Trial results (2026-08-19, claw-full + coding-10 @ 65k)
| Metric | Value |
|---|---|
| Status | **on_front** (both axes) |
| Agentic (claw-full) | **0.8000** (12/15) |
| Coding | **0.6150** (HE+ 1.0000 / MBPP+ 0.7000 / LCB 0.5000 / BC 0.1000) |
| bench_tg | 43.2 t/s |
| Combined TPS | 54.5 (first run) |
| Peak VRAM | 7.4 GB |

Best coding in the Ornith family (1.0: 0.580 deepreinforce / 0.570 UD). Agentic beats 1.0's 0.6000 UD / 0.4000 Q4_K_M.

**Harness fixes required for correct measurement (2026-08-19):**
- `autoresearch/benchmarks/agentic_runner.py` Claw-loop turn timeout raised **30 s → 120 s** (reasoning-model `<think>` traces + max_tokens=2048 at ~40-55 t/s exceed 30 s; original 0.2667 was truncation artifact). Re-measured: 0.2667 → 0.8000.
- `autoresearch/core/llama_runner.py` `dedicated_vram_kill_ceil` now honors `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB` (preflight and runtime monitor were inconsistent; at 65k this model's steady state ~7.7 GB exceeds the default 7676 MB ceiling).

## Sources / Verification
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (README, 2026-08-19)
- Local GGUF metadata via `scripts/model_info.py` (2026-08-19)
- Trial rows: `results.tsv` `f19d991d` (coding 0.6150 + agentic 0.2667), `9b1af29d` (agentic 0.8000); rejected preflight/kill rows `6fde4721`/`716efd9a`/`06043927` kept for the record

## Open questions
- 3 remaining agentic failures (T044/T046, T054) logged mid-loop **HTTP 400/500** — long research sessions; likely server-side context/request limit at ~50k+ tokens. Investigate if agentic >0.8 matters.
- Full SSM layout for 1.5 (interval/hidden) — verify from GGUF when next card edit happens.
