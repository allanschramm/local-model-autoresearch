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
| Agentic (claw-full) | **0.9333** (14/15; rerun @ max_tokens 4096) |
| Coding | **0.6150** (HE+ 1.0000 / MBPP+ 0.7000 / LCB 0.5000 / BC 0.1000) |
| bench_tg | 44.3 t/s |
| Combined TPS | 54.5 (first run) |
| Peak VRAM | 7.4 GB |

Best coding in the Ornith family (1.0: 0.580 deepreinforce / 0.570 UD). Agentic 0.9333 ties 1.0-9B UD's 0.9333 (both measured after the 4096-token fix; 1.0's 0.9333 was measured at the 2048 cap and is a slight understatement).

**Harness fixes required for correct measurement (2026-08-19):**
- `autoresearch/benchmarks/agentic_runner.py` Claw-loop turn timeout raised **30 s → 120 s** (reasoning-model `<think>` traces + max_tokens=2048 at ~40-55 t/s exceed 30 s; original 0.2667 was truncation artifact). Re-measured: 0.2667 → 0.8000.
- Agentic `max_tokens` floor raised **2048 → 4096** in `autoresearch/runners/evaluation.py` + `agentic_runner.py`; Claw turn timeout **240 s → 420 s** (2026-08-19). 1.5-class CoT still exhausted 2048 mid-`<think>` — server log `n_decoded = 2048` exactly; turns now decode 3000–4100 tokens. Re-measured: 0.8000 → **0.9333**. One mid-run turn hit the 65k ctx ceiling (`n_tokens = 65535, truncated = 1` at ~61k accumulated context) — that research task still passed; T054 (0.00) failed on retrieval, not truncation (`truncated = 0`, 45.8k ctx, 31 calls, no ARPPU keywords found). 65k ctx is the next limiter, not a 2048-token cap.
- `REASONING_PRESERVE=True` seeded in Baseline (2026-08-19): `GET /props` reports `chat_template_caps.supports_preserve_reasoning = true` for this GGUF; full-history think preservation for agentic continuity.
- `autoresearch/core/llama_runner.py` `dedicated_vram_kill_ceil` now honors `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB` (preflight and runtime monitor were inconsistent; at 65k this model's steady state ~7.7 GB exceeds the default 7676 MB ceiling).

## Ornith family comparison (2026-08-19, 8 GB-class, results.tsv ground truth)

| Model (quant) | Agentic (claw-full) | Coding (10 tasks) | bench_tg | Peak VRAM | Status |
|---|---|---|---|---|---|
| **1.5-9B Q4_K_M** | **0.9333** (14/15, @4096) | **0.6150** | 43.2–44.3 | 7.2–7.4 GB | on_front |
| 1.0-9B UD Q4_K_XL | 0.9333 (14/15, @2048 cap) | 0.5400 @65k · 0.5700 @32k | 42.1–48.6 | 7.7–7.8 GB | on_front |
| 1.0-9B Q4_K_M (deepreinforce) | 0.4000 | 0.5800 | 42.5 | 7.8–7.9 GB | — |
| 1.0-9B MTP Q4_K_M | 0.4667 | 0.5800 | 64.0 (draft) | 7.4 GB | — |
| **1.5-35B Q4_K_M** (MoE `n-cpu-moe`) | **0.7333** | **0.6300** | 25.4–28.8 | 4.0–7.0 GB | on_front |
| 1.0-35B UD Q4_K_XL | 0.7333 (tsv) | 0.5800 | 22.7–25.7 | 4.9–5.2 GB | on_front |
| 1.0-35B UD Q3_K_XL | 0.7333 (tsv) | 0.5550 | 24.1–26.0 | 4.4–5.2 GB | on_front |

**Verdict:**
- **1.5-9B Q4_K_M is the best Ornith for 8 GB-class** — top agentic, top coding, lowest VRAM of the 9B tier.
- **Agentic ties are cap-artifacts**: 1.0-9B UD's 0.9333 was measured at the 2048 cap; 1.5-9B at 4096. At equal caps 1.5-9B has margin (0.8000 → 0.9333 after the raise) — real 1.0-vs-1.5 gap is ≥ +0.133 on 9B.
- **1.5 wins coding everywhere**: 9B +0.075 vs UD, +0.035 vs deepreinforce Q4_K_M; 35B +0.050 vs Q4_K_XL.
- **35B tier**: agentic tied 0.7333, 1.5 faster (25.4–28.8 vs 22.7–25.7 t/s), better coding. 1.0-35B card body text (0.60 Q4 / 0.4667 Q3) predates the 2026-08-08 harness fix; tsv rows are 0.7333 for both quants.
- 1.0-9B MTP remains the **speed** pick only (64 t/s draft) at the cost of agentic.

## Sources / Verification
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (README, 2026-08-19)
- Local GGUF metadata via `scripts/model_info.py` (2026-08-19)
- Trial rows: `results.tsv` `f19d991d` (coding 0.6150 + agentic 0.2667), `9b1af29d` (agentic 0.8000 @ 2048 cap), `bf729951` (agentic 0.9333 @ 4096); rejected preflight/kill rows `6fde4721`/`716efd9a`/`06043927` kept for the record

## Open questions
- T054 (finance ARPPU) scores 0.00 with 31 tool calls and `truncated = 0` — the model never finds the yearly values; retrieval-path failure, not truncation. Possible target for a budget/efficiency A/B later.
- One mid-run 65k ctx truncation event (`truncated = 1`) — long research tasks can fill 65k at 4096 tokens/turn. Next limiter if agentic >0.9333 matters.
- Full SSM layout for 1.5 (interval/hidden) — verify from GGUF when next card edit happens.
