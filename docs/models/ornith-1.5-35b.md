# Ornith-1.5-35B-A3B — Model Card (Local)

**Source repo:** https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF
**License:** MIT
**Local file:** `models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf` (21.7 GB)
**Family:** Ornith-1.5 (ornith-ai; Qwen 3.5 MoE architecture)
**Quantization:** official Q4_K_M (only official GGUF repo)

## Architecture (verified from local GGUF, harness-backed `model_info.py` 2026-08-19)
- Causal LM, Qwen 3.5 MoE arch (`qwen35moe.*`), **MoE**
- **`block_count` = 41** (1.0-35B was 40)
- **~35B total / ~3B activated** per card (MoE, 256 experts, 8 active + shared)
- `kv_f16_mb @ 65536 ctx` = 2624 MB (q4_0 KV ≈ 0.7 GB at 65k)
- **Embedded MTP:** `qwen35moe.nextn_predict_layers` present (verified field scan) — unlike 1.0-35B
- **TBD:** exact expert/hidden layout — verify full SSM/MoE params on next card edit

## Hardware requirements
| Quant | Size |
|---|---|
| **Q4_K_M (our pick)** | **21.7 GB file**; MoE expert offload → **peak 4.0 GB VRAM** |
| Q5_K_M / Q6_K / Q8_0 | 25.3 / 29.2 / 37.8 GB — bigger file, marginal expert-quality gain |

**Target:** 8 GB-class discrete NVIDIA + 32 GB-class RAM. Experts offloaded to CPU (`--n-cpu-moe 41` auto), active path on GPU (`-ngl 99`). Host-memory preflight est 21.7 GB fits. RAM is the binding resource (not VRAM).

## Recommended settings (HF card, 2026-08-19)
Reasoning model: emits `<think>` blocks; card suggests qwen3-style reasoning/tool-call parsers.

- **General tasks:** TEMP 0.6, TOP_P 0.95, TOP_K 20
- **To reproduce reported benchmarks:** TEMP 1.0

**Seeded for this Trial:** TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / presence 0.0 / repeat 1.0 (matches card's general profile and the 1.5-9B coding profile).

## MTP (Multi-Token Prediction)
- **Embedded `nextn` tensors present.** `SPEC_TYPE='draft-mtp'`, `SPEC_DRAFT_N_MAX=4` should work (llama.cpp embedded-MTP path, proven on qwen3.5-9B: +48% on NVIDIA). **Untested on this file** — separate fingerprint from the non-MTP vector; speed-path candidate.
- 1.0-35B had NO MTP — do not carry that assumption to 1.5.

## VITRIOL / Split strategy (MoE expert offloading)
- Auto `--n-cpu-moe 41` (GGUF block_count) — experts on CPU/RAM, attention + shared expert + routing on GPU.
- 1.0-35B A/B: n-cpu-moe=block_count beat manual 32. Peak VRAM 4.0 GB @ 65k.

## Our config baseline (Trial 2026-08-19)
- `MODEL = 'Ornith-1.5-35B-Q4_K_M.gguf'`
- `CTX_SIZE = 65536` (family complete-vector context; MoE keeps VRAM low)
- `VRAM_LIMIT_MB = 8000.0` + run env `AUTORESEARCH_SKIP_FREE_CLAMP=1`
- `KV q4_0`, batch 256/128, threads 6/8, `FLASH_ATTN on`, `NO_MMAP False` (mmap — 21.7 GB file pages), `CONT_BATCHING True`, NGL 99, `N_CPU_MOE=None` (auto → 41)
- Sampler: TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / presence 0.0 / repeat 1.0

## Trial results (2026-08-19, claw-full + coding-10 @ 65k)
| Metric | Value |
|---|---|
| Status | **on_front** |
| Agentic (claw-full) | **0.7333** (11/15) — stable across 120 s and 240 s turn-timeout runs |
| Coding | **0.6300** (HE+ 1.0000 / MBPP+ 0.9000 / LCB 0.4000 / BC 0.1000) |
| bench_tg | 25.4 t/s |
| Combined TPS | 36.8 |
| Peak VRAM | 4.0 GB |

Best coding in the Ornith family (1.5-9B 0.6150 / 1.0-35B 0.580). Agentic close to 1.5-9B's 0.8000, above 1.0-35B's 0.6000. Card's own ClawEval claims 72.5 — local 0.7333 aligns.

**Remaining agentic failures (4/15, web_research cluster):** T046/T048/T050/T054 — model loops on long research sessions without synthesizing a final report (T048: 42 calls, no timeout, empty report). Not timeout truncation (stable across 120 s/240 s). Possible TEMP 1.0 / presence lever.

## Sources / Verification
- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF (README, 2026-08-19)
- Local GGUF metadata via `scripts/model_info.py` + field scan (2026-08-19)
- Trial rows: `results.tsv` `53ba75b7` (complete vector on_front), `500e9967` (agentic rerun @240 s)

## Open questions
- MTP speed path (`SPEC_TYPE=draft-mtp`) untested — expected +20-46% TPS, separate fingerprint.
- Web-research looping: does TEMP 1.0 / presence penalty break the loop? (card benchmark repro uses TEMP 1.0)
