# Ornith-1.5-9B — Model Card (Local)

**Source repo:** https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF
**HF URLs (fetched 2026-08-29):** https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (GGUF repo) · https://huggingface.co/ornith-ai/Ornith-1.5-9B (base model repo)
**License:** MIT (https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/LICENSE)
**Local file:** `models/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf` (5.63 GB)
**Family:** Ornith-1.5 (ornith-ai; Qwen 3.5 architecture)
**Quantization:** official Q4_K_M (only official GGUF repo; no Unsloth/MTP pack yet)
**MTP repo:** none — dense qwen35; the Q4_K_M GGUF had no MTP tensors at the 2026-08-22 scan (0/427 `nextn`); an upstream re-upload later carried an MTP head, but our local file is the original no-MTP artifact (see Architecture note).

## Architecture (verified from local GGUF, `gguf.GGUFReader` + `autoresearch/core/model_arch.py:126` `gguf_has_mtp()`, 2026-08-22)
- Causal LM, Qwen 3.5 arch (`qwen35.*` fields), **dense**
- **`block_count` = 32**
- `kv_f16_mb @ 65536 ctx` = 4096 MB (q4_0 KV ≈ 1 GB at 65k)
-
**2026-08-25 artifact swap:** official `Q4_K_M` was re-uploaded and now carries an MTP head — 4 `nextn` tensors, `block_count` 33 (= 32 + 1 MTP layer; GGUF `block_count` counts the MTP head; base config `num_hidden_layers` is 32), 442 tensors (verified `GGUFReader`). Old no-MTP file kept as `Ornith-1.5-9B-Q4_K_M.premtp-fix.gguf`. Measured vectors (agentic 0.9333 / coding 0.6150) refer to the **old** artifact; remeasurement pending. Head verified **trained** 2026-08-26 (harness ladder): acceptance 0.624, decode **61.6 t/s = +49.5 %** vs control @80k — the family dense-MTP win is real. MTP @100k ctx peaks 7.92 GB (rejected by the 7676 keepout); 80k is the 8 GB-class ceiling with the head.
- **TBD:** exact SSM/attention layout (full_attention_interval, hidden dim) — same file size and arch family as Ornith-1.0-9B Q4_K_M; see [ornith-1.0-9b.md](./ornith-1.0-9b.md) for the family layout.

## Hardware requirements
| Quant | Size |
|---|---|
| **Q4_K_M (our pick)** | **~5.6 GB VRAM** + KV cache |
| BF16 (publisher) | ≈ 19 GB VRAM (single 80 GB GPU per HF card) |

**Publisher hardware (https://huggingface.co/ornith-ai/Ornith-1.5-9B, fetched 2026-08-29):** BF16 (≈19 GB) targets a single 80 GB GPU; quantized Q4_K_M / Q5_K_M / Q6_K / Q8_0 are quantized variants in the GGUF repo (file sizes inferred from GGUF directory listing — see `models/ornith-ai/Ornith-1.5-9B-GGUF/`). The 8 GB-class envelope is **not** publisher-supported; the 1.0 / 1.5 family are validated on ≥40 GB. The HF README publishes a quantized `Ornith-1.5-9B-Mobile` variant explicitly for edge/mobile deployment.

## Recommended settings (HF card, 2026-08-19; refetched 2026-08-29)
Reasoning model: emits `<think>` blocks; card suggests qwen3-style reasoning/tool-call parsers at the server. Publisher runtime floors: vLLM ≥ 0.19.1 / SGLang ≥ 0.5.9 / Transformers ≥ 5.8.1. Serving recipes use `--reasoning-parser qwen3` + `--tool-call-parser qwen3_xml` (vLLM) or `qwen3_coder` (SGLang) to surface `reasoning_content` and `tool_calls` as OpenAI-style fields.

- **General tasks:** TEMP 1.0, TOP_P 0.95, TOP_K 20, MIN_P 0.0, presence 1.5, repeat 1.0
- **Precise coding tasks:** TEMP 0.6, TOP_P 0.95, TOP_K 20, MIN_P 0.0, presence 0.0, repeat 1.0

**Context window (publisher claim, 2026-08-29):** 262,144 tokens native (HF `gguf.context_length=262144`); YaRN `rope_scaling` with `factor: 4.0` extends the effective window to ~1,048,576 tokens (1M). Publisher validates YaRN on bf16; **TBD:** whether YaRN ships in the Q4_K_M GGUF config — our Q4_K_M file inherits the bf16 RoPE, llama.cpp applies scaling at server start.

**Publisher benchmarks (https://huggingface.co/ornith-ai/Ornith-1.5-9B, fetched 2026-08-29):** 5-run averages on the bf16 checkpoint with vLLM / SGLang / OpenHands / Claude-Code harnesses.
- Coding: Terminal-Bench 2.1 (Terminus-2) **46.2** · Terminal-Bench 2.1 (Claude Code) **47** · SWE-bench Verified **70.6** · SWE-bench Pro **47.5** · SWE-bench Multilingual **54.4** · NL2Repo **32.4** · SWE Atlas QnA **20.6**
- Reasoning: HLE no-tools **20.2** · HLE with-tools **30.5** · GPQA Diamond **86.4**
- Agentic: MCP-Atlas **54.2** · Toolathlon-Verified **41.2** · WideSearch **59.5** · BrowseComp **56.4** · ClawEval **66.5**

Eval notes (verbatim from HF card, 2026-08-29): Terminal-Bench 2.1 uses `parser=json, temperature=1.0, top_p=1.0, 128K ctx, 4-hour timeout, 32 CPU cores, 48 GB RAM, 5-run avg`. SWE-bench uses OpenHands harness `temp=1.0, top_p=0.95, 256K ctx` with anti-hacking safeguards (no Git history, no network). ClawEval uses `temp=0.6, 256K ctx`. These are **publisher** numbers on the bf16 model at 128k–400k ctx — not our Q4_K_M measurements.

**Seeded for this Trial:** coding profile (TEMP 0.6) — matches the card's own ClawEval eval config (temp=0.6).

## Reasoning control

**Template reads only `enable_thinking`** (verified 2026-08-29, embedded `tokenizer.chat_template` via `gguf_dump.py --no-tensors --json` on the local Q4_K_M GGUF): the template contains zero `reasoning_effort` / `thinking_budget` variables, so `--reasoning-effort` is a **silent no-op** on this GGUF. The card already documents the 2026-08-24 no-op finding (see [Trial 2026-08-24](#trial-2026-08-24--reasoning_budget4096-ab-claw-full--coding-10--65k) below); this 2026-08-29 re-confirmation is the third independent check.

**Working levers on this GGUF:**
- `--reasoning on|off` (Baseline `REASONING`): enables/disables the `<think>` block at the server (the rendering itself is a `--reasoning` / `enable_thinking` toggle).
- `--reasoning-budget N` (Baseline `REASONING_BUDGET`): template-independent think cap; server-side forces the end-of-think tag at exhaustion. 4096 seeded for the 2026-08-24 A/B; 2048 in the daily-driver alias; 2048+message in operator-bumped alias.
- `--reasoning-budget-message "..."` (Baseline `REASONING_BUDGET_MESSAGE`): nudge injected at budget exhaustion — prevents silent truncation.
- `--reasoning-preserve` (Baseline `REASONING_PRESERVE`): full-history think preservation; `GET /props` reports `chat_template_caps.supports_preserve_reasoning = true` for this GGUF (sealed in Baseline 2026-08-19, see [Overthinking behavior](#overthinking-behavior--daily-driver-levers-2026-08-25) below).

## MTP (Multi-Token Prediction)
- **MTP head present and trained (2026-08-25 swap + 2026-08-26 probe).** 4 `nextn` tensors, `block_count` 33 (= 32 + 1 MTP layer), `SPEC_TYPE='draft-mtp'` via `--spec-type draft-mtp --spec-draft-n-max 2` (llama.cpp never auto-enables). Measured 2026-08-26 (b10549, harness ladder): @80k pp 1755 t/s, tg **61.6** (+49.5 % vs 41.2 control @80k), acceptance **0.624**; @100k MTP peaks 7924 MB > keepout 7676 — VRAM-rejected, 80k is the ceiling. n-max 4 vs 2 VRAM delta negligible (7961 vs 7924 @100k).

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
## Trial 2026-08-24 — REASONING_BUDGET=4096 A/B (claw-full + coding-10 @ 65k)
Requested "reasoning effort medium": `--reasoning-effort` is a **silent no-op on this GGUF** — embedded `tokenizer.chat_template` (7828 chars, `<think>`-based) contains zero `reasoning_effort` variables (verified via `gguf_dump.py --no-tensors --json`; b10549 server accepts `LLAMA_ARG_REASONING_EFFORT`, template never reads it). The harness had no `REASONING_EFFORT` knob at the time (plumbed 2026-08-29 as a config.py-only Baseline key; still a silent no-op on this GGUF's template). Nearest working lever = `--reasoning-budget`; seeded 4096 (operator-instructed; prior Trial rows ran `reasoning_budget:null`).

| Metric | Value |
|---|---|
| Status | **on_front** (row `56a3c78f`) |
| Agentic (claw-full) | **0.7333** (11/15) |
| Coding | **0.6050** (HE+ 0.9000 / MBPP+ 0.9000 / LCB 0.4000 / BC 0.1000) |
| bench_tg | 44.1 t/s |
| Combined TPS | 53.6 |
| Peak VRAM | 7.0 GB |

Verdict: budget 4096 did **not** lift agentic vs 2026-08-19 (0.7333 vs 0.9333); coding within noise (0.6050 vs 0.6150). T046 (0.20) and T053 (0.00) failed on HTTP 400 `exceed_context_size_error` at 69052/104611 prompt tokens — `REASONING_PRESERVE` think re-render inflation, the Open Questions risk, now confirmed on 1.5-9B; T048 (0.40) / T054 (0.40) hit `length` stops. Same config minus budget already holds the better vector; no rerun recommendation from this A/B.

Best coding in the Ornith family (1.0: 0.580 deepreinforce / 0.570 UD). Agentic 0.9333 beats 1.0-9B UD's fair rerun **0.8667** (2026-08-19, same 4096 floor + TEMP 0.4); 1.0's older 0.9333 was a 2048-cap run whose success was run variance, not a floor artifact — the fair remeasure scored lower, not higher.

**Harness fixes required for correct measurement (2026-08-19):**
- `autoresearch/benchmarks/agentic_runner.py` Claw-loop turn timeout raised **30 s → 120 s** (reasoning-model `<think>` traces + max_tokens=2048 at ~40-55 t/s exceed 30 s; original 0.2667 was truncation artifact). Re-measured: 0.2667 → 0.8000.
- Agentic `max_tokens` floor raised **2048 → 4096** in `autoresearch/runners/evaluation.py` + `agentic_runner.py`; Claw turn timeout **240 s → 420 s** (2026-08-19). 1.5-class CoT still exhausted 2048 mid-`<think>` — server log `n_decoded = 2048` exactly; turns now decode 3000–4100 tokens. Re-measured: 0.8000 → **0.9333**. One mid-run turn hit the 65k ctx ceiling (`n_tokens = 65535, truncated = 1` at ~61k accumulated context) — that research task still passed; T054 (0.00) failed on retrieval, not truncation (`truncated = 0`, 45.8k ctx, 31 calls, no ARPPU keywords found). 65k ctx is the next limiter, not a 2048-token cap.
- `REASONING_PRESERVE=True` seeded in Baseline (2026-08-19): `GET /props` reports `chat_template_caps.supports_preserve_reasoning = true` for this GGUF; full-history think preservation for agentic continuity.
- `autoresearch/core/llama_runner.py` `dedicated_vram_kill_ceil` now honors `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB` (preflight and runtime monitor were inconsistent; at 65k this model's steady state ~7.7 GB exceeds the default 7676 MB ceiling).

## Overthinking behavior + daily-driver levers (2026-08-25)

Operator benchmark sessions (2026-08-20/21, agent loops) are ~90 % thinking by volume: think blocks mean 3.4k chars/turn (median 1.6k, p90 8.7k, max 17.9k), answer+tool text mean 406 chars; one session had a single 146,893-char think (~35k tokens ≈ 13 min at 44 t/s). Long thinks show **zero self-repetition** (no repeated 8-grams) — verbose RL-trained deliberation, not a repetition loop. Vendor evaluates with max_new_tokens up to 131k / 256k ctx, so unbounded thinking was rewarded.

Lever verdicts (mechanisms verified in pinned-build source, 2026-08-25):
- `--reasoning-budget` (already 4096 in the daily-driver alias, operator-bumped from 2048) is the only direct think cap; it forces the end-of-think tag at exhaustion and is template-independent. 4096 ≈ 93 s/turn at 44 t/s. Add `--reasoning-budget-message "Stop thinking and act now."` — forced cutoff without a nudge degrades quality. Per-request `reasoning_budget_tokens` (OpenAI API) allows per-task override without restart.
- Repeat/presence penalties are windowed to the last 64 tokens by default (`--repeat-last-n` 64) — the card's `presence 1.5` / alias `repeat 1.15` cannot see a 4k think as configured; raise `--repeat-last-n` (e.g. 2048) if A/B-ing presence. Measured zero repetition ⇒ modest expected effect.
- KV q4_0→q8_0 does NOT address think length; on this hybrid arch q4_0@131k measured ≥ q8_0@131k (family A/B 2026-08-25), and q8_0 KV @131k adds ~2.1 GB (peak ~9.5 GB) — physically unloadable on 8 GB-class. Not a loop lever.
- Suggested active profile: budget 2048 + budget message; per-request `reasoning_budget_tokens` override for hard tasks. **TBD:** think-token distribution at budget ∈ {1024, 2048, 4096} × {no message, message}.

## Ornith family comparison (2026-08-19, 8 GB-class, results.tsv ground truth)

| Model (quant) | Agentic (claw-full) | Coding (10 tasks) | bench_tg | Peak VRAM | Status |
|---|---|---|---|---|---|
| **1.5-9B Q4_K_M** | **0.9333** (14/15, @4096) | **0.6150** | 43.2–44.3 | 7.2–7.4 GB | on_front |
| 1.0-9B UD Q4_K_XL | 0.8667 (13/15, @4096 rerun) · 0.9333 (@2048 cap run) | 0.5400 @65k · 0.5700 @32k | 41.6–48.6 | 7.7–7.8 GB | on_front |
| 1.0-9B Q4_K_M (deepreinforce) | 0.4000 | 0.5800 | 42.5 | 7.8–7.9 GB | — |
| 1.0-9B MTP Q4_K_M | 0.4667 | 0.5800 | 64.0 (draft) | 7.4 GB | — |
| **1.5-35B Q4_K_M** (MoE `n-cpu-moe`) | **0.8667** (13/15, @4096 rerun) | **0.6300** | 25.4–28.8 | 3.6–4.0 GB | on_front |
| 1.0-35B UD Q4_K_XL | 0.7333 (tsv) | 0.5800 | 22.7–25.7 | 4.9–5.2 GB | on_front |
| 1.0-35B UD Q3_K_XL | 0.7333 (tsv) | 0.5550 | 24.1–26.0 | 4.4–5.2 GB | on_front |

**Verdict:**
- **1.5-9B Q4_K_M is the best Ornith for 8 GB-class** — top agentic, top coding, lowest VRAM of the 9B tier.
- **Agentic, fair-measured**: 1.5-9B **0.9333** beats 1.0-9B UD's 0.8667 on the same 4096 floor (2026-08-19 rerun, TEMP 0.4 per 1.0 card). 1.0's older 0.9333 @ 2048 was run variance — the equal-cap remeasure scored lower, refuting the "cap understatement" hypothesis. T053 (1.0 FAIL 0.20 vs 1.5 PASS 0.70) and T054 (both fail) drive the gap.
- **1.5 wins coding everywhere**: 9B +0.075 vs UD, +0.035 vs deepreinforce Q4_K_M; 35B +0.050 vs Q4_K_XL.
- **35B tier**: 1.5-35B **0.8667** beats 1.0-35B's 0.7333 after the 4096 fix (was tied 0.7333 @2048 cap); faster (25.4–28.8 vs 22.7–25.7 t/s), better coding (0.630 vs 0.580), lower VRAM (3.6–4.0 vs 4.9–5.2 GB). 1.0-35B card body text (0.60 Q4 / 0.4667 Q3) predates the 2026-08-08 harness fix; tsv rows are 0.7333 for both quants.
- 1.0-9B MTP remains the **speed** pick only (64 t/s draft) at the cost of agentic.

## Sources / Verification
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (README, 2026-08-19)
- https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF (HF repo API JSON: tags, `safetensors.parameters`, `cardData.license`, downloads, `lastModified`, `siblings` — fetched 2026-08-29)
- https://huggingface.co/ornith-ai/Ornith-1.5-9B (base-model repo API + README.md: benchmarks, usage, recommended settings, hardware table — fetched 2026-08-29)
- Local GGUF metadata via `scripts/model_info.py` (2026-08-19) + `gguf.GGUFReader` field+tensor scan 2026-08-22 (`autoresearch/core/model_arch.py:126` `gguf_has_mtp()` → false, 0/427 `nextn` tensors, no `nextn_predict_layers`)
- Trial rows: `results.tsv` `f19d991d` (coding 0.6150 + agentic 0.2667), `9b1af29d` (agentic 0.8000 @ 2048 cap), `bf729951` (agentic 0.9333 @ 4096); rejected preflight/kill rows `6fde4721`/`716efd9a`/`06043927` kept for the record
**Extraction date:** 2026-08-29

## TPS Hill Climb — dense plateau (2026-08-22, b10549, 8 GB-class discrete NVIDIA, ctx >65k, bench-only no validation)

**Request:** hill climb TPS only, keep `CTX_SIZE > 65k`, no agentic/coding validation. Explicit harness `benchmark_search.py --no-coding --no-agentic-*` (`llama-cli` `n=512`, `c=4096` capped, `TPS_REPS=3` median) with `AUTORESEARCH_LLAMA_CPP_ROOT=llama.cpp-releases/upstream/b10549` + `AUTORESEARCH_SKIP_FREE_CLAMP=1` + `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256` (required — `estimate 100k q4_0=7418MB` vs ceiling 7932).

| Config (ctx, batch/ubatch, threads, KV, flags) | median t/s (3 reps) | harness log | decision |
|---|---|---|---|
| **100000, 512/128, 8/8, q4_0/q4_0, --no-mmap --flash-attn on --cont-batching** | **44.40 [44.3,44.4,44.4] 0.2%** | `llama-server-20260822-204923` (44.4) + `205115` (6/8 44.3) plateau pair | **winner — persisted to `autoresearch/core/config.py:30,38-39` (VRAM_LIMIT 8000)** |
| 70000, 512/128, 6/8, q4_0 | 44.4 [44.3,44.4,44.4] | `llama-server-20260822-203840` | tie — ctx flat 70k–120k |
| 110000, 512/128, 6/8, q4_0 | 44.3 | `llama-server-20260822-2039xx` | tie |
| 120000, 512/128, 6/8, q4_0 | 44.4 | `llama-server-20260822-204015` max viable q4_0 (7768<7932) | tie |
| 100000, 1024/512, 6/8, q4_0 | 44.3 | `llama-server-20260822-204150` | tie — batch neutral |
| 100000, 512/128, 6/8, q4_0 (vs 8/8) | 44.3 vs 44.4 | `205115` vs `204923` | tie ±0.1 (0.2% noise) — reconciled thread discrepancy (alias 6/8 vs probe leftover 8/8); standardized to 8/8 |
| 100000, 512/128, 8/8, q4_0, --mmap (NO_MMAP False) | 44.4 | `llama-server-20260822-205829` (8/8) + `205329` (6/8) | tie — mmap neutral |
| 100000, 512/128, 8/8, q4_0, --cont-batching off | 44.4 | `llama-server-20260822-205653` | tie — cont-batching neutral |
| 100000, 512/128, 8/8, q4_0, --flash-attn off | ConfigError `FLASH_ATTN must be on` | validate_config:99 | closed — not a valid axis |
| 100000, 512/128, 8/8, q8_0 | preflight FAIL `est 9106MB >7932` | bench harness | **reject** — VRAM (repeat confirmed 2026-08-22) |
| 131072, 512/128, 6/8, q4_0 | bench 44.4 but `est 7962>7932` preflight FAIL | — | needs keepout 0 (risky) — out of scope |
| 100000, 512/128, 6/8, f16 | 44.50 +0.10 | prior sweep | bench win but `est 11918>7932` not server-viable |

**Plateau:** dense 32-layer qwen35 at `Q4_K_M` saturates **44.3–44.4 t/s** on 8 GB-class discrete NVIDIA without MTP. The trained MTP head (2026-08-26) breaks the plateau: **61.6 t/s @80k ctx (+49.5 %)**, acceptance 0.624 — the family dense-MTP target. No `BATCH`/`UBATCH`/`THREADS`/`NO_MMAP`/`CONT_BATCHING`/`FLASH_ATTN`/`KV` lift beyond ±0.1 (0.2–0.7% noise). `CTX >65k` does not affect `bench_tg` (capped `c=4096`) — VRAM estimate is limiter. Viable `70k–120k q4_0` all same; `100000` chosen as ≥100k sweet spot, `8/8` threads persisted. Previous ad-hoc `hill_bench_only.py` plateau now reproduced via durable `benchmark_search.py` logs — not prose.

**Env for reproduction:** `AUTORESEARCH_LLAMA_CPP_ROOT=b10549`, `AUTORESEARCH_SKIP_FREE_CLAMP=1`, `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256`, `VRAM_LIMIT_MB=8000`, `TPS_REPS=3`.
## Open questions
- T054 (finance ARPPU) scores 0.00 for **all four Ornith variants** — 1.5-9B: 31 calls; 1.0-9B: 100 calls, len=154; 1.5-35B: 29 calls, report len=7173 (no keywords). Retrieval-path failure common to the family, not truncation. Possible target for a budget/efficiency A/B later.
- **65k ctx ceiling is the next limiter (now proven)**: the 1.5-35B rerun hit an HTTP 400 `exceed_context_size_error` — `request (124983 tokens) exceeds the available context size (65536)` (row `f8980537`, T053). `REASONING_PRESERVE` think re-renders inflate request size; candidate A/B: ctx 131072 or `REASONING_PRESERVE` off. Long research tasks can fill 65k at 4096 tokens/turn.
- Full SSM layout for 1.5 (interval/hidden) — verify from GGUF when next card edit happens.
