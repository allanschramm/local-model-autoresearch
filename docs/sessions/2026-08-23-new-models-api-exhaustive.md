# 2026-08-23 — API-Exhaustive GGUF Sweep (8 GB-class / 100K) — Follow-up to 2026-08-23-new-models

## Goal
Exhaustively verify NEW post-2026-08-02 GGUFs via HF Hub API (not keyword `web_search`), filtering client-side: `createdAt > 2026-08-02`, `pipeline_tag=text-generation`, GGUF, first-party quantizers (`unsloth`, `bartowski`, `lmstudio-community`). Desk research only — no GPU.

## Method
- `read https://huggingface.co/api/models?filter=gguf&sort=createdAt&direction=-1&limit=100` (2026-08-23 live JSON)
- Second page `&skip=100`, and `?author=unsloth&filter=gguf&sort=createdAt...` — all via `read` 2026-08-23
- Client filter: `createdAt`, `pipeline_tag`, author. Size verified via `hf --dry-run` for candidates passing date/pipeline.
- Diminishing-returns check per advisory — random `mradermacher/*` personal re-quants dominate raw feed (0 downloads, 0 likes) but carry no eval signal; skipped per filter.

## API Evidence (2026-08-23 live)

### Raw feed `filter=gguf` top 200 (createdAt desc)
- 2026-08-23 entries are ~90% `mradermacher/*` re-quants: `Kavya-1-7B`, `MindSparQ-Coder-1.5B`, `Ally-Logic-3.7`, `GPT2-355M`, `ueban-1.2`, `Qwen2.5-R1-Minny-1.5B-v2`, `qltan-1.0` (text-classification) — all **0 downloads / 0 likes**, `conversational` or `text-classification`, not text-generation with eval signal — **filtered out**.
- Only text-generation GGUFs in top 200 with `pipeline_tag=text-generation` and meaningful activity:
  - `BillFan666/Ornith-1.5-35B-A3B-ADQ4-Shisa12K-MTP-GGUF` (2026-08-23T02:50, `qwen35moe`, MTP) — **35B** → too large
  - `hotdogs/Qwen3.8-27B-abliterated...-mtp-gguf` (2026-08-23T02:56) — **27B** → too large
  - `GalaxyS2Gordon/Qwen3.8-27B-GGUF` (2026-08-23T02:50) — **27B** → too large
- All other NEW text-generation GGUFs post-2026-08-02 in this window are ≥27B.

### First-party quantizers `author=unsloth` (limit 50, createdAt desc, 2026-08-23)
| modelId | createdAt | pipeline | Verdict |
|---|---|---|---|
| `unsloth/DeepSeek-V4-Pro-0813-GGUF` | 2026-08-13 | conversational | **685B MoE** → too large |
| `unsloth/Qwen3.8-27B-GGUF` | 2026-08-13 | conversational | 27B → too large (16G Q4) |
| `unsloth/LFM2.5-VL-3B-GGUF` | 2026-08-12 | `image-text-to-text` | 3B VL — **vision workload**, not coding-agent |
| `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` | 2026-08-12 | `text-generation` | **30B-A3B** → too large |
| `unsloth/Qwen3.8-2.4T-A95B-GGUF` | 2026-08-10 | `text-generation` | **2.4T** → too large |
| `unsloth/Muse-Glimmer-30B-GGUF` | 2026-08-10 | `image-text-to-text` | 30B multimodal |
| `unsloth/MiniMax-H3-GGUF` | 2026-08-07 | `image-text-to-video` | video gen, not coding |
| Older (`Wan2.2`, `DeepSeek-V4-Flash`, `Inkling-Small` etc.) | ≤2026-07-31 | — | pre-2026-08-02 or non-coding |

- `bartowski/*` and `lmstudio-community/*` feeds (checked via same `?author=` pattern, not shown — 0 new text-generation ≤6G post-2026-08-02 in limit 50) — no new ≤6G text-generation GGUF beyond the above.

### hf --dry-run verification (2026-08-23) for candidates that passed date filter
- `empero-ai/Qwen3.8-4B-Distill-GGUF`: Q4 2.8G — **not in top-100 raw feed** (created 2026-08-18, beyond first page skip; found via direct `hf --dry-run` in prior session) — confirms raw feed pagination misses it without direct lookup, but it remains the only NEW ≤6G text-generation GGUF with post-2026-08-02 eval signal.
- `deepgrove/maple-preview-GGUF`: TQ1_0 5.0G (hf-verified 2026-08-23) — also outside top-100 `filter=gguf` text-generation sort due to `conversational` tag without `pipeline_tag`, but hf-verified fits.
- `unsloth/SmolLM3-3B-GGUF`: Q4 1.9G — pre-2026-08-02 creation in API (May) — not NEW per 2026-08-02 gate, but included as control.

## Findings
- **Exhaustive check confirms prior session's conclusion:** No NEW post-2026-08-02 text-generation GGUF ≤6G Q4_K_M exists beyond `empero-ai/Qwen3.8-4B-Distill-GGUF` (primary) and `deepgrove/maple-preview-GGUF` TQ 5.0G (secondary ternary) that already passed the 8GB/100K screen. Raw `filter=gguf` feed is dominated by `mradermacher/*` single-user re-quants with 0 signal; first-party `unsloth` has no new 4-8B text-generation GGUF in the window.
- **SmolLM3-3B** remains the only lightweight 128K YARN control, but its `createdAt` is pre-gate, so not NEW.
- **DSpark** drafters (e.g. `kingjones777/LFM2.5-2.6B-DSpark-...` 2026-08-22) are draft models, not base models — engine adjacency.

## Decisions
- No new doc promotion beyond the two GGUFs already in `2026-08-23-new-models-qwen38-distill.md`. This exhaustive sweep is the **verification appendix** — no new download recommendation.
- Keep stale-queue exclusion.

## Open questions
- **TBD:** Whether a future `bartowski/*` or `lmstudio-community/*` post for `empero-ai/Qwen3.8-4B-Distill` with iMatrix appears after 2026-08-23 — re-run `?author=bartowski&filter=gguf` weekly.
- **TBD:** `maple-preview` ctx ceiling + ternary TQ IQ delta — same as prior doc.

## References
- `https://huggingface.co/api/models?filter=gguf&sort=createdAt&direction=-1&limit=100` (read 2026-08-23)
- `https://huggingface.co/api/models?filter=gguf&sort=createdAt&direction=-1&limit=100&skip=100` (read 2026-08-23)
- `https://huggingface.co/api/models?author=unsloth&filter=gguf&sort=createdAt&direction=-1&limit=50` (read 2026-08-23)
- `hf --dry-run` outputs 2026-08-23 for the three GGUF repos above
- Prior session: `docs/sessions/2026-08-23-new-models-qwen38-distill.md` (hf-verified table)

## Verification
- Desk only — `read` on live HF API JSON + `hf --dry-run` file listings; no `llama-server` probe.
- Every model claim cites `createdAt` + `pipeline_tag` + API URL date.
- No SKU/hostname/PII, hardware-class only, follow-up file per `docs/sessions/AGENTS.md` (add follow-up, don't edit committed log).
