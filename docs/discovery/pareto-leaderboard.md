# Pareto Frontier Leaderboard (Local Rig)

Global **Pareto Set** on this hardware budget: maximize **ctx × TPS × agentic × coding** ([ADR 0006](../adr/0006-pareto-frontier-search.md)). Selection lenses: **Day** ([ADR 0009](../adr/0009-day-profile-tps-floor.md)) / **Night** ([ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md) Night rule).

Hardware: discrete **8 GB-class** NVIDIA, `VRAM_LIMIT_MB=7900`, Windows, upstream CUDA unless noted.  
Ground truth: the results store — canonical `results.db` (SQLite), legacy `results.tsv` fallback (`scripts\rebuild_results_db.py` keeps them in sync). TPS axis = claw-full `bench_tg` when available. Complete vector = claw-full **and** coding-10 (exact 10 tasks/dataset).

**Point = GGUF basename** (ADR 0012): max claw × coding × TPS × ctx across Trials for that file. Different quants stay separate. ENGINE+SAMPLER Fingerprint is a Baseline/Search hint, not Day/Night Point identity. Live recompute: `scripts/rank_results.py`.

Recompute live (do not invent temp scripts):

```bash
.\venv\Scripts\python.exe scripts\rank_results.py
.\venv\Scripts\python.exe scripts\rank_results.py --day-tps-floor 50
```

## Usage Profile picks (ADR 0009 Day / ADR 0008 Night)

| Lens | Rule | Pick (this front) |
| :--- | :--- | :--- |
| **Night** | `CTX ≥ 65536` then max `min(agentic, coding)`; if `agentic_coding` is measured, max `min(agentic, coding, agentic_coding)` ([ADR 0013](../adr/0013-agentic-coding-night-selector.md)) | Recompute live; snapshot below predates the SWE-lite column |
| **Day** | `TPS ≥ DAY_TPS_FLOOR` (default 50) then max `min(agentic, coding)` | Recompute live with `rank_results.py` (snapshot below used older ADR 0008 IQ band) |

## `on_front` (complete / merged)

**Snapshot synced from `scripts\rank_results.py` output 2026-08-23** (earlier hand-patched snapshot missed `LFM2.5-2.6B` and `Nemotron-Nano` front points — always recompute live; **TSV wins over this doc**). Sorted by `min(agentic, coding)` descending.

### Day lens (TPS ≥ 50)

| # | Model | ctx | TPS | agentic | coding | min |
|---|---|---|---|---|---|---|
| 1 | `Qwen3.8-4B-Q4_K_M.gguf` | 131k | 74.9 | 0.8667 | 0.6400 | **0.6400** ← DAY pick |
| 2 | `LFM2.5-2.6B-Q8_0.gguf` | 65k | 82.2 | 0.8667 | 0.5200 | 0.52 |
| 3 | `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf` | 131k | 79.2 | 0.7333 | 0.5100 | 0.51 |
| 4 | `granite-4.1-3b-Q4_K_M.gguf` | 131k | 92.0 | 0.6667 | 0.4300 | 0.43 |
| 5 | `Qwen3.5-4B-MTP-Q4_K_M.gguf` | 131k | 92.6 | 0.8667 | 0.4150 | 0.42 |
| 6 | `LFM2.5-1.2B-Instruct-Q8_0.gguf` | 128k | 180.6 | 0.6000 | 0.3700 | 0.37 |
| 7 | `SmolLM3-3B-Q4_K_M.gguf` | 131k | 110.0 | 0.5333 | 0.3650 | 0.365 |
| 8 | `LFM2.5-8B-A1B-Q4_K_M.gguf` | 65k | 185.8 | 0.2667 | 0.3800 | 0.27 |

### Night lens (CTX ≥ 65536)

Night adds the sub-floor-TPS high-IQ points that Day excludes:

| # | Model | ctx | TPS | agentic | coding | min |
|---|---|---|---|---|---|---|
| 1 | `Qwen3.8-4B-Q4_K_M.gguf` | 131k | 74.9 | 0.8667 | 0.6400 | **0.6400** ← NIGHT pick |
| 2 | `Ornith-1.5-9B-Q4_K_M.gguf` | 131k | 44.4 | 0.9333 | 0.6150 | 0.62 |
| 3 | `Ornith-1.0-9B-UD-Q4_K_XL.gguf` | 131k | 48.6 | 0.9333 | 0.5700 | 0.57 |
| 4–10 | same tail as Day (LFM2.5-2.6B → LFM2.5-8B) | | | | | |

Notable drops vs older snapshots: `POCKET-35B` (min 0.615 @35.7 t/s) and `KAT-Coder` (0.60 @30.2 t/s) are now **dominated** by Qwen3.8-4B (≥ every axis, higher ctx).

Exact domination membership can shift when TPS sources differ (claw vs coding Combined TPS). Treat the table as the teaching front; recompute from the results store (`scripts\rank_results.py`) before deleting GGUFs. **The store wins over this doc.**

## `dominated` (complete, someone covers)

- `POCKET-35B-Q3_K_M.gguf` (0.6667 / 0.6150 @65k, 35.7 t/s) — covered by `Qwen3.8-4B-Q4_K_M`: higher ctx + agentic + coding + TPS (2026-08-23).
- `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` (0.6000 / 0.6400 @65k, 30.2 t/s) — covered by `Qwen3.8-4B-Q4_K_M`: same coding, higher agentic + ctx + TPS (2026-08-23).
- `Ornith-1.5-35B-A3B-Heretic-MTP-APEX-I-Mini.gguf` (0.8667 / 0.5300 @65k, 34.9 t/s, 2.5 GB) — covered by `Qwen3.8-4B-Q4_K_M`: higher coding + TPS + ctx at equal agentic (2026-08-23).
- `Ling-3.0-tiny-Q4_K_M.gguf` (0.8667 / 0.3900 @65k, 52.8 t/s, 2.5 GB) — covered by `Qwen3.8-4B-Q4_K_M`: higher coding + TPS + ctx at equal agentic; kept as VRAM-efficient fallback (2026-08-23).
- `Ornith-1.0-35B-UD-Q3_K_XL.gguf` (0.4667 / 0.5550 @65k) — covered by KAT: same ctx, better agentic + coding + TPS.
- `Qwythos-9B-Claude-Mythos-5-1M-MTP.Q4_K_M.gguf` (0.2667 / 0.5500 @65k) — covered by v2-MTP: higher ctx + agentic + TPS.
- `POCKET-26B-Q4_K_M.gguf` (0.2000 / 0.4900 @65k) — covered by POCKET-35B.
- `Qwen3.8-4B-Heretic (model-Q4_K_M.gguf)` (0.6667 / 0.6400 @131k) — covered by base `Qwen3.8-4B-Q4_K_M`: same ctx + coding + TPS, higher agentic. Abliterated variant not superior.
- `MindSparQ-Coder-1.5B.Q4_K_M.gguf` (0.0000 / 0.0250 @65k) — 1.5B too small for tool use; complete but dominated everywhere.

Note: `SmolLM3-3B` is **not** in this list — it survives the front via its TPS axis (110 > 74.9), so it appears in the Day table above despite lower IQ axes. Domination here is strict Pareto over ctx × TPS × agentic × coding (`autoresearch/core/pareto.py::dominates`); a point with any single higher axis stays on the front.

## `incomplete` / rejected

| Model | Gap | Status |
| :--- | :--- | :--- |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | coding | **Rejected** — VRAM kill @ 32k+MTP and 65k no MTP mid coding-10 (attempt rows in TSV). |
| `deepgrove/maple-preview-TQ1_0-head-Q4_K.gguf` | load | **Rejected** — `unknown model architecture: 'maple'` on b10549; needs upstream arch support. |
| `ornith-1.5-9b-function-calling-xlam-unsloth.q2_k.gguf` | load | **Rejected** — `blk.32.attn_norm.weight not found`; truncated/corrupt Q2_K quant. |
| `cesium2-v7-q8_0.gguf` | coding | **Rejected** — coding preflight 0/40 tasks (not instruction-tuned for code); agentic quick 0/5. |
| `Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf` | coding-10 | claw-full **0.4000** in TSV; **no** fair coding-10 row |
| `nanbeige4.2-3b-Q4_K_M.gguf` | coding-10 | claw-full **0.2667** in TSV; **no** fair coding-10 row |
| `Qwythos-9B-v2*` failed trials | — | reclassified `incomplete` → **`rejected`** in results.tsv (MODEL_REJECTED / INFRA_ERROR) |

**Config-split history** — pre-0012, agentic and coding under different Baselines never merged. ADR 0012 merges max axes by basename; prefer remeasuring both axes under a Preferred Baseline when reproducing a single Fingerprint.

## Quantizations are separate Trials
Different quants of the same family (e.g. Ornith-35B **Q3_K_XL** vs **Q4_K_XL**) are **not** duplicates. Each needs its own Objective Vector. Prefer the better quant for aliases; keep both scores in leaderboards until a delete decision.

## Prefer by job (the operator host)

| Job | Prefer |
| :--- | :--- |
| Night / long agent loops | **Qwen3.8-4B-Q4_K_M** (Ornith-1.5-9B close second: higher raw agentic, lower coding) |
| Day / supervised (ADR 0009) | **Qwen3.8-4B-Q4_K_M** — only point clearing the 50 t/s floor above min 0.6 |
| Direct coding | Qwen3.8-4B (coding 0.64 ties KAT at 2.5× its TPS) → POCKET-35B |
| Low-VRAM co-run / fallback | **Ling-3.0-tiny** — agentic 0.8667 at 2.5 GB peak |
| SSD cleanup | delete weak/`rejected` first (`Cesium`, `MindSparQ`, corrupt `Ornith-FC` Q2_K, unloadable `maple` TQ1_0) |

## See also

* [ADR 0006](../adr/0006-pareto-frontier-search.md) — Pareto Set membership  
* [ADR 0009](../adr/0009-day-profile-tps-floor.md) — Day TPS floor → max IQ  
* [ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md) — Night ctx floor / historical Day IQ ε-band  
* [pareto-selection.md](pareto-selection.md) — method citations (maximin / Day floors)  
* [claw-eval-leaderboard.md](claw-eval-leaderboard.md) · [coding-leaderboard.md](coding-leaderboard.md)  
* Session: [2026-07-27 incomplete vectors + Pareto](../sessions/2026-07-27-incomplete-vectors-pareto.md), [KAT-Coder pipeline](../sessions/2026-07-27-kat-coder-v2.5-dev-pipeline.md)
