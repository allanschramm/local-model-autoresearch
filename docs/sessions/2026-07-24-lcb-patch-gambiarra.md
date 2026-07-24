# Session 2026-07-24 — LCB patch (gambiarra)

## Goal

Re-measure LiveCodeBench only for three coding-10 rows that had `lcb=0` due to Windows symlink failure (`WinError 1314`), then patch `results.tsv` in place.

## Fixes landed first

1. `benchmark_coding._download_lcb_file`: `symlink_to` → `shutil.copy2`
2. `benchmark_coding._run_subprocess`: `timeout=30` (LFM hung forever on infinite-loop model code)

## Tool

`scripts/lcb_only.py` — Baseline from `config.py`, server via `LlamaServerRunner`, LCB n=10 only.

## Measured LCB (n=10)

| Model | LCB | HE/MBPP/BC (kept) | coding before → after |
| :--- | ---: | :--- | :--- |
| Laguna-XS @ 65k | **0.20** | 0.2 / 0.3 / 0.0 | 0.125 → **0.195** |
| LFM2.5-1.2B @ 65k f16 | **0.10** | 0.5 / 0.7 / 0.1 | 0.315 → **0.350** |
| Ornith-9B UD @ 32k | **0.40** | 0.9 / 0.7 / 0.2 | 0.430 → **0.570** |

Formula: `0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BC`.

## results.tsv patch

In-place on trials:

* `d8851b80-…` Laguna
* `9af96f3c-…` LFM
* `15f6bed0-…` Ornith 32k

Fields touched: `val_score`, `lcb_score`, `description` (`coding=` / `lcb=` + `lcb_patch=2026-07-24 scripts/lcb_only.py`).

**Caveat:** LCB was not co-timed with HE/MBPP/BC. Marked in description. Prefer full coding-10 re-run for pure Trials later.

Parent coding session: [2026-07-24-coding-10-claw-leaders.md](./2026-07-24-coding-10-claw-leaders.md).  
Leaderboard: [coding-leaderboard.md](../discovery/coding-leaderboard.md).

## Errors

* LFM first attempt hung (no subprocess timeout) — killed; timeout fix; re-ran OK.
* Ornith coding still requires **ctx 32k** on this rig (65k VRAM-kill during coding).
