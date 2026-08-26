# 2026-08-26 Ornith MTP × cache ladder (new trained heads)

## Goal

Re-measure the 4-profile MTP × expert-cache ladder on the re-uploaded Ornith-1.5 GGUFs
(trained MTP heads, vendor 2026-08-23 fix) and compare against the 2026-08-24 ladder
measured on the old files (untrained 35B head / no 9B head). Question: does the trained
head turn MTP into a win? Profiles: NONE / MTP / CACHE / MTP+CACHE; metrics: prefill t/s,
decode t/s, draft acceptance.

## Hardware

- `discrete_gpu`, 8 GB-class. Baseline `VRAM_LIMIT_MB` 8000, keepout 512 → effective 7676.
- OS: Windows. Unified memory: 32 GB-class.
- Engines: 35B → codacus perf fork; 9B → upstream b10549.

## Setup

- Files: `models/ornith-ai/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf` (NEW,
  trained head) and `models/ornith-ai/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf`
  (NEW, head present). Old `*.premtp-fix` artifacts deleted per operator (2026-08-26).
- 35B config: ctx 131072, q4_0 KV, `n-cpu-moe 41` (auto from block_count), batch/ubatch
  512/512, threads 6, cache-reuse 256.
- 9B config: ctx 100000 / 80000, q4_0 KV, ngl 99, batch/ubatch 256/128, threads 6/8.
- MTP: `--spec-type draft-mtp --spec-draft-n-max 2`. CACHE: `--moe-cache-profile
  models/traces/ornith-1.5-35b-merged.csv --moe-cache-slots 48`.
- Harness change (this session): `MOE_CACHE_PROFILE` / `MOE_CACHE_SLOTS` config
  passthrough added to `autoresearch/core/llama_runner.py` (`ServerIntent` +
  `_build_cmd`) and `autoresearch/core/config.py.example` — all 4 profiles now
  expressible via config.py, no aliases required for the ladder.

## Commands (reproducible)

Probe pattern (per arm): `ServerIntent.from_config(config.load_config(), models)`
inside `LlamaServerRunner(intent)` context manager (harness owns launch + RAM watchdog +
VRAM keepout); POST `/v1/chat/completions` with a ~3431-token prompt, `max_tokens 256`,
temp 0.6; parse server-log `print_timing` lines (`prompt eval time` → pp,
`eval time` → tg, `draft acceptance`). Engine per arm via `AUTORESEARCH_LLAMA_CPP_ROOT`
(codacus fork for 35B, b10549 for 9B).

## Findings

35B Q4_K_M (codacus fork, q4_0 KV, n-cpu-moe 41, ub 512/512):

| Profile | config | prefill t/s | decode t/s | acceptance |
|---|---|---|---|---|
| NONE | ngl 99, ctx 131k | 232.1 | 31.0 | — |
| MTP (n-max 2) | ngl 99, ctx 131k | 213.5 | 28.0 | 0.567 |
| CACHE (48 slots) | ngl 99, ctx 131k | 204.2 | 36.9 | — |
| MTP+CACHE | ngl 36, ctx 115k* | 189.4 | 31.2 | 0.634 |

9B Q4_K_M (b10549, q4_0 KV, ub 256/128):

| Profile | ctx | prefill t/s | decode t/s | acceptance |
|---|---|---|---|---|
| NONE | 100k | 1794 | 40.6 | — |
| NONE | 80k | 1809 | 41.2 | — |
| MTP (n-max 2) | 80k | 1755 | 61.6 | 0.624 |

\* Stack VRAM: ngl 99 + ctx 131k peaks 7700–7896 MB (run-to-run ±200 MB variance) >
keepout 7676 (physical 8188 − 512). `ngl ≥ 41` is a no-op (41 = block_count incl. the
MTP head; 96 and 99 offload identically). `ngl 40` (head → CPU) frees only ~100 MB
(7793 — still over). Only `ngl 36 + ctx 115k` fits with real margin. The 2026-08-24
alias path ran the stack unguarded at 7.64 GB; the harness no-spill policy rejects it.

Verdicts:

1. **35B trained head: MTP still loses.** Acceptance 0.38 → 0.567 (+49%) but decode
   still −9.7% (28.0 vs 31.0) on the CPU-offloaded MoE. The head is no longer the
   bottleneck; MTP overhead is.
2. **35B cache-only is the winner** (36.9, +19% — matches the old-file number exactly).
   Stack cancels (31.2 ≈ control, carrying a ~4–5 ms/token CPU-attention penalty at
   ngl 36, so the true GPU-resident stack would be ~35–37 — consistent with "cancels").
3. **9B trained head: MTP is real.** 61.6 t/s = **+49.5%** vs control at matched 80k,
   acceptance 0.624 — the dense-family MTP win. VRAM: MTP @100k peaks 7924 (rejected);
   80k is the 8 GB-class ceiling. n-max barely moves VRAM (4: 7961 vs 2: 7924 @100k).

## Errors

- `VRAM_LIMIT_EXCEEDED` × 6 across stack/9B-MTP attempts — guard working as designed;
  the empirical ngl semantics (no-op ≥ block_count) are recorded above.
- Alias/driver approach abandoned this session per operator (harness only, no new
  files/aliases/queue scripts); `models/aliases/ladder-*` writes did not persist on the
  operator host — moot after the harness passthrough.
- First two alias-note edits wrote the read-tool elision marker (`…`) into the YAML as a
  literal char; removed via direct replace (the sparse-edit tool reserves `…`).

## Decisions

- 35B daily config: cache-only (48 slots), MTP off. Alias metrics refreshed to 36.9.
- 9B daily config: MTP n-max 2 @ ctx 80000 (was 131072). Alias updated (61.6 t/s);
  131k MTP is VRAM-infeasible on 8 GB-class.
- Old `*.premtp-fix` artifacts deleted per operator.
- `autoresearch/core/config.py` Baseline restored to the prior seed after the ladder.

## Cross-links

- [`2026-08-24-codacus-cache-full-offload.md`](./2026-08-24-codacus-cache-full-offload.md)
  — the old-file ladder this re-measures (cache +18–22 %, embedded MTP −25 % on the
  untrained head, stacked cancels).
- [`2026-08-24-codacus-fork-validation.md`](./2026-08-24-codacus-fork-validation.md)
- Model cards: [`../models/ornith-1.5-35b.md`](../models/ornith-1.5-35b.md),
  [`../models/ornith-1.5-9b.md`](../models/ornith-1.5-9b.md)
