# 2026-08-19 — Agentic max_tokens 4096 + Ornith-1.5 remeasure

## Goal
Fix the Claw/agentic truncation artifact for reasoning models (answers starved at the 2048-token turn cap), verify on `Ornith-1.5-9B-Q4_K_M.gguf`, and settle the Ornith 1.0-vs-1.5 family comparison.

## Hardware
- Discrete 8 GB-class NVIDIA, Baseline `VRAM_LIMIT_MB = 8000`, dense + MoE (`--n-cpu-moe`) targets, Windows.
- Engine: `llama.cpp-releases/upstream/b10375` (CUDA build), run env `AUTORESEARCH_SKIP_FREE_CLAMP=1` + `AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB=256`.

## Setup
- Baseline (`autoresearch/core/config.py`, gitignored): `MODEL='Ornith-1.5-9B-Q4_K_M.gguf'`, `CTX_SIZE=65536`, `NO_MMAP=True` (dense), `KV q4_0`, batch 256/128, threads 6/8, `N_GPU_LAYERS=99`, sampler TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / presence 0.0 / repeat 1.0.
- Harness fix (commit `df57de4`): agentic `max_tokens` floor 2048 → 4096 (`autoresearch/runners/evaluation.py` + `autoresearch/benchmarks/agentic_runner.py`); Claw turn urlopen timeout 240 → 420 s.

## Commands
- Probe: llama-server with the 9B GGUF (ctx 65536, q4_0 KV, `--jinja`, `--no-mmap`) → `GET /props` → `chat_template_caps.supports_preserve_reasoning = true` → seeded `REASONING_PRESERVE=True` in Baseline.
- Rerun: `benchmark_search.py --agentic-full --no-coding --desc "trial Ornith-1.5-9B-Q4_K_M.gguf agentic rerun (max_tokens 4096)"` (no timeout; one Trial at a time).
- Local tests: `pytest tests/test_agentic_runner.py` → 13 passed.

## Findings
- **0.8000 → 0.9333 (14/15)**; row `bf729951` `on_front`, bench_tg 44.3, peak 7.4 GB. Run wall ~51 min.
- Previously HTTP-400/truncated tasks recovered: T044 **1.00** (report len 9674/800), T046 **0.75** (report len 2588/1000). T046 still misses remediation/real_world_impact keywords.
- Turn-length proof the cap was the bug: server log `n_decoded = 4123` on one research turn (old cap: exactly 2048, `eval time = 73978.76 ms / 2048 tokens`).
- **T054 (finance ARPPU) still 0.00** — NOT truncation: `truncated = 0`, final ctx 45.8k < 65k, 31 calls, no yearly-value keywords found. Retrieval-path failure; candidate for a later budget/efficiency A/B.
- **65k ctx ceiling hit once**: `n_tokens = 65535, truncated = 1` at ~61k accumulated context mid-run; that research task still passed. Next limiter for long research tasks at 4096 tokens/turn.
- `REASONING_PRESERVE=True` (Baseline): `/props` `supports_preserve_reasoning = true` for this GGUF; server banner also printed "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve".
- Family comparison (results.tsv): 1.5-9B 0.9333/0.6150 vs 1.0-9B UD 0.9333 (2048 cap)/0.5400; 1.5-35B 0.7333/0.6300 vs 1.0-35B UD 0.7333/0.5800. 1.0-35B card body (0.60/0.4667) predates the 2026-08-08 harness fix; tsv is ground truth.

## Errors
- Probe readiness regex never matched the actual "llama_server: listening" line — server was up; `/props` worked. Cosmetic.
- One 65k context truncation (`truncated = 1`) — recorded above, no `REASONING_BUDGET` added (capability policy).
- Prior-session HTTP 400/500 bodies were overwritten logs; root cause confirmed via `n_decoded` evidence instead.

## Decisions
- Raise `max_tokens` (capability), do **not** set `REASONING_BUDGET` for the verification rerun — measuring a model under an artificial think cap under-scores it. Efficiency profile (`REASONING_BUDGET` N) is a separate, documented Fingerprint if overthinking becomes the cost.
- Seed `REASONING_PRESERVE=True` (evidence: `/props`).
- Ship harness fix upstream (`df57de4`, pushed, CI green) — 6 files, pre-commit ruff + pytest passed.
- 35B rerun deferred; only after the 9B validates and the operator asks.
- 1.0-35B card body left as historical text with an explicit staleness note (tsv ground truth).
