# 2026-09-07 — Qwen3.8-4B-Distill Q4_K_M Trial with --spec-type ngram-cache @131072 q4_0 (Issue #57)

## Goal
Implement and evaluate `--spec-type ngram-cache` as a Search neighbor on coding-10 + Claw-Eval full per GitHub Issue #57, comparing against the active baseline pick (`6069530a`) on the same Fingerprint.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, Windows host
- Engine: upstream `llama.cpp-releases/upstream/b10819` (CUDA 13.3)
- Gated DeltaNet hybrid architecture, dense `block_count 33`

## Setup
- Model: `models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf`
- Flags: `ctx=131072`, `kv=q4_0`, `flash_attn=on`, `batch=512`, `ubatch=128`, `threads=8`, `cont_batching=True`, `spec_type=ngram-cache`, `spec_draft_n_max=0`
- Sampler: `temp=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `repeat_penalty=1.0`

## Commands
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "trial Qwen3.8-4B-Q4_K_M @131072 q4_0 spec-type ngram-cache"
```

## Findings
- **Trial ID**: `0593e117-dffd-4bb8-85fb-b9b1cddbd923`
- **Bench throughput (`bench_tg`)**: 60.3 t/s (vs 74.9 t/s baseline, -19.5%)
- **Blended generation TPS**: 72.1 t/s (vs 94.2 t/s baseline, -23.5%)
- **Coding (10 each)**:
  - HumanEval: 8/10 (0.8000)
  - MBPP: 8/10 (0.8000)
  - LiveCodeBench: 5/10 (0.5000)
  - BigCodeBench: 1/10 (0.1000)
  - **Combined Coding**: `0.5900` (vs 0.6400 baseline, -7.8%)
- **Agentic Full (15 tasks)**: `13/15 (0.8667)` (tied with baseline 0.8667)
  - T002 0.50, T004 1.00, T006 1.00, T008 1.00, T010 1.00, T012 1.00, T014 0.70, T016 0.50, T018 1.00, T044 1.00, T046 1.00, T048 1.00, T050 1.00
  - Fails: T053 0.00, T054 0.00
- **Peak VRAM**: 5.6 GB (vs 5.4 GB baseline)
- **Host Memory & Allocation**:
  - `ngram-cache` allocates an unbounded dynamic n-gram map (`std::map<common_ngram, common_ngram_cache_part>`) in host RAM.
  - Observed RSS expansion of approximately ~160 MB per multi-turn request without LRU cache eviction.
- **Draft Acceptance Rate**:
  - Very low in practice on non-repetitive code and agentic tasks (~1.7% to 9.5% across tasks, peak 33.9% on repetitive boilerplate).
  - Target model verification passes in llama.cpp must evaluate drafted tokens and rewind on mismatch; on fast dense GPU-resident targets (~94 t/s baseline), this verification penalty exceeds the small latency savings from accepted drafts.

## Pareto Assessment & Verdict
- **Domination Analysis**:
  - Relative to Baseline Trial `6069530a` (`--spec-type none`) at the exact same Fingerprint:
    - Context: 131,072 = 131,072 (tied)
    - Agentic: 0.8667 = 0.8667 (tied)
    - Coding: 0.5900 < 0.6400 (strictly worse)
    - TPS: 72.1 < 94.2 (strictly worse)
  - Under multi-objective Pareto rules, candidate Trial `0593e117` is strictly **dominated** by `6069530a`.
- **Results Store Status**:
  - Recorded in `results.tsv` / `results.db` with `on_front` status due to ADR 0012 (all rows of a GGUF basename share the cluster's Pareto status).
  - The hill-climbing search rejects `ngram-cache` and keeps `--spec-type none` as the active Baseline.
