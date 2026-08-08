# 2026-08-08 — Thinking-model Claw harness bug + Ornith remasure

## Goal

Diagnose Ornith “feels smart” vs low Claw after re-download; fix agentic harness so thinking models are scored fairly; record remasure policy for other thinking GGUFs.

## Hardware

RTX 4060 **8 GB** class rig, Windows, upstream `llama.cpp` CUDA. Baseline `VRAM_LIMIT_MB` used for Pareto budget bucket (not peak alone).

## Findings

### Harness failure (agentic / Claw only)

- Loop ignored OpenAI-style `reasoning_content` when `content` was empty.
- Chat history stored `content` only → blank continuations → empty graders and mid-loop HTTP 400.
- Agentic `GenerationParams` defaulted to `max_tokens=512`; thinking ate the budget.
- Coding-10 already fell back to `reasoning_content` — asymmetric Objective Vector (coding looked fine, Claw looked dumb).

### Fix shipped on `main` (`52b05e3`)

- `agentic_runner.py`: surface + history round-trip for reasoning.
- `evaluation.py`: agentic `max_tokens ≥ 2048`.
- Also in same ship: Pareto maximize-axis merge = best remasured score; classify budget = `round(VRAM_LIMIT_MB/1024)`; `AUTORESEARCH_SKIP_FREE_CLAMP=1` escape for dense WDDM free-clamp.

### Ornith UD evidence (`Ornith-1.0-9B-UD-Q4_K_XL.gguf`, CTX 65536)

| Stage | claw-full | note |
| :---: | :---: | :--- |
| Pre-fix remasure | **0.3333** | False floor |
| Post-fix remasure | **0.9333** (14/15) | peak ~7.7 GB, TPS ~42.1 |

Durable guide: [thinking-models-claw-harness.md](../discovery/thinking-models-claw-harness.md). Card note: [ornith-1.0-9b.md](../models/ornith-1.0-9b.md).

## Decisions

1. **Not** remasure every model — only thinking / reasoning-default families (and log-confirmed empty-`content` cases).
2. Do **not** remasure coding-10 for this bug alone.
3. Document regression checklist in discovery so future thinking bugs are treated as harness suspects first.

## Errors / traps

- First-non-None Pareto merge kept stale **0.3333** after a better remasure — fixed to best/max on maximize axes before trusting status.
- Dense free-at-start VRAM clamp (issue #10) can false-reject when desktop holds VRAM; escape env above; runtime monitor remains kill guard.

## Follow-up

Claw-full remasure queue for thinking-family GGUFs (pre-fix scores suspect): see operator list in chat / derive from `results.tsv` using the discovery guide criteria. Start with Qwen3.5 / Qwen3.6 / Qwythos / remaining Ornith siblings / Gemma-4 / KAT-Coder / Nanbeige / Pocket.
