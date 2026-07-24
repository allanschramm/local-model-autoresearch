# Agent Onboarding Guide

Bootstrap context for agents working on this repo.

## Codebase Map

| File/Path | Role |
|:---|:---|
| `autoloop.py` | Autonomous hill-climbing runner. Mutates configs and loops forever. |
| `autoresearch/core/config.py` | **Only mutable file.** Holds current baseline configs. |
| `autoresearch/core/search.py` | Hill-climbing engine (`SearchStrategy`). Evaluates improvement. |
| `autoresearch/core/llama_runner.py` | Wrapper around `llama-server`. Handles port collision and VRAM telemetry. |
| `autoresearch/benchmarks/benchmark_coding.py` | Evaluates coding capabilities via LCB, HE+, MBPP+, and BigCodeBench. |
| `results.tsv` | Tab-separated database recording trial history. |

## Local Rules

1. **Be Terse**: Respond in smart caveman style (drop articles, filler, pleasantries).
2. **Loop Rule**: If running `autoloop.py` and a crash or code error occurs, **stop immediately**. Do not edit code to fix bugs during active search unless explicitly requested.
3. **No Pushing**: Never push results or config tweaks to remote branches. Keep all benchmark runs offline.
4. **DOX Framework**: Read the `AGENTS.md` hierarchy path to any file before touching it.

## Essential Commands

```bash
# Setup check
bash scripts/setup-check.sh

# Test suite
.\venv\Scripts\python.exe -m pytest tests/

# Default speed path (good enough, fast) — see docs/discovery/good-enough-tuning.md
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate <model>"
.\venv\Scripts\python.exe autoloop.py --mode tps --vram-limit-mb=<budget>

# Champion quality check (after TPS is acceptable)
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "champion claw-full"

# Single manual trial (Baseline already in config.py)
.\venv\Scripts\python.exe benchmark_search.py --desc "Hypothesis details here"
```

## Reading Order

1. `AGENTS.md` (root) — DOX hierarchy, work contracts
2. `program.md` — Search protocol rules
3. `GOLDEN-RULES.md` — Performance flags, safety, validation
4. `CONTEXT.md` — Terminology and definitions
5. `docs/discovery/good-enough-tuning.md` — **default** path: TPS first, quality second
6. `docs/discovery/discover-models.md` — Model selection workflow (which GGUF)
7. `docs/llamacpp-toolset.md` — llama.cpp binary reference
