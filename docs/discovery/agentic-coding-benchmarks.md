# Agentic Coding Benchmarks

This repo evaluates local models across three intelligence surfaces:

* **Preflight (direct code gen):** HumanEval+, MBPP+, LiveCodeBench, BigCodeBench.
  Optional single-turn checks. Exactly 10 tasks per dataset when enabled.
* **Agentic (multi-turn, office tools):** Claw-Eval quick/full tiers via `run_agentic_eval`
  in `autoresearch/benchmarks/agentic_runner.py` (orchestrated by `ExperimentRunner`).
  Claw full = the agentic Objective Vector axis
  ([ADR 0006](../adr/0006-pareto-frontier-search.md)); Val Score is legacy display, not Search truth.
* **Agentic coding (SWE-lite issue loop):** frozen GitHub-issue fixtures with workspace tools
  via `run_agentic_coding_eval` ([ADR 0013](../adr/0013-agentic-coding-night-selector.md)).
  Night selector when measured; not a Pareto axis in v1. Default **off**.

## Tier Structure

| Tier | Tasks | Scoring | CLI Flag |
|------|-------|---------|----------|
| quick | 5 | Rule-based (tool_called, keywords, categories) | `--agentic-quick` / `--no-agentic-quick` |
| full | 15 | Rule-based (same) | `--agentic-full` / `--no-agentic-full` |
| agentic-coding | 5 frozen issues | Hidden pytest; loop/hallucination = fail | `--agentic-coding` / `--no-agentic-coding` |

**Task selection policy:**
- English-only tasks (no zh variants)
- Rule-based scoring only (no `llm_judge` — fully local, no API keys)
- Quick tier: `difficulty=easy`, ≤2 mock services — observational smoke (no score-floor reject), not fair cross-model score
- Full tier: `difficulty=easy` first, then fills with `medium` — the agentic axis of the Objective Vector
- Discovered at runtime from `claw-eval/tasks/` (local vendor tree)
- Agentic coding: frozen fixtures under `autoresearch/benchmarks/agentic_coding/tasks/` (issue markdown + mini-workspace + hidden tests). No live GitHub. Pass = tests green and no detector flag.

## Current Code Hook

List approved benchmarks:
```bash
python benchmark_search.py --list-agentic-benchmarks
```

List Claw-Eval tier task IDs:
```bash
python benchmark_search.py --list-claw-tiers
```

Run quick agentic smoke test:
```bash
python benchmark_search.py --agentic-quick --desc "agentic smoke test"
```

Run full agentic quality gate:
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "agentic quality gate"
```

Run SWE-lite issue loop (Night selector, ADR 0013; default off):
```powershell
.\venv\Scripts\python.exe benchmark_search.py --agentic-coding --no-agentic-full --no-coding --desc "agentic-coding SWE-lite"
```

`ExperimentRunner` starts mock services (via `run_agentic_eval`), runs the agent loop against the
local OpenAI-compatible endpoint, and scores with deterministic `task.yaml` rules.
No Docker, remote judges, or external APIs.

## Scoring Rule

* Preflight coding score = `0.35*LCB + 0.25*HE + 0.25*MBPP + 0.15*BigCode`
* Agentic score = `passed / total` (simple pass@1, single trial for quick/full)
* Use exactly 10 tasks per dataset for preflight comparisons
* Claw-Eval full is the main quality gate; quick is smoke only
* Context: use Baseline `CTX_SIZE` that fits physical VRAM under agentic load. On 8 GB, **65k + KV q4_0** is the usual agentic band; 131k can smoke-pass then VRAM-kill mid-full (see Bonsai 2026-07-24)
* **LCB cache on Windows:** harness copies `test6.jsonl` into `autoresearch/data/benchmark_cache/livecodebench_v6/` (no symlink — `WinError 1314` without Developer Mode)
* **LCB / coding sandbox:** `_run_subprocess` uses `timeout=30` so infinite-loop model code cannot hang the coding bench forever

## Local leaderboard (8 GB)

* **Claw-Eval full (Val Score):** [claw-eval-leaderboard.md](claw-eval-leaderboard.md) — best Laguna-XS **0.6667**.
* **Coding-10 preflight:** [coding-leaderboard.md](coding-leaderboard.md) — best Mythos **0.6400**; current Ornith UD **0.5700** @ 32k.

### Coding CLI

```powershell
# Full 10-task preflight (HE+MBPP+LCB+BC)
.\venv\Scripts\python.exe benchmark_search.py --include-coding --no-agentic-quick --no-agentic-full --desc "coding-10 …"

# LCB-only remeasure (gambiarra / patch helper)
.\venv\Scripts\python.exe scripts\lcb_only.py
```

## Approved Targets

`claw-eval/` — local vendor checkout (not a git submodule; gitignored). 300 human-verified
autonomous-agent tasks across general, multimodal, and multi-turn splits.

Quick tier sample (auto-discovered, may vary):
- T002 email triage (1 mock service)
- T004 calendar scheduling (1 mock service)
- T006 email reply draft (1 mock service)
- T008 todo management (1 mock service)
- T010 contact lookup (1 mock service)
