# Agentic Coding Benchmarks

This repo evaluates local models across two tiers:

* **Preflight (direct code gen):** HumanEval+, MBPP+, LiveCodeBench, BigCodeBench.
  Optional single-turn checks. Exactly 10 tasks per dataset when enabled.
* **Agentic (multi-turn, tool use):** Claw-Eval quick/full tiers via `ClawEvalAdapter`
  in `autoresearch/runners/evaluation.py`. Canonical Val Score for Search.

## Tier Structure

| Tier | Tasks | Est. Time | Scoring | CLI Flag |
|------|-------|-----------|---------|----------|
| quick | 5 | ~5–10 min | Rule-based (tool_called, keywords, categories) | `--agentic-quick` / `--no-agentic-quick` |
| full | 15 | ~15–40 min (8 GB MoE/dense) | Rule-based (same) | `--agentic-full` / `--no-agentic-full` |

**Task selection policy:**
- English-only tasks (no zh variants)
- Rule-based scoring only (no `llm_judge` — fully local, no API keys)
- Quick tier: `difficulty=easy`, ≤2 mock services — observational smoke (no score-floor reject), not fair cross-model score
- Full tier: `difficulty=easy` first, then fills with `medium` — canonical Val Score
- Discovered at runtime from `claw-eval/tasks/` (local vendor tree)

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
```bash
python benchmark_search.py --agentic-full --desc "agentic quality gate"
```

Adapter (`ClawEvalAdapter`) starts mock services, runs the agent loop against the
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
