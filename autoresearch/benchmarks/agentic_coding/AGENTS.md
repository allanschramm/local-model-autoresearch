# `autoresearch/benchmarks/agentic_coding/` — SWE-lite issue loop

## Purpose
Frozen GitHub-issue fixtures plus the workspace tool loop and loop/hallucination detector that produce the `agentic_coding` Trial column (ADR 0013).

## Ownership
Repository developers. Parent: [`autoresearch/AGENTS.md`](../../AGENTS.md).

## Local Contracts
- Tasks are frozen snapshots (issue markdown + mini-workspace + hidden pytest). No live `gh` during a Trial.
- One session = one issue. Worktrees live in OS `$TEMP`, never under the repo.
- Hidden tests are not in the prompt. Pass = tests green and no detector flag.
- No Docker, no remote judge, no network tools.

## Work Guidance
- Add a task by copying an existing `tasks/<id>/` tree: `task.yaml`, `workspace/`, `hidden_tests/`.
- Detector rules live in `detector.py`; keep them deterministic and unit-tested.
- PATH tools must stay in the worktree **and** on `task.allowlist` (writes = exact file; reads/list/grep may use parent dirs of allowlisted files).
- Stall (`STALL_TURNS`) counts consecutive **mutating** turns with no allowlisted file-hash change. Read-only inspection does not stall.

## Verification
`.\venv\Scripts\python.exe -m pytest tests/test_agentic_coding.py`

## Child DOX Index
None
