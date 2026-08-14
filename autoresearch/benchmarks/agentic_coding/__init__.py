"""Public exports for the SWE-lite agentic coding pack."""

from autoresearch.benchmarks.agentic_coding.detector import (
    ALLOWED_TOOLS,
    DetectorState,
    canonical_call,
    resolve_in_worktree,
    workspace_hash,
)
from autoresearch.benchmarks.agentic_coding.runner import (
    discover_tasks,
    load_task,
    run_agentic_coding_eval,
    run_task_loop,
)

__all__ = [
    "ALLOWED_TOOLS",
    "DetectorState",
    "canonical_call",
    "discover_tasks",
    "load_task",
    "resolve_in_worktree",
    "run_agentic_coding_eval",
    "run_task_loop",
    "workspace_hash",
]
