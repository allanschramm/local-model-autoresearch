"""Harness-backed Trials reader for the dashboard (#25)."""

from __future__ import annotations

from typing import Any

from autoresearch.core.classify import is_on_front
from autoresearch.runners import run as run_mod


def read_last_50_trials() -> list[dict[str, str]]:
    """Last 50 results.tsv rows, newest first ([] when missing/empty)."""
    rows = run_mod.read_rows(run_mod.RESULTS_FILE)
    if not rows:
        return []
    # File order is append-chronological; newest last → reverse for UI.
    return list(reversed(rows))[:50]


def format_trial_for_ui(row: dict[str, str]) -> dict[str, Any]:
    """Operator columns; display keep as on_front; tolerate discard."""
    raw_status = row.get("status") or ""
    if is_on_front(raw_status):
        status = "on_front"
    else:
        status = raw_status
    return {
        "status": status,
        "outcome": row.get("outcome") or "",
        "ctx": row.get("ctx") or "",
        "tps": row.get("tps") or "",
        "agentic": row.get("agentic") or "",
        "coding": row.get("coding") or "",
        "memory": row.get("memory_gb") or "",
        "elapsed": row.get("elapsed_sec") or "",
        "diagnostic": row.get("diagnostic") or "",
        "description": row.get("description") or "",
    }
