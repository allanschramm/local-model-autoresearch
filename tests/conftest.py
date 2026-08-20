"""Seed local Baseline from the tracked template when missing."""

from __future__ import annotations

from pathlib import Path

import pytest

_CORE = Path(__file__).resolve().parents[1] / "autoresearch" / "core"
_CFG = _CORE / "config.py"
_EXAMPLE = _CORE / "config.py.example"

if not _CFG.exists() and _EXAMPLE.exists():
    _CFG.write_text(_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate_run_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Keep per-run log rotation + agentic sidecars out of the real repo dir.

    run_trial writes rotated llama-server logs and agentic-*.json sidecars under
    evaluation.LOG_DIR; Trial tests (mocked runners) must not leak them into
    autoresearch/runners/logs/.
    """
    monkeypatch.setattr("autoresearch.runners.evaluation.LOG_DIR", tmp_path / "logs")
