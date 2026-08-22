"""Persisted in-flight Trial marker so a reboot-loop config can be skipped."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

JOURNAL_PATH = Path(__file__).resolve().parents[2] / ".autoresearch_crash.journal"


def write_journal(payload: dict[str, Any]) -> None:
    """Atomic tmp + fsync + replace."""
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{JOURNAL_PATH.name}.", dir=JOURNAL_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, JOURNAL_PATH)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def read_journal() -> dict[str, Any] | None:
    if not JOURNAL_PATH.exists():
        return None
    try:
        text = JOURNAL_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return None
        data = json.loads(text)
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def clear_journal() -> None:
    JOURNAL_PATH.unlink(missing_ok=True)
