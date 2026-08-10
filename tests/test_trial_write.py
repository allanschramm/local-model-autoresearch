"""Trial write path serialization tests (issue #3).

Objective Vector axes (ctx, TPS, agentic, coding — partials allowed) and
Trial Status (on_front | dominated | incomplete | rejected) persist through
write_row into results.tsv. Legacy keep/discard are rejected on write.
"""

from __future__ import annotations

import csv

import pytest

from autoresearch.runners.run import (
    TRIAL_STATUSES,
    get_previous_best,
    write_row,
)


def _read(tmp_tsv) -> list[dict]:
    with open(tmp_tsv, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _write_row(tmp_tsv, status: str = "on_front", val_score: float = 0.5, **kw) -> None:
    write_row(
        tmp_tsv,
        "abc123",
        val_score,
        0.4,
        0.3,
        0.2,
        1.0,
        status,
        "trial write test",
        **kw,
    )


def test_write_row_persists_objective_vector_axes(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(
        tsv,
        ctx=131072,
        tps=30.5,
        agentic=0.6000,
        coding=0.650000,
    )
    row = _read(tsv)[0]
    assert row["ctx"] == "131072"
    assert row["tps"] == "30.5"
    assert row["agentic"] == "0.6000"
    assert row["coding"] == "0.650000"


def test_write_row_preserves_engine_version_tag(tmp_path, monkeypatch):
    """Trial evidence carries engine@tag when the server comes from a fork release."""
    from pathlib import Path

    from autoresearch.runners import run

    tsv = tmp_path / "results.tsv"
    fork = Path("llama.cpp-releases/turboquant/tqp-v0.3.0/build-cuda/bin/llama-server.exe")
    monkeypatch.setattr(run, "resolve_llama_server", lambda: fork)
    _write_row(tsv, ctx=100000, tps=158.5)
    row = _read(tsv)[0]
    assert row["binary_version"] == "turboquant@tqp-v0.3.0"


def test_partial_axes_render_blank(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, ctx=32768, tps=20.0)  # agentic + coding unmeasured
    row = _read(tsv)[0]
    assert row["ctx"] == "32768"
    assert row["tps"] == "20.0"
    assert row["agentic"] == ""
    assert row["coding"] == ""


def test_on_front_persists_literally(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front")
    assert _read(tsv)[0]["status"] == "on_front"


def test_new_trial_statuses_persist_literally(tmp_path):
    tsv = tmp_path / "results.tsv"
    for status in ("dominated", "incomplete", "rejected"):
        _write_row(tsv, status=status)
    persisted = [row["status"] for row in _read(tsv)]
    assert persisted == ["dominated", "incomplete", "rejected"]


def test_legacy_keep_discard_rejected_on_write(tmp_path):
    tsv = tmp_path / "results.tsv"
    for status in ("keep", "discard"):
        with pytest.raises(ValueError, match=status):
            _write_row(tsv, status=status)
    assert not tsv.exists()


def test_invalid_status_rejected(tmp_path):
    tsv = tmp_path / "results.tsv"
    try:
        _write_row(tsv, status="bogus")
    except ValueError as e:
        assert "bogus" in str(e)
    else:
        raise AssertionError("expected ValueError for invalid status")
    assert not tsv.exists()  # nothing written


def test_status_enum_matches_issue_vocabulary():
    assert {
        "on_front",
        "dominated",
        "incomplete",
        "rejected",
    } == TRIAL_STATUSES


def test_previous_best_sees_on_front_rows(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_row(tsv, status="on_front", val_score=0.42)
    _write_row(tsv, status="dominated", val_score=0.10)
    assert get_previous_best(tsv) == 0.42


def test_previous_best_ignores_legacy_keep_cells(tmp_path):
    """Stale keep cells (no longer recognized) do not count as frontier."""
    tsv = tmp_path / "results.tsv"
    rows = [
        {"trial_id": "a", "model": "m.gguf", "status": "keep", "val_score": "0.30"},
        {"trial_id": "b", "model": "m.gguf", "status": "on_front", "val_score": "0.55"},
        {"trial_id": "c", "model": "m.gguf", "status": "discard", "val_score": "0.90"},
    ]
    with open(tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    assert get_previous_best(tsv) == 0.55
