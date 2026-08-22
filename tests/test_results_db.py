"""Tests for the derived SQLite mirror of results.tsv (results_db)."""

from __future__ import annotations

import csv
import sqlite3

from autoresearch.core import results_db

NUMERIC_COLS = [
    "val_score",
    "tps",
    "ctx",
    "agentic",
    "coding",
    "memory_gb",
]


def _conn(tmp_path):
    conn = sqlite3.connect(tmp_path / "results.db")
    results_db.ensure_schema(conn)
    return conn


def test_schema_creates_trials_table(tmp_path):
    conn = _conn(tmp_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "trials" in names


def test_numeric_columns_are_real(tmp_path):
    conn = _conn(tmp_path)
    cols = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(trials)")}
    for c in NUMERIC_COLS:
        assert cols[c].upper() == "REAL", c


def test_trial_id_is_primary_key(tmp_path):
    conn = _conn(tmp_path)
    pk = [r[1] for r in conn.execute("PRAGMA table_info(trials)") if r[5]]
    assert pk == ["trial_id"]


def test_indexes_exist(tmp_path):
    conn = _conn(tmp_path)
    idx = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert any("model" in i for i in idx)
    assert any("status" in i for i in idx)


def test_to_cell_coercion():
    assert results_db._to_cell("tps", "27.8") == 27.8
    assert results_db._to_cell("tps", "") is None
    assert results_db._to_cell("tps", None) is None
    assert results_db._to_cell("model", "Ornith-35B") == "Ornith-35B"
    assert results_db._to_cell("status", "on_front") == "on_front"


def _row(**overrides):
    base = {
        c: ""
        for c in (
            "schema_version",
            "trial_id",
            "commit",
            "model",
            "backend",
            "status",
            "val_score",
            "tps",
            "ctx",
            "agentic",
            "coding",
            "description",
        )
    }
    base.update(
        trial_id="t-0001",
        model="Ornith-35B",
        status="on_front",
        val_score="0.57",
        tps="27.8",
        ctx="32768",
    )
    base.update(overrides)
    return base


def _write_tsv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def test_replace_all_inserts_and_is_idempotent(tmp_path):
    conn = _conn(tmp_path)
    rows = [_row(), _row(trial_id="t-0002", model="Mythos", status="dominated")]
    results_db.replace_all(conn, rows)
    results_db.replace_all(conn, rows)  # rerun changes nothing, no dupes
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 2


def test_upsert_row_updates_existing_trial(tmp_path):
    conn = _conn(updates := tmp_path) or None
    results_db.replace_all(conn, [_row()])
    results_db.upsert_row(conn, _row(status="dominated"))
    got = conn.execute("SELECT status FROM trials WHERE trial_id='t-0001'").fetchone()[0]
    assert got == "dominated"


def test_numeric_values_stored_typed(tmp_path):
    conn = _conn(tmp_path)
    results_db.replace_all(conn, [_row()])
    tps, ctx, model = conn.execute(
        "SELECT tps, ctx, model FROM trials WHERE trial_id='t-0001'"
    ).fetchone()
    assert tps == 27.8 and isinstance(tps, float)
    assert ctx == 32768.0
    assert model == "Ornith-35B"


def test_sync_from_tsv_populates_mirror(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row(), _row(trial_id="t-0002")])
    n = results_db.sync_from_tsv(tsv, tmp_path / "results.db")
    assert n == 2
    conn = sqlite3.connect(tmp_path / "results.db")
    assert conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 2


def test_try_sync_swallows_missing_tsv(tmp_path):
    # Missing TSV must not raise — mirror is best-effort.
    n = results_db.try_sync_from_tsv(tmp_path / "nope.tsv", tmp_path / "r.db")
    assert n == 0


def test_try_sync_reports_failure_without_raising(tmp_path, monkeypatch):
    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row()])
    monkeypatch.setattr(
        results_db,
        "replace_all",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("boom")),
    )
    n = results_db.try_sync_from_tsv(tsv, tmp_path / "results.db")
    assert n == 0  # failed, but did not raise


def test_parity_check_detects_drift(tmp_path):
    tsv = tmp_path / "results.tsv"
    _write_tsv(tsv, [_row(), _row(trial_id="t-0002")])
    db = tmp_path / "results.db"
    results_db.sync_from_tsv(tsv, db)
    ok, report = results_db.parity_check(tsv, db)
    assert ok

    # Drift: delete a row behind the mirror's back.
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM trials WHERE trial_id='t-0002'")
    conn.commit()
    conn.close()

    ok, report = results_db.parity_check(tsv, db)
    assert not ok
    assert "t-0002" in report
