"""Derived SQLite mirror of results.tsv (TSV stays canonical).

Read-mostly convenience layer: scripts and dashboards can query indexed
columns without scanning the multi-MB append-log. The mirror is rebuilt from
the TSV, never hand-edited; a stale or corrupt mirror is always fixable with
scripts/rebuild_results_db.py.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

# Columns stored as REAL (blank -> NULL). Everything else TEXT.
_NUMERIC_COLUMNS = frozenset(
    {
        "val_score",
        "swe_score",
        "lcb_score",
        "he_score",
        "mbpp_score",
        "bigcode_score",
        "agentic",
        "coding",
        "agentic_coding",
        "memory_gb",
        "elapsed_sec",
        "tps",
        "bench_tg",
        "ctx",
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "n_cpu_moe",
        "temp",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "gpu_temp_c",
        "tps_spread",
    }
)

_COLUMNS: list[str] = [
    "schema_version",
    "trial_id",
    "commit",
    "model",
    "model_id",
    "backend",
    "category",
    "evaluation_profile",
    "scoring_benchmark",
    "outcome",
    "diagnostic",
    "status",
    "val_score",
    "swe_score",
    "lcb_score",
    "he_score",
    "mbpp_score",
    "bigcode_score",
    "agentic",
    "coding",
    "agentic_coding",
    "memory_gb",
    "elapsed_sec",
    "tps",
    "bench_tg",
    "kv",
    "ctx",
    "threads",
    "threads_batch",
    "batch_size",
    "ubatch_size",
    "n_cpu_moe",
    "temp",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "cont_batching",
    "flash_attn",
    "no_mmap",
    "spec_draft_n_max",
    "task_ids",
    "random_seed",
    "config_json",
    "binary_version",
    "tps_source",
    "gpu_temp_c",
    "tps_reps",
    "tps_spread",
    "description",
]


def _to_cell(column: str, raw: str | None) -> float | str | None:
    """Blank -> NULL; numeric columns coerced to float; others verbatim."""
    if raw is None or raw == "":
        return None
    if column in _NUMERIC_COLUMNS:
        try:
            return float(raw)
        except ValueError:
            return None
    return raw


def default_db_path(results_file: Path) -> Path:
    """Mirror lives next to its TSV: results.tsv -> results.db."""
    return Path(results_file).with_name("results.db")


def ensure_schema(conn: sqlite3.Connection) -> None:
    cols = ",\n  ".join(
        f"{_q(c)} {'REAL' if c in _NUMERIC_COLUMNS else 'TEXT'}"
        + (" PRIMARY KEY" if c == "trial_id" else "")
        for c in _COLUMNS
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS trials (\n  {cols}\n)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_model ON trials(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trials_model_status ON trials(model, status)")
    conn.commit()


def _q(name: str) -> str:
    """Quote an identifier — `commit` is a SQLite reserved word."""
    return f'"{name}"'


def _insert_sql() -> str:
    placeholders = ", ".join("?" for _ in _COLUMNS)
    cols = ", ".join(_q(c) for c in _COLUMNS)
    return f"INSERT OR REPLACE INTO trials ({cols}) VALUES ({placeholders})"


def _cells(row: dict) -> list:
    return [_to_cell(c, row.get(c)) for c in _COLUMNS]


def replace_all(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Transactional wipe + bulk insert from TSV-shaped dict rows."""
    with conn:  # implicit BEGIN/COMMIT; rolls back on error
        conn.execute("DELETE FROM trials")
        conn.executemany(_insert_sql(), (_cells(r) for r in rows))
    return len(rows)


def upsert_row(conn: sqlite3.Connection, row: dict) -> None:
    with conn:
        conn.execute(_insert_sql(), _cells(row))


def _read_tsv(results_file: Path) -> list[dict]:
    """Minimal local reader — avoids importing runners (keeps core leaf-pure)."""
    if not results_file.exists() or results_file.stat().st_size == 0:
        return []
    with open(results_file, encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f, delimiter="\t")]


def sync_from_tsv(results_file: Path, db_path: Path | None = None) -> int:
    """Full rebuild of the mirror from the canonical TSV. Returns row count."""
    db_path = db_path or default_db_path(Path(results_file))
    rows = _read_tsv(Path(results_file))
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        return replace_all(conn, rows)
    finally:
        conn.close()


def try_sync_from_tsv(results_file: Path, db_path: Path | None = None) -> int:
    """Best-effort sync for the hot write path: never raises, logs on failure.

    The Trial result is already durable in the TSV when this runs; a mirror
    failure must not affect the Trial outcome.
    """
    try:
        return sync_from_tsv(results_file, db_path)
    except Exception as exc:  # mirror must never break a Trial write
        print(f"[results-db] mirror sync failed (TSV unaffected): {exc}")
        return 0


def parity_check(results_file: Path, db_path: Path | None = None) -> tuple[bool, str]:
    """Compare mirror vs TSV: row count + trial_id set (+ duplicate detection).

    A missing or un-migrated mirror (no `trials` table) is drift, not a
    crash: reports (False, reason) so callers can rebuild.
    """
    db_path = db_path or default_db_path(Path(results_file))
    rows = _read_tsv(Path(results_file))
    if not db_path.exists():
        return False, f"mirror missing: {db_path}"
    conn = sqlite3.connect(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "trials" not in tables:
            return False, "mirror has no 'trials' table (un-migrated or corrupt)"
        db_ids = {r[0] for r in conn.execute("SELECT trial_id FROM trials")}
    finally:
        conn.close()
    tsv_ids = {r.get("trial_id", "") for r in rows}
    problems: list[str] = []
    if len(rows) != len(tsv_ids):
        problems.append(f"duplicate trial_id in TSV ({len(rows)} rows, {len(tsv_ids)} ids)")
    missing = sorted(tsv_ids - db_ids)
    extra = sorted(db_ids - tsv_ids)
    if missing:
        problems.append(f"in TSV, not in mirror: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if extra:
        problems.append(f"in mirror, not in TSV: {extra[:5]}{'…' if len(extra) > 5 else ''}")
    if problems:
        return False, "; ".join(problems)
    return True, f"parity OK: {len(rows)} rows"
