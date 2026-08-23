#!/usr/bin/env python
"""Results-store maintenance: seed/repair the canonical DB and legacy TSV.

``results.db`` (SQLite) is the canonical Trial store; ``results.tsv`` is the
legacy append-log kept in sync as a fallback. This tool moves data between
them and verifies parity.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\rebuild_results_db.py [path\\to\\results.tsv]
    .\\venv\\Scripts\\python.exe scripts\\rebuild_results_db.py --rebuild-tsv
    .\\venv\\Scripts\\python.exe scripts\\rebuild_results_db.py --force

Modes:
    (default)     seed the canonical DB from the legacy TSV when the DB is
                  missing or empty, then parity-check both stores
    --force       rebuild the DB from the TSV even when the DB looks healthy
    --rebuild-tsv rewrite the legacy TSV from the canonical DB

Exit 0 on parity, 1 on drift. Safe to run while a Search is idle; if the
autoloop is mid-write, rerun afterwards.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autoresearch.core import results_db  # noqa: E402

DEFAULT_TSV = REPO_ROOT / "results.tsv"


def main() -> int:
    args = [a for a in sys.argv[1:] if a in ("--rebuild-tsv", "--force")]
    positional = [a for a in sys.argv[1:] if a not in ("--rebuild-tsv", "--force")]
    if len(args) > 1:
        print("[results-db] --rebuild-tsv and --force are mutually exclusive")
        return 1
    tsv = Path(positional[0]) if positional else DEFAULT_TSV
    db = results_db.default_db_path(tsv)

    if args == ["--rebuild-tsv"]:
        try:
            n = results_db.sync_to_tsv(tsv, db)
        except FileNotFoundError as exc:
            print(f"[results-db] {exc}")
            return 1
        print(f"[results-db] rewrote legacy TSV {tsv} from canonical DB: {n} rows")
    else:
        db_rows = results_db.read_rows(db)
        if args == ["--force"] or db_rows is None or not db_rows:
            n = results_db.sync_from_tsv(tsv, db)
            print(f"[results-db] seeded canonical DB {db} from {tsv}: {n} rows")
        else:
            print(
                f"[results-db] canonical DB already populated ({len(db_rows)} rows); use --force to rebuild"
            )

    ok, report = results_db.parity_check(tsv, db)
    print(f"[results-db] {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
