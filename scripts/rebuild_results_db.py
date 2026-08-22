#!/usr/bin/env python
"""Force-rebuild the derived results.db mirror from results.tsv + parity check.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\rebuild_results_db.py [path\\to\\results.tsv]

Exit 0 when the rebuilt mirror matches the TSV, 1 otherwise. Safe to run while
a Search is idle; if the autoloop is mid-write, rerun afterwards.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from autoresearch.core import results_db  # noqa: E402

DEFAULT_TSV = REPO_ROOT / "results.tsv"


def main() -> int:
    tsv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TSV
    db = results_db.default_db_path(tsv)
    n = results_db.sync_from_tsv(tsv, db)
    print(f"[results-db] rebuilt {db} from {tsv}: {n} rows")
    ok, report = results_db.parity_check(tsv, db)
    print(f"[results-db] {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
