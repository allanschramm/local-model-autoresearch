#!/usr/bin/env python3
"""Recompute Pareto Set statuses in a results store (issue #5).

Reads a results.tsv, refreshes every row's Trial status so the Pareto Set
stays consistent: a new on_front point demotes rows it dominates to
dominated; incomplete and rejected rows are left out. Stored status is the
global-by-hardware+budget view (ADR 0006). Idempotent: running twice changes
nothing. No GPU required.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\recompute_status.py
    .\\venv\\Scripts\\python.exe scripts\\recompute_status.py path\\to\\results.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.runners import run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh Trial statuses in a results.tsv (Pareto Set recompute)."
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        default=str(REPO_ROOT / "results.tsv"),
        help="path to results.tsv",
    )
    args = parser.parse_args()
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"results store not found: {results_file}", file=sys.stderr)
        return 1
    run.recompute_statuses(results_file)
    print(f"statuses refreshed: {results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
