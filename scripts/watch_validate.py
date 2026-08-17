"""Wait for GitHub Actions `validate.yml` on HEAD (or --sha).

Windows pytest never imports `fcntl` and does not execute POSIX lock/path
branches. CI is `ubuntu-latest` (`.github/workflows/validate.yml`). After every
push, run this from repo root until it exits 0:

  ./venv/Scripts/python.exe scripts/watch_validate.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = "validate.yml"


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _list_runs(sha: str) -> list[dict]:
    proc = subprocess.run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            WORKFLOW,
            "--commit",
            sha,
            "--limit",
            "5",
            "--json",
            "databaseId,status,conclusion,url,headSha",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise SystemExit(f"gh run list failed: {err}")
    return json.loads(proc.stdout or "[]")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch GitHub Actions validate.yml for a commit until it finishes."
    )
    parser.add_argument("--sha", help="Commit SHA (default: HEAD)")
    args = parser.parse_args()

    if shutil.which("gh") is None:
        print(
            "gh CLI not found. Install GitHub CLI (https://cli.github.com) and run: gh auth login",
            file=sys.stderr,
        )
        return 2

    sha = (args.sha or _git_head()).strip()
    print(f"Waiting for {WORKFLOW} on {sha}…", flush=True)
    run: dict | None = None
    while run is None:
        rows = _list_runs(sha)
        if rows:
            run = rows[0]
            break
        print("No validate run yet (queued). Retrying…", flush=True)
        time.sleep(3)

    run_id = str(run["databaseId"])
    url = run.get("url") or ""
    print(f"Watching run {run_id} {url}", flush=True)
    return subprocess.call(
        ["gh", "run", "watch", run_id, "--exit-status"],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
