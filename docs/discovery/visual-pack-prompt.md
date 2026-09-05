# Visual Pack Prompt 01 — Notes CLI Fix (FROZEN)

**Status: frozen 2026-09-05 (issue #54). Do not edit.** Paste verbatim into Pi.
New tasks get a new numbered file; this one never changes so runs stay comparable.

You are driving a local checkout. Work in an empty temporary directory
(`notes-fix-01/` under the OS temp dir — no repo paths, no existing files).

## Setup

Create exactly this file as `notes.py` in the empty directory:

```python
"""Tiny notes CLI (visual-pack fixture 01)."""
import json
import sys
from pathlib import Path

STORE = Path("notes.jsonl")


def cmd_add(text: str) -> None:
    with STORE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": text}) + "\n")
    print(f"added: {text}")


def cmd_list() -> None:
    if not STORE.exists():
        print("(empty)")
        return
    for line in STORE.read_text(encoding="utf-8").splitlines():
        print("- " + json.loads(line)["note"])


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"add", "list"}:
        sys.exit("usage: python notes.py add <text> | python notes.py list")
    if sys.argv[1] == "add":
        cmd_add(" ".join(sys.argv[2:]))
    else:
        cmd_list()
```

## Task

1. Run `python notes.py add "hello camera"` — it should succeed.
2. Run `python notes.py list` — reproduce the failure and read the traceback.
3. Fix the bug with the smallest change that keeps `add` output format stable
   (the stored JSON shape is the contract; fix the reader side).
4. Re-run both commands: `add` a second note, then `list` — both exit 0.
5. Show the workspace state at the end: file listing plus the full `list` output,
   proving notes persist across runs.

## Done when

- Both commands exit 0 and `list` prints every added note.
- You narrate the root cause in one sentence before the fix, and confirm the
  persisted store after it. No scores, no rankings — the camera is the judge.
