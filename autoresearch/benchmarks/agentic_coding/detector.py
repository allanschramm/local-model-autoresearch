"""Loop / stall / hallucination detector for the SWE-lite coding-agent loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ALLOWED_TOOLS = frozenset(
    {"read_file", "write_file", "str_replace", "grep", "list_dir", "run_tests"}
)
PATH_TOOLS = frozenset({"read_file", "write_file", "str_replace", "grep", "list_dir"})
MUTATING_TOOLS = frozenset({"write_file", "str_replace"})
REPEAT_LIMIT = 3
STALL_TURNS = 3
HIDDEN_DIR_NAME = "_hidden_tests"


def canonical_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"tool": name, "arguments": arguments}, sort_keys=True, separators=(",", ":"))


def resolve_in_worktree(worktree: Path, rel: str) -> Path | None:
    """Return resolved path if it stays inside worktree; else None (escape)."""
    rel = (rel or "").strip()
    if not rel or rel.startswith("/") or (len(rel) >= 2 and rel[1] == ":"):
        return None
    raw = Path(rel)
    if raw.is_absolute():
        return None
    root = worktree.resolve()
    candidate = (worktree / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def path_on_allowlist(
    worktree: Path,
    resolved: Path,
    allowlisted: tuple[str, ...],
    *,
    mutate: bool,
) -> bool:
    """True if `resolved` is an allowlisted file (mutate) or inspectable prefix."""
    if not allowlisted:
        return True
    root = worktree.resolve()
    try:
        rel = resolved.resolve().relative_to(root).as_posix()
    except ValueError:
        return False
    allowed = {a.replace("\\", "/").strip("/") for a in allowlisted if a}
    if mutate:
        return rel in allowed
    if resolved.resolve() == root:
        return True
    for entry in allowed:
        if rel == entry:
            return True
        if entry.startswith(rel + "/") or rel.startswith(entry + "/"):
            return True
    return False


def workspace_hash(worktree: Path, allowlisted: tuple[str, ...]) -> str:
    """SHA256 of allowlisted files (missing files count as empty)."""
    h = hashlib.sha256()
    root = worktree.resolve()
    for rel in sorted(allowlisted):
        path = resolve_in_worktree(worktree, rel)
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        if path is None or not path.is_file():
            h.update(b"missing")
            continue
        h.update(path.read_bytes())
        h.update(b"\n")
    h.update(str(root).encode("utf-8"))
    return h.hexdigest()


@dataclass
class DetectorState:
    worktree: Path
    allowlisted: tuple[str, ...]
    docs_only: bool = False
    calls: list[str] = field(default_factory=list)
    last_hash: str | None = None
    stall_turns: int = 0
    fail_reason: str | None = None

    def snapshot_hash(self) -> str:
        return workspace_hash(self.worktree, self.allowlisted)

    def note_turn_hash(self, *, mutated: bool = False) -> str | None:
        """Stall only after consecutive mutating turns with no allowlisted hash change."""
        current = self.snapshot_hash()
        if not mutated:
            self.last_hash = current
            return self.fail_reason
        if self.last_hash is not None and current == self.last_hash:
            self.stall_turns += 1
            if self.stall_turns >= STALL_TURNS:
                self.fail_reason = f"stall:{STALL_TURNS}_turns_no_file_change"
        else:
            self.stall_turns = 0
        self.last_hash = current
        return self.fail_reason

    def record_tool(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Return fail reason or None. First failure sticks."""
        if self.fail_reason:
            return self.fail_reason
        if name not in ALLOWED_TOOLS:
            self.fail_reason = f"hallucination:unknown_tool:{name}"
            return self.fail_reason
        if self.docs_only and name == "run_tests":
            self.fail_reason = "restraint:run_tests_on_docs_only"
            return self.fail_reason
        if name in PATH_TOOLS:
            rel = str(arguments.get("path") or "")
            if name == "grep" and not rel:
                rel = "."
            if name == "list_dir" and not rel:
                rel = "."
            resolved = resolve_in_worktree(self.worktree, rel)
            if resolved is None:
                self.fail_reason = f"hallucination:path_escape:{rel}"
                return self.fail_reason
            try:
                rel_to_hidden = resolved.relative_to((self.worktree / HIDDEN_DIR_NAME).resolve())
            except ValueError:
                rel_to_hidden = None
            if rel_to_hidden is not None:
                self.fail_reason = "hallucination:hidden_tests"
                return self.fail_reason
            if not path_on_allowlist(
                self.worktree,
                resolved,
                self.allowlisted,
                mutate=name in MUTATING_TOOLS,
            ):
                self.fail_reason = f"hallucination:path_not_allowlisted:{rel}"
                return self.fail_reason
        key = canonical_call(name, arguments)
        self.calls.append(key)
        if self.calls.count(key) >= REPEAT_LIMIT:
            self.fail_reason = f"loop:repeat_{REPEAT_LIMIT}:{name}"
            return self.fail_reason
        return None
