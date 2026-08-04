"""Trial classifier over the Pareto nucleus (issue #4).

Pure decision logic — no file I/O. The run.py write path feeds results.tsv
rows as dicts and persists the outcome. Vocabulary follows CONTEXT.md /
ADR 0006: `rejected` | `incomplete` | `on_front` | `dominated`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from autoresearch.core.config import DEFAULTS
from autoresearch.core.pareto import ObjectiveVector, Trial, dominates, fingerprint, merge

# Fingerprint identity = ENGINE_DEFAULTS + SAMPLER_DEFAULTS only (ADR 0006).
# Extra keys (e.g. bench INCLUDE_* flags in AutoLoop configs) must not split
# two write paths that share the same engine+sampler Baseline.
_IDENTITY_KEYS = frozenset(str(k).lower() for k in DEFAULTS)

# Hardware+budget identity of the known Set (ADR 0006: the global front is
# ranked per hardware+budget). The TSV has no hardware column; peak VRAM
# rounded is the practical budget proxy.
BUCKET_PROXY = "memory_gb"


def bucket(memory_gb: float) -> int:
    return round(memory_gb)


def _cell_float(cell: Any) -> float | None:
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def vector_from_row(row: Mapping[str, Any]) -> ObjectiveVector:
    """Objective Vector of a results.tsv row; blank axis = not measured."""
    return ObjectiveVector(
        ctx=_cell_float(row.get("ctx")),
        tps=_cell_float(row.get("tps")),
        agentic=_cell_float(row.get("agentic")),
        coding=_cell_float(row.get("coding")),
    )


def fp_from_baseline(baseline: Mapping[str, Any]) -> str:
    """Fingerprint of the full Baseline = ENGINE_DEFAULTS + SAMPLER_DEFAULTS.

    Keys are lowercased so the live config dict and the persisted config_json
    cell hash identically (ADR 0006: engine + sampler, not model-only).
    """
    canonical = {str(k).lower(): v for k, v in baseline.items() if str(k).lower() in _IDENTITY_KEYS}
    return fingerprint({"baseline": canonical}, {})


def fp_from_config_json(config_json: Any) -> str | None:
    """Recompute a row's Fingerprint from its persisted config_json cell."""
    if not config_json:
        return None
    try:
        loaded = json.loads(config_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(loaded, dict):
        return None
    return fp_from_baseline(loaded)


def persist_status(status: str) -> str:
    """on_front persists as the legacy `keep` alias (issue #3 reader compat)."""
    return "keep" if status == "on_front" else status


def is_on_front(status: Any) -> bool:
    """True for `on_front` and the legacy `keep` alias (issue #12 reader compat)."""
    return status in ("on_front", "keep")


def classify_trial(
    *, failed: bool, vector: ObjectiveVector, known: Iterable[ObjectiveVector]
) -> str:
    """ADR 0006 status for one Trial: rejected | incomplete | on_front | dominated."""
    if failed:
        return "rejected"
    if not vector.complete:
        return "incomplete"
    if any(dominates(other, vector) for other in known):
        return "dominated"
    return "on_front"


def _known_vectors(rows: Sequence[Mapping[str, Any]], bucket_gb: int) -> list[ObjectiveVector]:
    """Complete points already in this hardware+budget bucket, merged per Fingerprint."""
    by_fp: dict[str, list[ObjectiveVector]] = {}
    for row in rows:
        if row.get("status") == "rejected":
            continue  # rejected Trials never compete for the front
        if row_bucket(row) != bucket_gb:
            continue
        fp = fp_from_config_json(row.get("config_json"))
        if fp is None:
            continue
        by_fp.setdefault(fp, []).append(vector_from_row(row))
    known = []
    for fp, vectors in by_fp.items():
        merged = merge([Trial(fp=fp, vector=v) for v in vectors])[0].vector
        if merged.complete:
            known.append(merged)
    return known


def row_bucket(row: Mapping[str, Any]) -> int | None:
    """Bucket of a results.tsv row (None when the memory cell is unset)."""
    mem = _cell_float(row.get(BUCKET_PROXY))
    return None if mem is None else bucket(mem)


def plan_write(
    rows: Sequence[Mapping[str, Any]],
    *,
    fp: str,
    vector: ObjectiveVector,
    bucket_gb: int,
    failed: bool = False,
) -> tuple[str, dict[str, str]]:
    """Status for a new Trial plus prior rows to flip.

    Returns (status, {trial_id: status}). When the Fingerprint merge completes
    an earlier incomplete point, every prior incomplete row of the same
    Fingerprint **in the same bucket** is flipped to the computed status
    (ADR 0006 merge rule; merge and the known Set never cross budgets).
    """
    if failed:
        return "rejected", {}
    prior = [
        vector_from_row(row)
        for row in rows
        if row.get("status") != "rejected"
        and fp_from_config_json(row.get("config_json")) == fp
        and row_bucket(row) == bucket_gb
    ]
    merged = merge([Trial(fp=fp, vector=v) for v in [*prior, vector]])[0].vector
    status = classify_trial(failed=False, vector=merged, known=_known_vectors(rows, bucket_gb))
    if status == "incomplete":
        return status, {}
    flips = {
        row.get("trial_id", ""): status
        for row in rows
        if row.get("status") == "incomplete"
        and fp_from_config_json(row.get("config_json")) == fp
        and row_bucket(row) == bucket_gb
    }
    return status, flips
