"""Store-wide Pareto status recompute (issue #5).

Pure decision logic — no file I/O. Takes results.tsv rows and returns a new
list with every row's status refreshed so the Pareto Set stays consistent:
a new on_front point demotes rows it dominates to dominated. Two scopes per
ADR 0006: `bucket` (default) is the canonical stored status — the
    global-by-hardware+budget front, every complete vector in a
    configured-VRAM_LIMIT bucket (fallback: round(memory_gb)) competes across models; `model` is the per-model
lens (Search/Neighbors stay per model) — rows compete only against
same-model complete vectors in the same bucket. The per-model lens is a
view; only the bucket scope is persisted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoresearch.core.classify import (
    fp_from_config_json,
    persist_status,
    row_bucket,
    vector_from_row,
)
from autoresearch.core.pareto import ObjectiveVector, Trial, dominates, merge

SCOPES = ("bucket", "model")


def recompute_rows(
    rows: Sequence[Mapping[str, Any]], *, scope: str = "bucket"
) -> list[dict[str, Any]]:
    """New row list with refreshed statuses (idempotent, pure).

    Rows without a fingerprint (no config_json) or a memory cell are left
    untouched — legacy keep/discard rows are not recomputed. rejected rows
    never compete. Every row of one group (bucket × fingerprint, or
    model × bucket × fingerprint in model scope) shares the merged vector's
    status: incomplete stays incomplete; a complete vector dominated by
    another complete merged vector in the same domination scope becomes
    dominated; the rest stay on_front (persisted as the keep alias).
    """
    if scope not in SCOPES:
        raise ValueError(f"invalid scope: {scope!r}; allowed: {sorted(SCOPES)}")
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        fp = fp_from_config_json(row.get("config_json"))
        bucket_gb = row_bucket(row)
        if fp is None or bucket_gb is None or row.get("status") == "rejected":
            continue
        key = (row.get("model") or "", bucket_gb, fp) if scope == "model" else (bucket_gb, fp)
        groups.setdefault(key, []).append(idx)
    merged_by_group: dict[tuple[Any, ...], ObjectiveVector] = {}
    for key, idxs in groups.items():
        vectors = [vector_from_row(rows[i]) for i in idxs]
        merged_by_group[key] = merge([Trial(fp=key[-1], vector=v) for v in vectors])[0].vector
    # Two passes: every group competes against the full front of its
    # domination scope (bucket, or model × bucket), so a later group can
    # demote an earlier one.
    complete_by_scope: dict[tuple[Any, ...], list[ObjectiveVector]] = {}
    for key, merged in merged_by_group.items():
        if merged.complete:
            complete_by_scope.setdefault(key[:-1], []).append(merged)
    statuses: dict[tuple[Any, ...], str] = {}
    for key, merged in merged_by_group.items():
        if not merged.complete:
            statuses[key] = "incomplete"
        elif any(dominates(other, merged) for other in complete_by_scope.get(key[:-1], ())):
            statuses[key] = "dominated"
        else:
            statuses[key] = "on_front"
    out = [dict(row) for row in rows]
    for key, idxs in groups.items():
        status = persist_status(statuses[key])
        for idx in idxs:
            out[idx]["status"] = status
    return out
