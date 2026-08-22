"""Store-wide Pareto status recompute (issue #5).

Pure decision logic — no file I/O. Takes results.tsv rows and returns a new
list with every row's status refreshed so the Pareto Set stays consistent:
a new on_front point demotes rows it dominates to dominated. Two scopes per
ADR 0006 / 0012: `bucket` (default) is the canonical stored status — the
global-by-hardware+budget front, every complete basename vector in a
configured-VRAM_LIMIT bucket (fallback: round(memory_gb)) competes across
models; `model` is the per-model lens (Search/Neighbors stay per model) —
rows compete only against same-model complete vectors in the same bucket.
The per-model lens is a view; only the bucket scope is persisted.
Point identity = GGUF basename (ADR 0012).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from autoresearch.core.classify import (
    MORRIS_SCREEN_PROFILE,
    row_bucket,
    vector_from_row,
)
from autoresearch.core.pareto import ObjectiveVector, Trial, dominates, merge

SCOPES = ("bucket", "model")


def recompute_rows(
    rows: Sequence[Mapping[str, Any]], *, scope: str = "bucket"
) -> list[dict[str, Any]]:
    """New row list with refreshed statuses (idempotent, pure).

    Point identity is the GGUF basename (ADR 0012). Rows without a model name
    or a budget bucket are left untouched. rejected rows never compete. Every
    row of one group (bucket × model, or model × bucket in model scope) shares
    the merged vector's status: incomplete stays incomplete; a complete vector
    dominated by another complete merged vector in the same domination scope
    becomes dominated; the rest stay on_front.
    """
    if scope not in SCOPES:
        raise ValueError(f"invalid scope: {scope!r}; allowed: {sorted(SCOPES)}")
    groups: dict[tuple[Any, ...], list[int]] = {}
    for idx, row in enumerate(rows):
        model = (row.get("model") or "").strip()
        bucket_gb = row_bucket(row)
        if (
            not model
            or bucket_gb is None
            or row.get("status") == "rejected"
            or (row.get("evaluation_profile") or "").strip() == MORRIS_SCREEN_PROFILE
        ):
            continue
        # bucket scope: (bucket, model); model scope: (model, bucket) — one point
        # per basename per budget; model scope never demotes across basenames.
        key = (model, bucket_gb) if scope == "model" else (bucket_gb, model)
        groups.setdefault(key, []).append(idx)
    merged_by_group: dict[tuple[Any, ...], ObjectiveVector] = {}
    for key, idxs in groups.items():
        vectors = [vector_from_row(rows[i]) for i in idxs]
        # merge() keys on Trial.fp — use basename as the merge id.
        merge_id = key[1] if scope == "bucket" else key[0]
        merged_by_group[key] = merge([Trial(fp=str(merge_id), vector=v) for v in vectors])[0].vector
    # Two passes: every group competes against the full front of its
    # domination scope (bucket, or model × bucket), so a later group can
    # demote an earlier one.
    complete_by_scope: dict[tuple[Any, ...], list[ObjectiveVector]] = {}
    for key, merged in merged_by_group.items():
        if merged.complete:
            # model scope: isolate per (model, bucket); bucket scope: all models
            # in the bucket compete (key = (bucket, model) → scope (bucket,)).
            scope_key = key if scope == "model" else key[:-1]
            complete_by_scope.setdefault(scope_key, []).append(merged)
    statuses: dict[tuple[Any, ...], str] = {}
    for key, merged in merged_by_group.items():
        scope_key = key if scope == "model" else key[:-1]
        if not merged.complete:
            statuses[key] = "incomplete"
        elif any(dominates(other, merged) for other in complete_by_scope.get(scope_key, ())):
            statuses[key] = "dominated"
        else:
            statuses[key] = "on_front"
    out = [dict(row) for row in rows]
    for key, idxs in groups.items():
        status = statuses[key]
        for idx in idxs:
            out[idx]["status"] = status
    return out
