"""Store-wide Pareto status recompute (issue #5).

Pure decision logic — no file I/O. Takes results.tsv rows and returns a new
list with every row's status refreshed so the Pareto Set stays consistent:
a new on_front point demotes rows it dominates to dominated. Stored status
is the global-by-hardware+budget view (ADR 0006): status derives from the
(round(memory_gb) bucket, fingerprint) merged vector. The per-model lens
stays a derived rank_results view — never stored here.
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


def recompute_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """New row list with refreshed statuses (idempotent, pure).

    Rows without a fingerprint (no config_json) or a memory cell are left
    untouched — legacy keep/discard rows are not recomputed. rejected rows
    never compete. Every row of one (bucket, fingerprint) group shares the
    merged vector's status: incomplete stays incomplete; a complete vector
    dominated by another complete merged vector in the same bucket becomes
    dominated; the rest stay on_front (persisted as the keep alias).
    """
    groups: dict[tuple[int, str], list[int]] = {}
    merged_by_group: dict[tuple[int, str], ObjectiveVector] = {}
    statuses: dict[tuple[int, str], str] = {}
    for idx, row in enumerate(rows):
        fp = fp_from_config_json(row.get("config_json"))
        bucket_gb = row_bucket(row)
        if fp is None or bucket_gb is None or row.get("status") == "rejected":
            continue
        groups.setdefault((bucket_gb, fp), []).append(idx)
    for (bucket_gb, fp), idxs in groups.items():
        vectors = [vector_from_row(rows[i]) for i in idxs]
        merged_by_group[(bucket_gb, fp)] = merge(
            [Trial(fp=fp, vector=vector) for vector in vectors]
        )[0].vector
    # Two passes: every group competes against the full per-bucket front,
    # so a later group can demote an earlier one.
    complete_by_bucket: dict[int, list[ObjectiveVector]] = {}
    for (bucket_gb, _fp), merged in merged_by_group.items():
        if merged.complete:
            complete_by_bucket.setdefault(bucket_gb, []).append(merged)
    for (bucket_gb, fp), merged in merged_by_group.items():
        if not merged.complete:
            statuses[(bucket_gb, fp)] = "incomplete"
        elif any(dominates(other, merged) for other in complete_by_bucket.get(bucket_gb, ())):
            statuses[(bucket_gb, fp)] = "dominated"
        else:
            statuses[(bucket_gb, fp)] = "on_front"
    out = [dict(row) for row in rows]
    for (bucket_gb, fp), idxs in groups.items():
        status = persist_status(statuses[(bucket_gb, fp)])
        for idx in idxs:
            out[idx]["status"] = status
    return out
