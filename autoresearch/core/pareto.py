"""Pure Pareto nucleus: Fingerprint, Objective Vector, Domination, merge, Pareto Set.

Issue #1. No harness I/O, no results store, no Search loop. Vocabulary follows
CONTEXT.md / ADR 0006: agentic = Claw-Eval full, coding = coding-10, four
maximize axes = configured ctx, TPS, agentic, coding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

# Four maximize axes of the Objective Vector (ADR 0006).
AXES = ("ctx", "tps", "agentic", "coding")


def fingerprint(engine: Mapping[str, Any], sampler: Mapping[str, Any]) -> str:
    """Stable identity of a configuration: the full ENGINE + SAMPLER Baseline.

    ADR 0006 — engine + sampler, not model-only. Any ENGINE/SAMPLER change
    yields a different Fingerprint; key order does not matter.
    """
    payload = json.dumps(
        {"engine": dict(engine), "sampler": dict(sampler)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObjectiveVector:
    """Four maximize axes of a Trial; None = axis not measured yet."""

    ctx: int | None = None
    tps: float | None = None
    agentic: float | None = None
    coding: float | None = None

    @property
    def complete(self) -> bool:
        return all(v is not None for v in (self.ctx, self.tps, self.agentic, self.coding))


@dataclass(frozen=True)
class Trial:
    """One execution of chosen benchmarks against a single Fingerprint."""

    fp: str
    vector: ObjectiveVector


class VectorLike(Protocol):
    """Anything exposing the four Objective Vector axes (e.g. rank_results.Point)."""

    ctx: int | float | None
    tps: float | None
    agentic: float | None
    coding: float | None
    complete: bool


def dominates(a: VectorLike, b: VectorLike) -> bool:
    """True iff A is >= B on every axis and > on at least one.

    Incomplete vectors never dominate and are never dominated; they merge
    (see merge) instead of competing for the front.
    """
    if not (a.complete and b.complete):
        return False
    ge = all(getattr(a, axis) >= getattr(b, axis) for axis in AXES)
    gt = any(getattr(a, axis) > getattr(b, axis) for axis in AXES)
    return ge and gt


def pareto_set(vectors: Iterable[VectorLike]) -> list[VectorLike]:
    """Non-dominated subset of the complete input vectors, in input order.

    Empty input -> empty set; a single complete point is its own front.
    Incomplete vectors are excluded — they are not on the front.
    """
    complete = [v for v in vectors if v.complete]
    return [
        candidate
        for candidate in complete
        if not any(dominates(other, candidate) for other in complete if other is not candidate)
    ]


def merge(trials: Iterable[Trial]) -> list[Trial]:
    """Merge partial Trials by Fingerprint; one merged Trial per Fingerprint.

    Axes fill in across Trials; a merged vector stays incomplete until all
    four axes are measured. When the same axis is measured more than once
    (e.g. two claw-full runs), keep the better value — all four axes maximize.
    Sorted by Fingerprint for stable output.
    """

    def _best(a: int | float | None, b: int | float | None) -> int | float | None:
        if a is None:
            return b
        if b is None:
            return a
        return a if a >= b else b

    merged: dict[str, ObjectiveVector] = {}
    for trial in trials:
        vector = merged.get(trial.fp)
        if vector is None:
            merged[trial.fp] = trial.vector
            continue
        merged[trial.fp] = ObjectiveVector(
            ctx=_best(vector.ctx, trial.vector.ctx),
            tps=_best(vector.tps, trial.vector.tps),
            agentic=_best(vector.agentic, trial.vector.agentic),
            coding=_best(vector.coding, trial.vector.coding),
        )
    return [Trial(fp=fp, vector=merged[fp]) for fp in sorted(merged)]
