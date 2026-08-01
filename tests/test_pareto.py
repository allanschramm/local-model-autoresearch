"""Pure Pareto nucleus tests (issue #1): fingerprint / dominate / tie / merge / empty / single."""

from __future__ import annotations

from autoresearch.core.pareto import (
    ObjectiveVector,
    Trial,
    dominates,
    fingerprint,
    merge,
    pareto_set,
)

_ENGINE = {
    "MODEL": "M.gguf",
    "CTX_SIZE": 131072,
    "KV_CACHE": "turbo2",
    "BATCH_SIZE": 64,
    "THREADS": 6,
    "FLASH_ATTN": "on",
    "N_CPU_MOE": None,
    "VRAM_LIMIT_MB": 7900,
}
_SAMPLER = {"TEMP": 0.4, "TOP_P": 0.95, "TOP_K": 20, "MIN_P": 0.0}


def v(**kw) -> ObjectiveVector:
    return ObjectiveVector(**kw)


def test_fingerprint_identity_matches_adr_engine_plus_sampler():
    # Same engine + sampler, key order irrelevant -> same Fingerprint.
    assert fingerprint(_ENGINE, _SAMPLER) == fingerprint(dict(_ENGINE), dict(_SAMPLER))
    assert fingerprint(dict(reversed(list(_ENGINE.items()))), _SAMPLER) == fingerprint(
        _ENGINE, _SAMPLER
    )
    # Engine-only change -> different Fingerprint (not model-only identity).
    assert fingerprint(dict(_ENGINE, THREADS=8), _SAMPLER) != fingerprint(_ENGINE, _SAMPLER)
    # Sampler-only change -> different Fingerprint.
    assert fingerprint(_ENGINE, dict(_SAMPLER, TEMP=0.7)) != fingerprint(_ENGINE, _SAMPLER)


def test_dominates_ge_all_axes_and_gt_at_least_one():
    strong = v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6)
    weak = v(ctx=65536, tps=20.0, agentic=0.5, coding=0.5)
    assert dominates(strong, weak)
    assert not dominates(weak, strong)


def test_dominates_exact_tie_is_false():
    a = v(ctx=65536, tps=30.0, agentic=0.6, coding=0.6)
    b = v(ctx=65536, tps=30.0, agentic=0.6, coding=0.6)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_dominates_incomplete_never_dominates():
    full = v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6)
    partial = v(ctx=131072, tps=30.0, agentic=0.6)
    assert full.complete
    assert not partial.complete
    assert not dominates(full, partial)
    assert not dominates(partial, full)


def test_merge_combines_partial_trials_by_fingerprint():
    fp = fingerprint(_ENGINE, _SAMPLER)
    merged = merge(
        [
            Trial(fp=fp, vector=v(ctx=131072, tps=30.0, agentic=0.6)),
            Trial(fp=fp, vector=v(ctx=131072, coding=0.5)),
        ]
    )
    assert len(merged) == 1
    assert merged[0].fp == fp
    assert merged[0].vector.complete
    assert merged[0].vector.agentic == 0.6
    assert merged[0].vector.coding == 0.5
    assert merged[0].vector.tps == 30.0


def test_merge_stays_incomplete_until_all_axes_measured():
    fp = fingerprint(_ENGINE, _SAMPLER)
    merged = merge(
        [
            Trial(fp=fp, vector=v(agentic=0.6)),
            Trial(fp=fp, vector=v(agentic=0.62, coding=0.5)),
        ]
    )
    assert len(merged) == 1
    assert not merged[0].vector.complete  # ctx and tps still missing
    assert merged[0].vector.agentic == 0.6  # first non-None value wins


def test_merge_separates_distinct_fingerprints_and_sorts():
    fp_a = fingerprint(_ENGINE, _SAMPLER)
    fp_b = fingerprint(dict(_ENGINE, THREADS=8), _SAMPLER)
    merged = merge(
        [
            Trial(fp=fp_b, vector=v(agentic=0.5)),
            Trial(fp=fp_a, vector=v(coding=0.5)),
        ]
    )
    assert [t.fp for t in merged] == sorted([fp_a, fp_b])
    assert len(merged) == 2


def test_pareto_set_returns_non_dominated_complete_vectors():
    strong = v(ctx=131072, tps=30.0, agentic=0.6, coding=0.6)
    fast_weak = v(ctx=131072, tps=100.0, agentic=0.2, coding=0.2)
    dominated = v(ctx=32768, tps=20.0, agentic=0.5, coding=0.5)
    partial = v(ctx=131072, tps=30.0, agentic=0.6)
    front = pareto_set([strong, fast_weak, dominated, partial])
    assert front == [strong, fast_weak]


def test_pareto_set_empty_and_single_point():
    assert pareto_set([]) == []
    alone = v(ctx=65536, tps=30.0, agentic=0.6, coding=0.6)
    assert pareto_set([alone]) == [alone]


def test_merged_trial_reaches_front_only_when_complete():
    fp = fingerprint(_ENGINE, _SAMPLER)
    partial = [Trial(fp=fp, vector=v(agentic=0.6, coding=0.6))]
    assert pareto_set(t.vector for t in partial) == []  # no ctx/tps -> not on front
    merged = merge(partial + [Trial(fp=fp, vector=v(ctx=131072, tps=30.0))])
    assert merged[0].vector.complete
    assert pareto_set(t.vector for t in merged) == [merged[0].vector]
