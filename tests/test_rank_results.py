"""rank_results.py — Pareto / Day / Night views over results.tsv."""

from __future__ import annotations

import json

from scripts import rank_results as rr

CJ = json.dumps({"CTX_SIZE": 65536, "TEMP": 0.4, "TOP_P": 0.95, "TOP_K": 20})


def test_pareto_front_keeps_non_dominated_only():
    points = [
        rr.Point("A", ctx=65536, tps=30.0, agentic=0.6, coding=0.6),  # strong all-round
        rr.Point("B", ctx=65536, tps=100.0, agentic=0.2, coding=0.2),  # fast weak
        rr.Point("C", ctx=32768, tps=20.0, agentic=0.5, coding=0.5),  # dominated by A
    ]
    front = rr.pareto_front(points)
    names = {p.model for p in front}
    assert names == {"A", "B"}


def test_day_pick_tps_floor_then_max_iq():
    # Day TPS floor = 50.0; night (35 TPS) and kat (30 TPS) out; fast_smart (64 TPS, IQ 0.47) beats fast_weak (166 TPS, IQ 0.35)
    front = [
        rr.Point("night", ctx=65536, tps=35.0, agentic=0.67, coding=0.62),
        rr.Point("fast_smart", ctx=32768, tps=64.0, agentic=0.47, coding=0.58),
        rr.Point("fast_weak", ctx=65536, tps=166.0, agentic=0.60, coding=0.35),
    ]
    pick = rr.pick_day(front, day_tps_floor=50.0)
    assert pick is not None
    assert pick.model == "fast_smart"


def test_night_pick_ctx_floor_then_maximin():
    front = [
        rr.Point("big_iq", ctx=65536, tps=35.0, agentic=0.67, coding=0.62),
        rr.Point("small_ctx", ctx=32768, tps=64.0, agentic=0.47, coding=0.58),
        rr.Point("ok_iq", ctx=65536, tps=30.0, agentic=0.60, coding=0.64),
    ]
    pick = rr.pick_night(front, night_ctx_floor=65536)
    assert pick is not None
    assert pick.model == "big_iq"


def test_build_vectors_merges_best_valid_scores_ignores_keep_and_pollution():
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "status": "discard",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "42.1",
            "tps": "42.1",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "status": "on_front",
            "outcome": "OK",
            "val_score": "39.5",  # pollution
            "bench_tg": "",
            "tps": "",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "status": "discard",
            "outcome": "OK",
            "val_score": "0.570000",
            "bench_tg": "40.0",
            "tps": "50.0",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "N.gguf",
            "category": "agentic-full",
            "status": "on_front",
            "outcome": "OK",
            "val_score": "0.400000",
            "bench_tg": "20.0",
            "tps": "20.0",
            "ctx": "65536",
            "config_json": "",
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert [p.model for p in complete] == ["M.gguf"]
    m = complete[0]
    assert m.agentic == 0.6
    assert m.coding == 0.57
    assert m.tps == 42.1
    assert m.ctx == 65536
    assert [p.model for p in incomplete] == ["N.gguf"]


def test_build_vectors_reads_tps_ctx_from_description_when_columns_empty():
    rows = [
        {
            "model": "L.gguf",
            "category": "agentic-full",
            "status": "on_front",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "",
            "tps": "",
            "ctx": "",
            "config_json": CJ,
            "description": "L.gguf kv=f16 ctx=65536 TPS=166.4 bench_tg=166.4 | claw-full",
        },
        {
            "model": "L.gguf",
            "category": "10-task",
            "status": "discard",
            "outcome": "OK",
            "val_score": "0.350000",
            "bench_tg": "",
            "tps": "",
            "ctx": "",
            "config_json": CJ,
            "description": "",
        },
    ]
    complete, _ = rr.build_vectors(rows)
    assert len(complete) == 1
    assert complete[0].tps == 166.4
    assert complete[0].ctx == 65536


def test_build_vectors_same_basename_different_fingerprints_never_merge():
    # M.gguf measured at two different configs: agentic at fpA, coding at fpB.
    fp_a = json.dumps({"CTX_SIZE": 65536, "TEMP": 0.4})
    fp_b = json.dumps({"CTX_SIZE": 32768, "TEMP": 0.6})
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "ctx": "65536",
            "config_json": fp_a,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "ctx": "32768",
            "config_json": fp_b,
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert complete == []  # no Fingerprint has both axes -> no complete point
    assert len(incomplete) == 2  # two single-axis Fingerprints, same basename
    assert {p.model for p in incomplete} == {"M.gguf"}


def test_build_vectors_legacy_rows_without_config_json_never_complete():
    # Both axes measured but no config_json -> legacy, no Fingerprint -> incomplete.
    rows = [
        {
            "model": "L.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "ctx": "65536",
            "config_json": "",
            "description": "",
        },
        {
            "model": "L.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "ctx": "65536",
            "config_json": "",
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert complete == []
    assert [p.model for p in incomplete] == ["L.gguf"]


def test_build_vectors_uses_axis_columns_when_populated():
    # Combined modern write path: agentic-full row with both columns populated,
    # no separate 10-task row. Same Fingerprint -> complete vector.
    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "status": "on_front",
            "val_score": "0.466700",
            "agentic": "0.4667",
            "coding": "0.490000",
            "ctx": "100000",
            "tps": "47.3",
            "config_json": CJ,
            "description": "",
        },
    ]
    complete, incomplete = rr.build_vectors(rows)
    assert [p.model for p in complete] == ["M.gguf"]
    assert complete[0].agentic == 0.4667
    assert complete[0].coding == 0.49
    assert complete[0].ctx == 100000
    assert incomplete == []


def test_pick_returns_fingerprint_loadable_as_baseline():
    # ADR 0006: Day/Night pick must carry the full Fingerprint so the caller
    # can resolve it back to a Baseline config without re-searching.
    from autoresearch.core.classify import fp_from_config_json

    rows = [
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "42.1",
            "tps": "42.1",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "10-task",
            "outcome": "OK",
            "val_score": "0.570000",
            "bench_tg": "40.0",
            "tps": "40.0",
            "ctx": "65536",
            "config_json": CJ,
            "description": "",
        },
    ]
    complete, _ = rr.build_vectors(rows)
    assert len(complete) == 1
    expected_fp = fp_from_config_json(CJ)
    assert complete[0].fp == expected_fp
    assert complete[0].fp is not None
    day = rr.pick_day(complete)
    night = rr.pick_night(complete)
    assert day is not None and night is not None
    assert day.fp == expected_fp
    assert night.fp == expected_fp
    # Legacy point (no config_json) carries fp=None, never loadable.
    legacy = rr.Point("L.gguf", ctx=65536, tps=10.0, agentic=0.5, coding=0.5)
    assert legacy.fp is None


def test_day_and_night_tables_are_aligned_columns():
    front = [
        rr.Point("POCKET.gguf", ctx=65536, tps=35.7, agentic=0.6667, coding=0.6150),
        rr.Point("MTP.gguf", ctx=32768, tps=63.7, agentic=0.4667, coding=0.5800),
        rr.Point("FAST.gguf", ctx=65536, tps=166.4, agentic=0.6000, coding=0.3500),
        rr.Point("KAT.gguf", ctx=65536, tps=30.2, agentic=0.6000, coding=0.6400),
    ]
    report = rr.format_report(
        front,
        [],
        day_tps_floor=50.0,
        night_ctx_floor=65536,
        mode="pareto",
    )
    assert "DAY" in report
    assert "NIGHT" in report
    assert "| #" in report
    assert "Model" in report.splitlines()[1]
    day_section, night_section = report.split("NIGHT", 1)
    assert "MTP.gguf" in day_section
    assert "63.7" in day_section
    assert "FAST.gguf" in day_section
    assert "POCKET.gguf" not in day_section
    assert "POCKET.gguf" in night_section
    assert "35.7" in night_section
