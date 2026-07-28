"""rank_results.py — Pareto / Day / Night views over results.tsv."""
from __future__ import annotations

from scripts import rank_results as rr


def test_pareto_front_keeps_non_dominated_only():
    points = [
        rr.Point("A", ctx=65536, tps=30.0, agentic=0.6, coding=0.6),  # strong all-round
        rr.Point("B", ctx=65536, tps=100.0, agentic=0.2, coding=0.2),  # fast weak
        rr.Point("C", ctx=32768, tps=20.0, agentic=0.5, coding=0.5),  # dominated by A
    ]
    front = rr.pareto_front(points)
    names = {p.model for p in front}
    assert names == {"A", "B"}


def test_day_pick_iq_band_then_max_tps():
    # IQ_best = min(0.6,0.64)=0.6 → floor 0.45; A min=0.35 out; B wins TPS in band
    front = [
        rr.Point("night", ctx=65536, tps=35.0, agentic=0.67, coding=0.62),
        rr.Point("day", ctx=32768, tps=64.0, agentic=0.47, coding=0.58),
        rr.Point("fast_weak", ctx=65536, tps=166.0, agentic=0.60, coding=0.35),
    ]
    pick = rr.pick_day(front, day_iq_ratio=0.75)
    assert pick is not None
    assert pick.model == "day"


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
            "config_json": "",
            "description": "",
        },
        {
            "model": "M.gguf",
            "category": "agentic-full",
            "status": "keep",
            "outcome": "OK",
            "val_score": "39.5",  # pollution
            "bench_tg": "",
            "tps": "",
            "ctx": "65536",
            "config_json": "",
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
            "ctx": "32768",
            "config_json": "",
            "description": "",
        },
        {
            "model": "N.gguf",
            "category": "agentic-full",
            "status": "keep",
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
            "status": "keep",
            "outcome": "OK",
            "val_score": "0.600000",
            "bench_tg": "",
            "tps": "",
            "ctx": "",
            "config_json": "",
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
            "config_json": "",
            "description": "",
        },
    ]
    complete, _ = rr.build_vectors(rows)
    assert len(complete) == 1
    assert complete[0].tps == 166.4
    assert complete[0].ctx == 65536


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
        day_iq_ratio=0.75,
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
    assert "FAST.gguf" not in day_section
    assert "POCKET.gguf" in night_section
    assert "35.7" in night_section
