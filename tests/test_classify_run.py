"""Integration tests: handle_single_run + _apply_flips over a real results.tsv.

Issue #4 — classifier wired into the single Trial completion path. No GPU:
run_evaluation is mocked, file I/O hits a real tmp results.tsv.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

import autoresearch.runners.run as run
from autoresearch.core import classify, config


def cfg_json(baseline: dict | None = None) -> str:
    baseline = baseline or config.load_config()
    return json.dumps(
        {k.lower(): val for k, val in baseline.items()}, sort_keys=True, separators=(",", ":")
    )


def eval_res(**over) -> dict:
    res = {
        "status": "OK",
        "outcome": "OK",
        "val_score": 0.75,
        "coding_val": 0.75,
        "coding_tps": 30.0,
        "lcb_val": 0.6,
        "he_val": 0.8,
        "mbpp_val": 0.7,
        "bigcode_val": 0.5,
        "swe_val": 0.0,
        "agentic_val": 0.6,
        "agentic_tier": "full",
        "agentic_task_count": 15,
        "avg_tps": 42.0,
        "peak_vram_gb": 7.9,
        "bench_tg_tps": 42.0,
        "bench_pp_tps": 190.0,
        "elapsed_sec": 10.0,
        "diagnostic": "",
        "task_ids": [],
        "tps_source": "coding-generation",
    }
    res.update(over)
    return res


def args(**over) -> SimpleNamespace:
    a = SimpleNamespace(
        desc="integration",
        model="M.gguf",
        kv="turbo2",
        ctx_size=131072,
        include_coding=False,
        agentic_quick=False,
        agentic_full=False,
        include_nexus=False,
        include_claw=False,
    )
    for key, value in over.items():
        setattr(a, key, value)
    return a


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "RESULTS_FILE", tmp_path / "results.tsv")
    monkeypatch.setattr(run, "get_git_commit", lambda: "abcdefg")
    return tmp_path / "results.tsv"


def test_partial_quick_only_writes_incomplete_without_agentic_axis(run_env, monkeypatch):
    # Claw quick is smoke, not the agentic axis (ADR 0006: agentic = Claw full).
    monkeypatch.setattr(run, "run_evaluation", lambda *a, **k: eval_res(agentic_tier="quick"))
    run.handle_single_run(args(agentic_quick=True))
    rows = run.read_rows(run_env)
    assert len(rows) == 1
    assert rows[0]["status"] == "incomplete"
    assert rows[0]["agentic"] == ""
    assert rows[0]["tps"] == "42.0"


def test_full_complete_writes_keep_with_axes(run_env, monkeypatch):
    monkeypatch.setattr(run, "run_evaluation", lambda *a, **k: eval_res())
    run.handle_single_run(args(include_coding=True, agentic_full=True))
    rows = run.read_rows(run_env)
    assert len(rows) == 1
    assert rows[0]["status"] == "keep"  # on_front persisted alias
    assert rows[0]["agentic"] == "0.6000"
    assert rows[0]["coding"] == "0.750000"
    assert rows[0]["ctx"] == "131072"


def test_failure_writes_rejected_and_exits(run_env, monkeypatch):
    monkeypatch.setattr(
        run,
        "run_evaluation",
        lambda *a, **k: eval_res(status="FAIL: VRAM_LIMIT_EXCEEDED", outcome="MODEL_REJECTED"),
    )
    with pytest.raises(SystemExit):
        run.handle_single_run(args())
    rows = run.read_rows(run_env)
    assert len(rows) == 1
    assert rows[0]["status"] == "rejected"


def test_merge_across_runs_flips_prior_incomplete_row(run_env, monkeypatch):
    results = [
        eval_res(agentic_tier="full", coding_val=0.75, avg_tps=42.0),  # run 1: agentic only
        eval_res(agentic_tier="", agentic_val=0.0, coding_val=0.8),  # run 2: coding only
    ]
    monkeypatch.setattr(run, "run_evaluation", lambda *a, **k: results.pop(0))
    run.handle_single_run(args(agentic_full=True))  # partial -> incomplete
    run.handle_single_run(args(include_coding=True))  # completes the Fingerprint
    rows = run.read_rows(run_env)
    assert len(rows) == 2
    assert [r["status"] for r in rows] == ["keep", "keep"]  # prior row flipped, new row classified


def test_merge_never_crosses_buckets_across_runs(run_env, monkeypatch):
    results = [
        eval_res(agentic_tier="full", coding_val=0.75, avg_tps=42.0, peak_vram_gb=7.9),
        # Same config on a different machine (16GB peak): complete on its own.
        eval_res(agentic_tier="full", agentic_val=0.6, coding_val=0.8, peak_vram_gb=16.0),
    ]
    monkeypatch.setattr(run, "run_evaluation", lambda *a, **k: results.pop(0))
    run.handle_single_run(args(agentic_full=True))  # 8GB bucket: agentic only -> incomplete
    run.handle_single_run(args(agentic_full=True, include_coding=True))  # 16GB bucket
    rows = run.read_rows(run_env)
    assert len(rows) == 2
    # No cross-budget merge: the 8GB partial is not completed by the 16GB point.
    assert [r["status"] for r in rows] == ["incomplete", "keep"]


def test_apply_flips_updates_only_matching_fp_incomplete_rows(run_env):
    fp2 = classify.fp_from_baseline(dict(config.load_config(), THREADS=8))
    assert classify.fp_from_baseline(config.load_config()) != fp2
    seed = [
        {"trial_id": "a", "status": "incomplete", "memory_gb": "8.0", "config_json": cfg_json()},
        {
            "trial_id": "b",
            "status": "incomplete",
            "memory_gb": "8.0",
            "config_json": cfg_json(dict(config.load_config(), THREADS=8)),
        },
    ]
    with open(run_env, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=run.CATEGORY_FIELDNAMES, delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(seed)
    run._apply_flips(run_env, {"a": "on_front"})
    out = {r["trial_id"]: r["status"] for r in run.read_rows(run_env)}
    assert out == {"a": "keep", "b": "incomplete"}
