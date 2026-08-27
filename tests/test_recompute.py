"""Store-wide Pareto status recompute tests (issue #5), no GPU needed.

Fixture store = tmp_path results.tsv seeded with rows; recompute runs over
the file exactly as the operator CLI would (run.recompute_statuses).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from autoresearch.core import recompute
from autoresearch.core.pareto import ObjectiveVector
from autoresearch.runners import run

BASELINE = {
    "MODEL": "M.gguf",
    "CTX_SIZE": 131072,
    "KV_CACHE": "turbo2",
    "THREADS": 6,
    "TEMP": 0.4,
    "TOP_P": 0.95,
}


def v(**kw) -> ObjectiveVector:
    return ObjectiveVector(**kw)


def cfg_json(**over) -> str:
    baseline = dict(BASELINE, **over)
    return json.dumps(
        {k.lower(): val for k, val in baseline.items()}, sort_keys=True, separators=(",", ":")
    )


def row(**kw) -> dict:
    base = {
        "trial_id": "t",
        "model": "M.gguf",
        "status": "incomplete",
        "memory_gb": "8.0",
        "config_json": cfg_json(),
        "ctx": "131072",
        "tps": "30.0",
        "agentic": "0.6",
        "coding": "0.6",
    }
    base.update(kw)
    return base


@pytest.fixture
def store(tmp_path) -> Path:
    path = tmp_path / "results.tsv"
    yield path
    path.unlink(missing_ok=True)


def write_store(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=run.CATEGORY_FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_store(path: Path) -> dict[str, str]:
    return {r["trial_id"]: r["status"] for r in run.read_rows(path)}


def test_complete_points_both_on_front_across_basenames(store):
    # ADR 0017: two DIFFERENT baselines in one bucket never demote each other —
    # every complete model appears once on the front regardless of being beaten.
    better = row(
        trial_id="better",
        model="B.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="B"),
    )
    worse = row(trial_id="worse", model="M.gguf", tps="30.0", agentic="0.6", coding="0.6")
    write_store(store, [better, worse])
    run.recompute_statuses(store)
    assert read_store(store) == {"better": "on_front", "worse": "on_front"}


def test_incomplete_and_rejected_left_out_of_domination(store):
    partial = row(
        trial_id="partial", model="P.gguf", agentic="", coding="", config_json=cfg_json(MODEL="P")
    )
    rejected = row(
        trial_id="rejected",
        model="R.gguf",
        status="rejected",
        tps="99.0",
        agentic="0.9",
        coding="0.9",
        config_json=cfg_json(MODEL="R"),
    )
    better = row(trial_id="better", tps="40.0", agentic="0.7", coding="0.7")
    write_store(store, [partial, rejected, better])
    run.recompute_statuses(store)
    assert read_store(store) == {
        "partial": "incomplete",  # never dominated, never on_front
        "rejected": "rejected",  # untouched, and does not demote better
        "better": "on_front",
    }


def test_domination_never_crosses_buckets(store):
    fp8 = row(trial_id="a8", tps="40.0", agentic="0.7", coding="0.7")
    same_fp_other_bucket = row(
        trial_id="b16", memory_gb="16.0", tps="30.0", agentic="0.6", coding="0.6"
    )
    write_store(store, [fp8, same_fp_other_bucket])
    run.recompute_statuses(store)
    assert read_store(store) == {"a8": "on_front", "b16": "on_front"}


def test_legacy_rows_without_config_json_recomputed_by_basename(store):
    # ADR 0012: basename + budget enough; config_json not required for status.
    legacy_keep = {
        "trial_id": "legacy-keep",
        "model": "L.gguf",
        "status": "keep",
        "memory_gb": "8.0",
        "config_json": "",
        "ctx": "131072",
        "tps": "30.0",
        "agentic": "0.6",
        "coding": "0.6",
    }
    legacy_discard = {**legacy_keep, "trial_id": "legacy-discard", "status": "discard"}
    write_store(store, [legacy_keep, legacy_discard])
    run.recompute_statuses(store)
    assert read_store(store) == {"legacy-keep": "on_front", "legacy-discard": "on_front"}


def test_same_fingerprint_partials_merge_to_one_status(store):
    agentic_only = row(trial_id="ag", coding="")
    coding_only = row(trial_id="cod", agentic="", tps="30.0")
    write_store(store, [agentic_only, coding_only])
    run.recompute_statuses(store)
    # Merged vector is complete -> both rows share the merged status.
    assert read_store(store) == {"ag": "on_front", "cod": "on_front"}


def test_same_fp_different_peak_vram_merges_via_vram_limit(store):
    """Coding peak 7.8 vs claw peak 7.4 must not leave the Objective Vector incomplete."""
    cfg = cfg_json(VRAM_LIMIT_MB=8100.0)
    coding_only = row(
        trial_id="cod",
        memory_gb="7.8",
        agentic="",
        coding="0.54",
        tps="48.6",
        config_json=cfg,
    )
    agentic_only = row(
        trial_id="ag",
        memory_gb="7.4",
        agentic="0.3333",
        coding="",
        tps="42.2",
        config_json=cfg,
    )
    write_store(store, [coding_only, agentic_only])
    run.recompute_statuses(store)
    assert read_store(store) == {"cod": "on_front", "ag": "on_front"}


def test_different_basenames_all_on_front_regardless_of_order(store):
    # ADR 0017: input order is irrelevant and cross-baseline demotion ceased —
    # both basenames stay on_front whatever order they were written.
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    run.recompute_statuses(store)
    assert read_store(store) == {"a": "on_front", "b": "on_front"}
    # Reversed input order, same verdict.
    write_store(store, [b, a])
    run.recompute_statuses(store)
    assert read_store(store) == {"a": "on_front", "b": "on_front"}


def test_idempotent_run_twice(store):
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    run.recompute_statuses(store)
    first = read_store(store)
    run.recompute_statuses(store)
    assert read_store(store) == first == {"a": "on_front", "b": "on_front"}


def test_bucket_and_model_scopes_agree_per_basename(store):
    # ADR 0017: domination is same-basename everywhere, so bucket and model
    # scopes produce identical verdicts — no more cross-baseline demotion.
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    run.recompute_statuses(store)
    assert read_store(store) == {"a": "on_front", "b": "on_front"}
    out = {
        r["trial_id"]: r["status"]
        for r in recompute.recompute_rows(run.read_rows(store), scope="model")
    }
    assert out == {"a": "on_front", "b": "on_front"}


def test_model_scope_respects_bucket_isolation(store):
    # Same model, two buckets: the 8GB point must not be demoted by the 16GB
    # point in the per-model lens either (domination never crosses budgets).
    rows = [
        row(trial_id="a8", tps="30.0", agentic="0.6", coding="0.6"),
        row(trial_id="b16", memory_gb="16.0", tps="40.0", agentic="0.7", coding="0.7"),
    ]
    out = {r["trial_id"]: r["status"] for r in recompute.recompute_rows(rows, scope="model")}
    assert out == {"a8": "on_front", "b16": "on_front"}


def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        recompute.recompute_rows([], scope="machine")


def test_recompute_rows_pure_and_idempotent():
    rows = [
        row(
            trial_id="a",
            model="A.gguf",
            tps="30.0",
            agentic="0.6",
            coding="0.6",
            config_json=cfg_json(MODEL="A"),
        ),
        row(
            trial_id="b",
            model="B.gguf",
            tps="40.0",
            agentic="0.7",
            coding="0.7",
            config_json=cfg_json(MODEL="B"),
        ),
    ]
    first = recompute.recompute_rows(rows)
    # Input untouched (pure function).
    assert rows[0]["status"] == "incomplete" and rows[1]["status"] == "incomplete"
    assert recompute.recompute_rows(first) == first


def test_cli_runs_from_repo_root(store):
    write_store(store, [row(trial_id="a", tps="30.0", agentic="0.6", coding="0.6")])
    proc = subprocess.run(
        [sys.executable, "scripts/recompute_status.py", str(store)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "statuses refreshed" in proc.stdout
    assert read_store(store) == {"a": "on_front"}


def test_cli_model_scope_prints_without_rewrite(store):
    a = row(
        trial_id="a",
        model="A.gguf",
        tps="40.0",
        agentic="0.7",
        coding="0.7",
        config_json=cfg_json(MODEL="A"),
    )
    b = row(
        trial_id="b",
        model="B.gguf",
        tps="30.0",
        agentic="0.6",
        coding="0.6",
        config_json=cfg_json(MODEL="B"),
    )
    write_store(store, [a, b])
    proc = subprocess.run(
        [sys.executable, "scripts/recompute_status.py", "--scope", "model", str(store)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "a\tA.gguf\ton_front" in proc.stdout
    assert "b\tB.gguf\ton_front" in proc.stdout
    # Read-only: stored statuses unchanged.
    assert read_store(store) == {"a": "incomplete", "b": "incomplete"}


def test_cli_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "scripts/recompute_status.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0


def test_morris_screen_rows_never_join_domination(store):
    # ADR 0016: a reps=1 Morris screen point must not compete with, demote,
    # or merge into real Trials — even in the same bucket and model.
    real = row(trial_id="real", tps="30.0", agentic="0.6", coding="0.6")
    screen = row(
        trial_id="scr",
        evaluation_profile="morris-screen",
        category="morris-screen",
        tps="99.0",
        agentic="",
        coding="",
        outcome="OK",
    )
    write_store(store, [screen, real])
    run.recompute_statuses(store)
    assert read_store(store) == {"real": "on_front", "scr": "incomplete"}
