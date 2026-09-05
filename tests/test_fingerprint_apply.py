"""Fingerprint -> Baseline apply tests (issue #50, ADR 0014 phase 1).

One command copies a Fingerprint file into the mutable Baseline so the
existing Claw / SWE-lite / coding-10 runs use that engine. No eval harness
rewrite: benches keep reading Baseline; only the Baseline changes.

No GPU needed. Disk writes go to a tmp config.py copy; the in-memory
ENGINE/SAMPLER_DEFAULTS are backed up and restored per test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.core import config as cfg_mod
from autoresearch.core import fingerprint
from autoresearch.core.config import ConfigError
from autoresearch.core.fingerprint import FingerprintError, dump


@pytest.fixture
def baseline(tmp_path: Path):
    """Tmp config.py copy; restore in-memory defaults after."""
    engine_bak = dict(cfg_mod.ENGINE_DEFAULTS)
    sampler_bak = dict(cfg_mod.SAMPLER_DEFAULTS)
    dest = tmp_path / "config.py"
    dest.write_text(Path(cfg_mod.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    cfg_mod._refresh_defaults()
    yield dest
    cfg_mod.ENGINE_DEFAULTS.clear()
    cfg_mod.ENGINE_DEFAULTS.update(engine_bak)
    cfg_mod.SAMPLER_DEFAULTS.clear()
    cfg_mod.SAMPLER_DEFAULTS.update(sampler_bak)
    cfg_mod._refresh_defaults()


def _engine(**over) -> dict:
    # Dense-valid: the operator Baseline may carry MoE offload, which
    # validate_config rejects for a dense basename like applied.gguf.
    engine = dict(cfg_mod.ENGINE_DEFAULTS, N_CPU_MOE=None)
    engine.update(over)
    return engine


def test_apply_engine_updates_baseline(tmp_path: Path, baseline: Path) -> None:
    fp = dump(
        tmp_path / "m.json",
        model="applied.gguf",
        engine=_engine(CTX_SIZE=65536, THREADS=3),
    )
    result = fingerprint.apply(fp, baseline_path=baseline)
    assert result["MODEL"] == "applied.gguf"
    assert result["CTX_SIZE"] == 65536
    assert result["THREADS"] == 3
    # Benches read Baseline: the live config + the persisted file agree.
    assert cfg_mod.load_config()["CTX_SIZE"] == 65536
    assert cfg_mod.ENGINE_DEFAULTS["MODEL"] == "applied.gguf"
    text = baseline.read_text(encoding="utf-8")
    assert "'MODEL': 'applied.gguf'" in text
    assert "'CTX_SIZE': 65536" in text


def test_apply_sampler_present_updates_sampler(tmp_path: Path, baseline: Path) -> None:
    fp = dump(
        tmp_path / "m.json",
        model="applied.gguf",
        engine=_engine(),
        sampler={"TEMP": 0.35, "TOP_P": 0.9, "TOP_K": 20, "MIN_P": 0.0},
    )
    result = fingerprint.apply(fp, baseline_path=baseline)
    assert result["TEMP"] == pytest.approx(0.35)
    assert cfg_mod.SAMPLER_DEFAULTS["TEMP"] == pytest.approx(0.35)


def test_apply_omitted_sampler_leaves_sampler_alone(tmp_path: Path, baseline: Path) -> None:
    before = dict(cfg_mod.SAMPLER_DEFAULTS)
    fp = dump(
        tmp_path / "m.json",
        model="applied.gguf",
        engine=_engine(CTX_SIZE=32768),
    )
    fingerprint.apply(fp, baseline_path=baseline)
    assert dict(cfg_mod.SAMPLER_DEFAULTS) == before
    assert cfg_mod.load_config()["CTX_SIZE"] == 32768


def test_apply_rejects_bad_fingerprint(tmp_path: Path, baseline: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"model": "m.gguf", "engine": {}}', encoding="utf-8")
    with pytest.raises(FingerprintError):
        fingerprint.apply(bad, baseline_path=baseline)


def test_apply_validates_engine_values(tmp_path: Path, baseline: Path) -> None:
    fp = dump(
        tmp_path / "m.json",
        model="applied.gguf",
        engine=_engine(CTX_SIZE=512),
    )
    with pytest.raises(ConfigError):
        fingerprint.apply(fp, baseline_path=baseline)


def test_apply_pins_top_level_model_over_engine(tmp_path: Path, baseline: Path) -> None:
    # Portable/third-party file: engine lacks MODEL (or stale). The
    # validated top-level `model` must win — never old flags on old GGUF.

    engine = _engine(CTX_SIZE=32768)
    engine.pop("MODEL")
    payload = {
        "schema_version": 1,
        "model": "pinned.gguf",
        "engine": engine,
    }
    fp = tmp_path / "third.json"
    fp.write_text(json.dumps(payload), encoding="utf-8")
    result = fingerprint.apply(fp, baseline_path=baseline)
    assert result["MODEL"] == "pinned.gguf"
    assert cfg_mod.ENGINE_DEFAULTS["MODEL"] == "pinned.gguf"


def test_apply_rejects_unknown_keys(tmp_path: Path, baseline: Path) -> None:

    engine = _engine()
    engine["CTX_SIZZE"] = 65536  # typo'd key must not vanish silently
    payload = {
        "schema_version": 1,
        "model": "m.gguf",
        "engine": engine,
    }
    fp = tmp_path / "typo.json"
    fp.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FingerprintError, match="unknown Baseline keys"):
        fingerprint.apply(fp, baseline_path=baseline)
