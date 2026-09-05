"""Trial rejects Fingerprint vs Baseline mismatch (issue #53, ADR 0014).

A Trial scores the live Baseline flags. When a Fingerprint file exists for
MODEL, its frozen engine must match the Baseline server flags — otherwise the
Trial would score (and log) the wrong flags. Mismatch (or an unreadable file)
rejects the Trial with a clear reason; no Fingerprint file keeps the existing
Baseline-only behavior.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from autoresearch.core import fingerprint
from autoresearch.core.fingerprint import dump


def _engine(**over) -> dict:
    engine = {
        "MODEL": "trial.gguf",
        "CTX_SIZE": 65536,
        "BATCH_SIZE": 512,
        "N_CPU_MOE": None,
        "SPEC_TYPE": None,
        "TPS_FLOOR": 20.0,
        "VRAM_LIMIT_MB": 7900.0,
    }
    engine.update(over)
    return engine


def test_no_file_means_baseline_only_proceeds(tmp_path: Path) -> None:
    assert fingerprint.mismatch_reason("trial.gguf", _engine(), directory=tmp_path) is None


def test_matching_file_proceeds(tmp_path: Path) -> None:
    engine = _engine()
    dump(tmp_path / "trial.json", model="trial.gguf", engine=engine)
    assert fingerprint.mismatch_reason("trial.gguf", engine, directory=tmp_path) is None


def test_harness_only_drift_proceeds(tmp_path: Path) -> None:
    """TPS_FLOOR / VRAM_LIMIT tune gates, not flags — never a reject."""
    dump(tmp_path / "trial.json", model="trial.gguf", engine=_engine())
    drifting = _engine(TPS_FLOOR=12.0, VRAM_LIMIT_MB=8000.0)
    assert fingerprint.mismatch_reason("trial.gguf", drifting, directory=tmp_path) is None


def test_server_flag_drift_rejects_with_key_and_values(tmp_path: Path) -> None:
    dump(tmp_path / "trial.json", model="trial.gguf", engine=_engine())
    reason = fingerprint.mismatch_reason("trial.gguf", _engine(CTX_SIZE=32768), directory=tmp_path)
    assert reason is not None
    assert "CTX_SIZE" in reason
    assert "32768" in reason and "65536" in reason


def test_wrong_model_identity_rejects(tmp_path: Path) -> None:
    dump(tmp_path / "other.json", model="other.gguf", engine=_engine(MODEL="other.gguf"))
    # Hand-copied file: content identity does not match the requested basename.
    other_as_trial = tmp_path / "trial.json"
    (tmp_path / "other.json").rename(other_as_trial)
    reason = fingerprint.mismatch_reason("trial.gguf", _engine(), directory=tmp_path)
    assert reason is not None
    assert "other.gguf" in reason


def test_invalid_file_rejects_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "trial.json").write_text("{not json", encoding="utf-8")
    reason = fingerprint.mismatch_reason("trial.gguf", _engine(), directory=tmp_path)
    assert reason is not None
    assert "invalid" in reason.lower()


def test_unknown_file_key_rejects(tmp_path: Path) -> None:
    engine = _engine()
    engine["TYPOD_KEY"] = 1
    dump(tmp_path / "trial.json", model="trial.gguf", engine=engine)
    reason = fingerprint.mismatch_reason("trial.gguf", _engine(), directory=tmp_path)
    assert reason is not None
    assert "TYPOD_KEY" in reason


def _args(**over) -> SimpleNamespace:
    base = {"model": "trial.gguf", "desc": "t"}
    base.update(over)
    return SimpleNamespace(**base)


def test_gate_helper_no_file_returns_none(tmp_path: Path) -> None:
    from autoresearch.runners import run

    with (
        patch.object(run.config, "load_config", return_value=_engine()),
        patch(
            "autoresearch.core.fingerprint.path_for",
            return_value=tmp_path / "trial.json",
        ),
    ):
        assert run.fingerprint_reject_reason(_args()) is None


def test_gate_helper_mismatch_returns_reason(tmp_path: Path) -> None:
    from autoresearch.runners import run

    dump(tmp_path / "trial.json", model="trial.gguf", engine=_engine())
    with (
        patch.object(run.config, "load_config", return_value=_engine(CTX_SIZE=32768)),
        patch(
            "autoresearch.core.fingerprint.path_for",
            return_value=tmp_path / "trial.json",
        ),
    ):
        reason = run.fingerprint_reject_reason(_args())
    assert reason is not None
    assert "CTX_SIZE" in reason


def test_handle_single_run_mismatch_never_scores(tmp_path: Path) -> None:
    """Mismatch: rejected row, no eval, exit 1 — never a quality score."""
    from autoresearch.runners import run
    from autoresearch.runners.evaluation import TrialOutcome

    dump(tmp_path / "trial.json", model="trial.gguf", engine=_engine())
    args = _args(ctx_size=32768)
    with (
        patch.object(run.config, "load_config", return_value=_engine(CTX_SIZE=32768)),
        patch(
            "autoresearch.core.fingerprint.path_for",
            return_value=tmp_path / "trial.json",
        ),
        patch("autoresearch.runners.run.run_evaluation") as mock_eval,
        patch("autoresearch.runners.run.write_row") as mock_write,
        patch("autoresearch.runners.run.recompute_statuses"),
    ):
        try:
            run.handle_single_run(args)
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("handle_single_run must exit 1 on Fingerprint mismatch")
    mock_eval.assert_not_called()
    mock_write.assert_called_once()
    status = mock_write.call_args.args[7]
    assert status == "rejected"
    kwargs = mock_write.call_args.kwargs
    assert kwargs.get("outcome") == TrialOutcome.MODEL_REJECTED.value
    assert "CTX_SIZE" in (kwargs.get("diagnostic") or "")
