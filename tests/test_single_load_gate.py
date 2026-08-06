"""Unit tests for the single-load gate (issue #41).

Refuse a second full server while one is live (default), let a speculative
draft ride on the same server without counting, and bypass via allow-multi
(--allow-multi / AUTORESEARCH_ALLOW_MULTI_SERVERS). No GPU, no real processes:
the #37 detection surface (listeners_on_ports / processes_by_name) is mocked.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import autoresearch.core.single_load as single_load
from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
from autoresearch.core.sglang_runner import SGLangServerRunner


def _intent(**overrides) -> ServerIntent:
    kwargs = dict(
        model_path=Path("models/test-model.gguf"),
        ctx_size=2048,
        kv_cache="q4_0",
        flash_attn="on",
        port=18080,
    )
    kwargs.update(overrides)
    return ServerIntent(**kwargs)


# ---------------------------------------------------------------------------
# allow-multi resolution
# ---------------------------------------------------------------------------


def test_allow_multi_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "0")
    assert single_load.resolve_allow_multi(allow_multi=True) is True
    assert single_load.resolve_allow_multi(allow_multi=False) is False


def test_allow_multi_env_truthy(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, value)
        assert single_load.resolve_allow_multi() is True


def test_allow_multi_env_absent_or_falsy(monkeypatch):
    monkeypatch.delenv(single_load.ALLOW_MULTI_ENV, raising=False)
    assert single_load.resolve_allow_multi() is False
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "0")
    assert single_load.resolve_allow_multi() is False
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "false")
    assert single_load.resolve_allow_multi() is False


# ---------------------------------------------------------------------------
# live-full-server detection
# ---------------------------------------------------------------------------


def test_live_full_server_is_name_plus_port_intersection(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {101, 202, 303})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {202, 404})
    assert single_load.live_full_server_pids() == [202]


def test_live_full_server_empty_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {101})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {404})
    assert single_load.live_full_server_pids() == []


# ---------------------------------------------------------------------------
# gate: refuse / allow
# ---------------------------------------------------------------------------


def test_gate_passes_without_live_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: set())
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: set())
    assert single_load.assert_single_load() == []


def test_gate_refuses_second_full_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    with pytest.raises(single_load.SingleLoadError) as ctx:
        single_load.assert_single_load()
    assert "4242" in str(ctx.value)
    assert single_load.ALLOW_MULTI_ENV in str(ctx.value)
    assert "--allow-multi" in str(ctx.value)


def test_gate_refuses_is_runtime_error():
    assert issubclass(single_load.SingleLoadError, RuntimeError)


def test_gate_bypassed_by_explicit_allow_multi(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    assert single_load.assert_single_load(allow_multi=True) == []


def test_gate_bypassed_by_env_flag(monkeypatch):
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "1")
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    assert single_load.assert_single_load() == []


def test_gate_uses_defaults_from_process_guard_surface():
    from autoresearch.core.process_guard import HARNESS_PORTS, TARGET_PROCESS_NAMES

    assert single_load.live_full_server_pids.__defaults__[0] is HARNESS_PORTS
    assert single_load.live_full_server_pids.__defaults__[1] is TARGET_PROCESS_NAMES


def test_gate_ignores_off_harness_ports_and_unrelated_names(monkeypatch):
    # PID 100 listens on a harness port but is not a target-name process
    # (e.g. LM Studio); PID 9999 matches a target name but listens on a
    # non-harness port (user's own server). Neither counts as a full server.
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {100})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {9999})
    assert single_load.live_full_server_pids() == []


def test_speculative_draft_does_not_count_as_second_server(monkeypatch):
    # A single live llama-server process serves both the main model and its
    # speculative draft (same process, --spec-draft-model flag). One process
    # on a harness port = one full server, so a new start is refused.
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    with pytest.raises(single_load.SingleLoadError):
        single_load.assert_single_load()


# ---------------------------------------------------------------------------
# Trial runner wiring (refuse + bypass, no GPU)
# ---------------------------------------------------------------------------


def test_llama_runner_enter_refuses_second_full_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    monkeypatch.setattr(
        "autoresearch.core.llama_runner.resolve_llama_server", lambda: Path("llama-server")
    )
    monkeypatch.setattr(LlamaServerRunner, "_start_vram_sampler", lambda self: None)
    sweep_calls = []
    monkeypatch.setattr(
        "autoresearch.core.llama_runner.sweep_leftover_processes",
        lambda: sweep_calls.append(1),
    )
    runner = LlamaServerRunner(_intent())
    with pytest.raises(single_load.SingleLoadError):
        runner.__enter__()
    assert sweep_calls == []
    assert runner._guard is None


def test_llama_runner_enter_allow_multi_bypasses_and_skips_sweep(monkeypatch):
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "1")
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    monkeypatch.setattr(
        "autoresearch.core.llama_runner.resolve_llama_server", lambda: Path("llama-server")
    )
    monkeypatch.setattr(LlamaServerRunner, "_start_vram_sampler", lambda self: None)
    sweep_calls = []
    monkeypatch.setattr(
        "autoresearch.core.llama_runner.sweep_leftover_processes",
        lambda: sweep_calls.append(1),
    )
    monkeypatch.setattr("autoresearch.core.llama_runner.candidate_ports", lambda port: [18080])

    class _FakeProc:
        pid = 4242

        def __init__(self):
            self._alive = True

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self._alive = False

        def kill(self):
            self._alive = False

        def wait(self, *args, **kwargs):
            self._alive = False
            return 0

    monkeypatch.setattr(
        "autoresearch.core.llama_runner.subprocess.Popen", lambda *a, **k: _FakeProc()
    )

    runner = LlamaServerRunner(_intent())
    runner._wait_for_server = lambda port: True
    entered = runner.__enter__()
    runner.__exit__(None, None, None)
    assert entered is runner
    assert sweep_calls == []


def test_sglang_runner_start_refuses_second_full_server(monkeypatch):
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    sweep_calls = []
    monkeypatch.setattr(
        "autoresearch.core.sglang_runner.sweep_leftover_processes",
        lambda: sweep_calls.append(1),
    )
    runner = SGLangServerRunner(_intent())
    with pytest.raises(single_load.SingleLoadError):
        runner.start()
    assert sweep_calls == []
    assert runner._guard is None


def test_sglang_runner_start_allow_multi_bypasses_and_skips_sweep(monkeypatch):
    monkeypatch.setenv(single_load.ALLOW_MULTI_ENV, "1")
    monkeypatch.setattr(single_load, "listeners_on_ports", lambda ports: {4242})
    monkeypatch.setattr(single_load, "processes_by_name", lambda names: {4242})
    sweep_calls = []
    monkeypatch.setattr(
        "autoresearch.core.sglang_runner.sweep_leftover_processes",
        lambda: sweep_calls.append(1),
    )
    monkeypatch.setattr("autoresearch.core.sglang_runner.candidate_ports", lambda port: [18080])
    monkeypatch.setattr(SGLangServerRunner, "is_port_in_use", lambda self, port: False)

    class _FakeProc:
        pid = 7777
        stdout = MagicMock()

        def poll(self):
            return 1

    monkeypatch.setattr(
        "autoresearch.core.sglang_runner.subprocess.Popen", lambda *a, **k: _FakeProc()
    )

    runner = SGLangServerRunner(_intent())
    with pytest.raises(RuntimeError) as ctx:
        runner.start()
    assert "Failed to start SGLang server" in str(ctx.value)
    assert not isinstance(ctx.value, single_load.SingleLoadError)
    assert sweep_calls == []
