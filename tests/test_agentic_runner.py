"""Tests for autoresearch.benchmarks.agentic_runner — Claw-Eval runner and scoring."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autoresearch.benchmarks.agentic_runner import (
    ServiceManager,
    _assistant_history_message,
    _assistant_visible_text,
    run_agent_loop,
    run_agentic_eval,
    score_task,
)
from autoresearch.core.llama_client import LlamaClient


@pytest.fixture
def dummy_task_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary task directory path."""
    return tmp_path / "dummy_task"


@pytest.fixture
def mock_llama_client() -> MagicMock:
    """Fixture providing a mocked LlamaClient per tests/AGENTS.md standards."""
    return MagicMock(spec=LlamaClient)


@pytest.fixture
def sample_task() -> dict:
    """Fixture providing a sample task dictionary with scoring components."""
    return {
        "scoring_components": [
            {
                "name": "check_tool",
                "weight": 1.0,
                "check": {
                    "type": "tool_called",
                    "tool_name": "fetch_data",
                    "min_calls": 1,
                },
            },
            {
                "name": "check_keyword",
                "weight": 1.0,
                "check": {
                    "type": "keywords_present",
                    "keywords": ["success"],
                },
            },
        ]
    }


def test_score_task_tool_called_and_keywords(sample_task: dict, dummy_task_dir: Path):
    """Test score_task with tool_called and keywords_present checks."""
    tool_calls = [{"tool": "fetch_data", "arguments": {}, "result": {}, "turn": 1}]
    final_text = "Operation completed with success."

    result = score_task(sample_task, final_text, tool_calls, dummy_task_dir)

    assert result["score"] == 1.0
    assert result["tool_calls_count"] == 1
    assert result["tools_used"] == ["fetch_data"]
    assert "check_tool: PASS" in result["details"]
    assert "check_keyword: PASS" in result["details"]


def test_score_task_llm_judge_skip(dummy_task_dir: Path):
    """Test that llm_judge tasks return score 0.0 with skipped message."""
    task = {
        "scoring_components": [
            {
                "name": "judge_check",
                "weight": 1.0,
                "check": {"type": "llm_judge"},
            }
        ]
    }
    result = score_task(task, "some text", [], dummy_task_dir)
    assert result["score"] == 0.0
    assert "skipped: llm_judge" in result["details"]


def test_score_task_categories_present(dummy_task_dir: Path):
    """Test categories_present check type."""
    task = {
        "scoring_components": [
            {
                "name": "cats",
                "weight": 1.0,
                "check": {
                    "type": "categories_present",
                    "categories": ["speed", "accuracy", "reliability"],
                },
            }
        ]
    }
    text = "Detailed info on speed and accuracy in benchmark."
    result = score_task(task, text, [], dummy_task_dir)
    assert result["score"] == 1.0


def test_score_task_min_length(dummy_task_dir: Path):
    """Test min_length check type."""
    task = {
        "scoring_components": [
            {
                "name": "len",
                "weight": 1.0,
                "check": {
                    "type": "min_length",
                    "field": "final_text",
                    "min_length": 20,
                },
            }
        ]
    }
    result_fail = score_task(task, "Too short", [], dummy_task_dir)
    assert result_fail["score"] == 0.0


def test_run_agentic_eval_missing_task(mock_llama_client: MagicMock):
    """Test run_agentic_eval handles non-existent task gracefully with mocked LlamaClient."""
    res = run_agentic_eval(mock_llama_client, ["non_existent_task_xyz_123"])

    assert res["passed"] == 0
    assert res["total"] == 1
    assert res["score"] == 0.0
    assert len(res["task_results"]) == 1
    assert res["task_results"][0]["details"] == "missing"


def test_service_manager_does_not_sweep_before_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Regression: ServiceManager.start must not run the harness-port orphan sweep.

    The full harness sweep (name∩harness-port intersection) would kill the live
    llama-server on :18080 mid-Trial, causing WinError 10061 and a 0.0 agentic
    score. Pre-flight sweep lives in LlamaServerRunner, not here.
    """
    sweep = MagicMock()
    monkeypatch.setattr("autoresearch.core.llama_runner.sweep_leftover_processes", sweep)
    guard = MagicMock()
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.ProcessGuard", lambda: guard)
    monkeypatch.setattr(ServiceManager, "_wait_healthy", lambda *_: None)

    mgr = ServiceManager(
        tmp_path,
        {
            "services": [
                {
                    "name": "web",
                    "port": 9113,
                    "command": "python mock_services/web/server.py",
                }
            ]
        },
    )
    mgr.start()

    sweep.assert_not_called()
    guard.spawn.assert_called_once()
    assert mgr._guard is guard


def test_service_manager_spawns_through_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Issue #39: mock services are spawned via the Process Guard, not raw Popen."""
    guard = MagicMock()
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.ProcessGuard", lambda: guard)
    monkeypatch.setattr(ServiceManager, "_wait_healthy", lambda *_: None)

    mgr = ServiceManager(
        tmp_path,
        {
            "services": [
                {
                    "name": "web",
                    "port": 9113,
                    "command": "python mock_services/web/server.py",
                },
                {
                    "name": "db",
                    "port": 9114,
                    "command": "python mock_services/db/server.py",
                },
            ]
        },
    )
    mgr.start()

    assert guard.spawn.call_count == 2
    assert len(mgr._procs) == 2
    for call in guard.spawn.call_args_list:
        assert call.args[0][0] == sys.executable


def test_service_manager_stop_tears_down_guard(monkeypatch: pytest.MonkeyPatch):
    """Issue #39: ServiceManager.stop tears the Process Guard down and clears procs."""
    guard = MagicMock()
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.ProcessGuard", lambda: guard)
    mgr = ServiceManager(Path("dummy"), {"services": []})
    proc = MagicMock()
    mgr._procs.append(proc)

    mgr.stop()

    guard.teardown.assert_called_once()
    assert mgr._procs == []


def test_service_manager_starts_mock_with_utf8(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    popen = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.subprocess.Popen", popen)
    monkeypatch.setattr(ServiceManager, "_wait_healthy", lambda *_: None)

    ServiceManager(
        tmp_path,
        {
            "services": [
                {
                    "name": "web",
                    "port": 9113,
                    "command": "python mock_services/web/server.py",
                }
            ]
        },
    ).start()

    assert popen.call_args.kwargs["env"]["PYTHONUTF8"] == "1"


def test_run_agentic_eval_successful_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_llama_client: MagicMock,
):
    """Test run_agentic_eval full orchestration with mocked task and agent loop."""
    task_id = "test_task_001"
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True)
    yaml_path = task_dir / "task.yaml"

    task_yaml_content = """
tools:
  - name: get_weather
    description: Get weather
    input_schema:
      type: object
      properties:
        city:
          type: string
tool_endpoints:
  - tool_name: get_weather
    url: http://127.0.0.1:8080/weather
scoring_components:
  - name: check_success
    weight: 1.0
    check:
      type: keywords_present
      keywords: ["success"]
"""
    yaml_path.write_text(task_yaml_content, encoding="utf-8")

    monkeypatch.setattr("autoresearch.benchmarks.agentic_runner.TASKS_DIR", tmp_path)

    dummy_svc_mgr = MagicMock()
    dummy_svc_mgr.__enter__.return_value = dummy_svc_mgr
    dummy_svc_mgr.__exit__.return_value = False

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.ServiceManager",
        lambda tdir, tdict: dummy_svc_mgr,
    )
    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.run_agent_loop",
        lambda client, task, gen_params, max_turns: (
            "Operation completed with success.",
            [{"tool": "get_weather", "arguments": {"city": "Paris"}, "result": {}, "turn": 1}],
            0.1,
        ),
    )

    res = run_agentic_eval(mock_llama_client, [task_id], trials=1)

    assert res["passed"] == 1
    assert res["total"] == 1
    assert res["score"] == 1.0
    assert len(res["task_results"]) == 1
    assert res["task_results"][0]["score"] == 1.0
    assert "check_success: PASS" in res["task_results"][0]["details"]


def test_assistant_visible_text_falls_back_to_reasoning_content():
    """Thinking models often leave content empty; graders need reasoning_content."""
    assert _assistant_visible_text({"content": "final", "reasoning_content": "think"}) == "final"
    assert _assistant_visible_text({"content": "", "reasoning_content": "needs reply FYI"}) == (
        "needs reply FYI"
    )
    assert _assistant_visible_text({"content": None, "reasoning_content": None}) == ""


def test_assistant_history_preserves_reasoning_content():
    msg = {"content": "", "reasoning_content": "plan…"}
    hist = _assistant_history_message(msg, tool_calls=[{"id": "c1"}])
    assert hist["reasoning_content"] == "plan…"
    assert hist["tool_calls"] == [{"id": "c1"}]
    assert hist["content"] == ""


def test_run_agent_loop_uses_reasoning_when_content_empty(monkeypatch, mock_llama_client):
    """Final turn with empty content + reasoning_content must not score as blank."""
    mock_llama_client.base_url = "http://127.0.0.1:18080"
    final_payload = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "categories: needs reply, FYI, spam. Summary follows.",
                    "tool_calls": [],
                }
            }
        ]
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json

            return json.dumps(final_payload).encode()

    monkeypatch.setattr(
        "autoresearch.benchmarks.agentic_runner.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    text, calls, _elapsed = run_agent_loop(
        mock_llama_client,
        {"prompt": {"text": "triage"}, "tools": [], "tool_endpoints": []},
        max_turns=2,
    )
    assert "needs reply" in text
    assert calls == []
