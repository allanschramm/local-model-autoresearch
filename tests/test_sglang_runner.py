import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoresearch.benchmarks.benchmark_harness import BenchmarkResult
from autoresearch.core.llama_runner import ServerIntent
from autoresearch.core.sglang_runner import SGLangServerRunner, run_sglang_bench_validation
from autoresearch.runners.evaluation import ExperimentRunner


class TestSGLangProcessGuardWiring(unittest.TestCase):
    """Issue #39: SGLangServerRunner uses the Process Guard (sweep + spawn + teardown)."""

    def _intent(self):
        return ServerIntent(
            model_path=Path("models/sglang/Qwen-GPTQ-Int4"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
        )

    @patch("autoresearch.core.sglang_runner.sweep_leftover_processes")
    @patch("autoresearch.core.sglang_runner.ProcessGuard")
    @patch("autoresearch.core.sglang_runner.SGLangServerRunner.is_port_in_use", return_value=False)
    @patch(
        "autoresearch.core.sglang_runner.SGLangServerRunner.is_server_ready",
        return_value=True,
    )
    @patch("autoresearch.core.sglang_runner.time.sleep")
    def test_start_sweeps_and_spawns_via_guard(
        self, _sleep, _ready, _in_use, _guard_cls, mock_sweep
    ):
        guard = MagicMock()
        _guard_cls.return_value = guard
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.return_value = "The server is fired up and ready to roll!\n"
        proc.stdout.__iter__.return_value = iter([])
        guard.spawn.return_value = proc

        runner = SGLangServerRunner(self._intent())
        port = runner.start()

        self.assertEqual(port, 18080)
        mock_sweep.assert_called_once()
        guard.spawn.assert_called_once()
        self.assertIs(runner._guard, guard)
        self.assertIs(runner._server_proc, proc)
        runner.stop()
        guard.teardown.assert_called_once()

    @patch("autoresearch.core.sglang_runner.sweep_leftover_processes")
    @patch("autoresearch.core.sglang_runner.ProcessGuard")
    @patch("autoresearch.core.sglang_runner.SGLangServerRunner.is_port_in_use", return_value=False)
    @patch(
        "autoresearch.core.sglang_runner.SGLangServerRunner.is_server_ready",
        return_value=True,
    )
    @patch("autoresearch.core.sglang_runner.time.sleep")
    def test_start_forwards_popen_group_kwargs_to_guard(
        self, _sleep, _ready, _in_use, _guard_cls, _sweep
    ):
        guard = MagicMock()
        _guard_cls.return_value = guard
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdout.readline.return_value = "Uvicorn running on http://127.0.0.1:18080\n"
        proc.stdout.__iter__.return_value = iter([])
        guard.spawn.return_value = proc

        runner = SGLangServerRunner(self._intent())
        runner.start()
        spawn_kwargs = guard.spawn.call_args.kwargs
        self.assertTrue(
            {"creationflags", "preexec_fn"} & set(spawn_kwargs),
            "expected a process-group spawn kwarg to reach guard.spawn",
        )
        runner.stop()

    @patch("autoresearch.core.sglang_runner._terminate_process_tree")
    def test_stop_tears_down_guard(self, mock_tree):
        runner = SGLangServerRunner(self._intent())
        proc = MagicMock()
        proc.poll.return_value = None
        runner._server_proc = proc
        guard = MagicMock()
        runner._guard = guard
        runner.stop()
        guard.teardown.assert_called_once()
        self.assertIsNone(runner._guard)
        mock_tree.assert_not_called()

    @patch("autoresearch.core.sglang_runner._terminate_process_tree")
    def test_stop_falls_back_to_tree_kill_without_guard(self, mock_tree):
        runner = SGLangServerRunner(self._intent())
        proc = MagicMock()
        proc.poll.return_value = None
        runner._server_proc = proc
        runner._guard = None
        runner.stop()
        mock_tree.assert_called_once_with(proc)


class TestSGLangRunner(unittest.TestCase):
    def test_build_cmd_adds_quantization_flags(self):
        intent = ServerIntent(
            model_path=Path("models/sglang/Qwen-GPTQ-Int4"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
        )
        runner = SGLangServerRunner(intent)

        cmd = runner._build_cmd(18080)

        self.assertIn("sglang.launch_server", cmd)
        self.assertIn("--context-length", cmd)
        self.assertEqual(cmd[cmd.index("--context-length") + 1], "131072")
        self.assertIn("--quantization", cmd)
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "gptq_marlin")

    @patch("subprocess.run")
    def test_sglang_bench_failure_closed_for_large_model_without_vram(self, mock_run):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(RuntimeError) as ctx:
                run_sglang_bench_validation(Path("models/sglang/Qwen-35B-GPTQ"), 1, 512, 128)

        self.assertIn("Refusing bench/server validation", str(ctx.exception))
        mock_run.assert_called_once()

    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 7000.0, 12000.0, ""),
    )
    @patch("autoresearch.core.llama_runner.detect_free_vram_mb", return_value=20000.0)
    @patch("autoresearch.runners.evaluation.LlamaServerRunner")
    @patch("autoresearch.runners.evaluation.SGLangServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_directory_model_uses_sglang_runner(
        self, mock_coding, mock_sglang, mock_llama, _mock_vram, _mock_host
    ):
        mock_sglang.return_value.__enter__.return_value = MagicMock(port=18080, peak_vram_mb=4096)
        mock_coding.return_value = BenchmarkResult(
            val_score=0.5,
            val_pass1=0.4,
            val_pass2=0.6,
            val_pass3=0.5,
            val_pass4=0.3,
            avg_tps=30.0,
        )

        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "sglang-model").mkdir()
            res = ExperimentRunner(models_dir).run_trial(
                {"model": "sglang-model", "include_coding": True},
                skip_bench=True,
            )

        self.assertEqual(res.status, "OK")
        self.assertEqual(res.val_score, 0.5)
        mock_sglang.assert_called_once()
        mock_llama.assert_not_called()

    @patch(
        "autoresearch.runners.evaluation.preflight_host_memory_for_intent",
        return_value=(True, 7000.0, 12000.0, ""),
    )
    @patch("autoresearch.core.llama_runner.detect_free_vram_mb", return_value=20000.0)
    @patch("autoresearch.runners.evaluation.run_sglang_bench_validation", return_value=10.0)
    @patch("autoresearch.runners.evaluation.SGLangServerRunner")
    @patch("autoresearch.runners.evaluation.run_coding")
    def test_sglang_bench_below_threshold_fails_before_server(
        self, mock_coding, mock_sglang, _mock_bench, _mock_vram, _mock_host
    ):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "sglang-model").mkdir()
            res = ExperimentRunner(models_dir).run_trial(
                {"model": "sglang-model", "include_coding": True, "bench_tts_threshold": 20.0},
            )

        self.assertIn("FAIL: sglang bench tg 10.0 < threshold 20.0", res.status)
        mock_sglang.assert_not_called()
        mock_coding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
