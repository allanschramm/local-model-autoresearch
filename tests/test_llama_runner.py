import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from autoresearch.core import llama_runner
from autoresearch.core.llama_runner import (
    LlamaServerRunner,
    ServerIntent,
    engine_version_tag,
    resolve_llama_server,
)


class TestLlamaRunner(unittest.TestCase):
    def setUp(self):
        self.intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            port=18080,
        )

    def test_candidate_binary_includes_windows_release_exe(self):
        with patch.object(llama_runner, "IS_WINDOWS", True):
            paths = llama_runner._candidate_binary(Path("root"), "llama-server")

        self.assertIn(Path("root/build-cuda/bin/llama-server.exe"), paths)
        self.assertIn(Path("root/build-cuda/bin/Release/llama-server.exe"), paths)
        self.assertIn(Path("root/build-cpu/bin/llama-server.exe"), paths)
        self.assertIn(Path("root/build-cpu/bin/Release/llama-server.exe"), paths)
        self.assertIn(Path("root/build/bin/Release/llama-server.exe"), paths)

    def _check_binary_priority(self, prefer_gpu: bool) -> tuple[int, int]:
        with patch.object(llama_runner, "should_prefer_gpu_build", return_value=prefer_gpu):
            paths = llama_runner._candidate_binary(Path("root"), "llama-server")
            exe = llama_runner._exe("llama-server")
            cuda_idx = paths.index(Path(f"root/build-cuda/bin/{exe}"))
            cpu_idx = paths.index(Path(f"root/build-cpu/bin/{exe}"))
            return cuda_idx, cpu_idx

    def test_candidate_binary_priority_respects_hardware(self):
        cuda_idx, cpu_idx = self._check_binary_priority(prefer_gpu=False)
        self.assertLess(cpu_idx, cuda_idx)

        cuda_idx, cpu_idx = self._check_binary_priority(prefer_gpu=True)
        self.assertLess(cuda_idx, cpu_idx)

    def test_resolve_llama_server_found(self):
        mock_cuda = MagicMock(spec=Path)
        mock_cuda.exists.return_value = True
        mock_cuda.__str__.return_value = "/fake/cuda"
        mock_cuda.resolve.return_value = Path("/fake/cuda")
        mock_cuda.absolute.return_value = Path("/fake/cuda")

        with patch("autoresearch.core.llama_runner.LLAMA_SERVER_CANDIDATES", (mock_cuda,)):
            path = resolve_llama_server()
        self.assertEqual(path, Path("/fake/cuda"))

    def test_engine_version_tag_fork_release(self):
        server = Path(
            "D:/repo/llama.cpp-releases/turboquant/tqp-v0.3.0/build-cuda/bin/llama-server.exe"
        )
        self.assertEqual(engine_version_tag(server), "turboquant@tqp-v0.3.0")

    def test_engine_version_tag_stock_submodule(self):
        server = Path("D:/repo/llama.cpp/build-cuda/bin/llama-server.exe")
        self.assertEqual(engine_version_tag(server), "")

    def test_engine_version_tag_deep_nested_engine(self):
        server = Path("llama.cpp-releases/some-fork/v1.2.3-rc/bin/llama-server.exe")
        self.assertEqual(engine_version_tag(server), "some-fork@v1.2.3-rc")

    def test_engine_version_tag_unknown_path(self):
        self.assertEqual(engine_version_tag(Path("/usr/bin/llama-server")), "")

    def test_resolve_llama_server_not_found(self):
        mock_fail = MagicMock(spec=Path)
        mock_fail.exists.return_value = False

        with patch("autoresearch.core.llama_runner.LLAMA_SERVER_CANDIDATES", (mock_fail,)):
            with self.assertRaises(FileNotFoundError):
                resolve_llama_server()

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_basic(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        runner = LlamaServerRunner(self.intent)
        cmd = runner._build_cmd(18080)
        self.assertEqual(Path(cmd[0]), Path("/bin/llama-server"))
        self.assertIn("--port", cmd)
        self.assertIn("18080", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("autoresearch.core.llama_runner.gguf_has_mtp", return_value=True)
    def test_build_cmd_mtp(self, mock_mtp, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/Gemma-MTP.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--spec-type", cmd)
        self.assertIn("mtp", cmd)

    def test_resolve_spec_estimate_args_uses_gguf_metadata(self):
        """MTP detection must read GGUF metadata, not the filename."""
        from autoresearch.core.llama_runner import resolve_spec_estimate_args

        with patch("autoresearch.core.llama_runner.gguf_has_mtp", return_value=True):
            spec, enabled, draft = resolve_spec_estimate_args(
                Path("models/no-mtp-in-name.gguf"), None, 1, None
            )
        self.assertEqual(spec, "mtp")
        self.assertTrue(enabled)

    def test_resolve_spec_estimate_args_ignores_filename_without_metadata(self):
        """A filename containing 'MTP' must not enable spec if metadata says no."""
        from autoresearch.core.llama_runner import resolve_spec_estimate_args

        with patch("autoresearch.core.llama_runner.gguf_has_mtp", return_value=False):
            spec, enabled, draft = resolve_spec_estimate_args(
                Path("models/Fake-MTP.gguf"), None, 1, None
            )
        self.assertIsNone(spec)
        self.assertFalse(enabled)

    def test_gguf_has_mtp_reads_nextn_key(self):
        from autoresearch.core import model_arch

        class FakeField:
            def __init__(self, v):
                self._v = v

            def contents(self):
                return self._v

        fake = MagicMock()
        fake.fields = {"nemotron_h_moe.nextn_predict_layers": FakeField(1)}
        with patch("gguf.GGUFReader", return_value=fake):
            self.assertTrue(model_arch.gguf_has_mtp(Path("models/mtp-meta-test.gguf")))

    def test_gguf_has_mtp_false_when_no_key(self):
        from autoresearch.core import model_arch

        class FakeField:
            def __init__(self, v):
                self._v = v

            def contents(self):
                return self._v

        fake = MagicMock()
        fake.fields = {"nemotron_h_moe.block_count": FakeField(53)}
        with patch("gguf.GGUFReader", return_value=fake):
            self.assertFalse(model_arch.gguf_has_mtp(Path("models/no-mtp-meta.gguf")))

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_vitriol_moe(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/DeepSeek-V3-MoE-A3B.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=40,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "40")
        self.assertNotIn("--override-tensor", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_vitriol_moe_full_gpu(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/LFM2.5-8B-A1B.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=0,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--n-cpu-moe", cmd)
        self.assertEqual(cmd[cmd.index("--n-cpu-moe") + 1], "0")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_ngl_zero_cpu_only(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            ngl=0,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--n-gpu-layers", cmd)
        self.assertEqual(cmd[cmd.index("--n-gpu-layers") + 1], "0")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_numa_distribute(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            numa="distribute",
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--numa", cmd)
        self.assertEqual(cmd[cmd.index("--numa") + 1], "distribute")

    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(40, True))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_auto_n_cpu_moe_from_block_count(self, mock_path, _mock_resolve_n):
        mock_path.return_value = Path("models/moe.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "moe.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_CPU_MOE": None,
            },
            Path("models"),
        )
        self.assertEqual(intent.n_cpu_moe, 40)
        self.assertTrue(intent.n_cpu_moe_auto)

    @patch("autoresearch.core.config.is_dense_model", return_value=False)
    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(0, False))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_keeps_explicit_zero(self, mock_path, mock_resolve_n, _mock_dense):
        mock_path.return_value = Path("models/moe.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "moe.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_CPU_MOE": 0,
            },
            Path("models"),
        )
        self.assertEqual(intent.n_cpu_moe, 0)
        self.assertFalse(intent.n_cpu_moe_auto)
        mock_resolve_n.assert_called_once()
        self.assertEqual(mock_resolve_n.call_args.args[1], 0)

    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(None, False))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_reasoning_preserve(self, mock_path, _mock_resolve_n):
        mock_path.return_value = Path("models/model.gguf")
        cfg = {
            "MODEL": "model.gguf",
            "CTX_SIZE": 4096,
            "FLASH_ATTN": "on",
            "BATCH_SIZE": 512,
            "UBATCH_SIZE": 128,
        }
        on, _ = ServerIntent.from_config({**cfg, "REASONING_PRESERVE": True}, Path("models"))
        self.assertIs(on.reasoning_preserve, True)
        off, _ = ServerIntent.from_config({**cfg, "REASONING_PRESERVE": False}, Path("models"))
        self.assertIs(off.reasoning_preserve, False)
        omitted, _ = ServerIntent.from_config(cfg, Path("models"))
        self.assertIsNone(omitted.reasoning_preserve)

    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(None, False))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_prefers_n_gpu_layers_over_legacy_ngl(self, mock_path, _mock_resolve_n):
        mock_path.return_value = Path("models/model.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "model.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_GPU_LAYERS": 0,
                "ngl": 99,
            },
            Path("models"),
        )
        self.assertEqual(intent.ngl, 0)

    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(None, False))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_n_gpu_layers_minus_one_auto(self, mock_path, _mock_resolve_n):
        mock_path.return_value = Path("models/model.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "model.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "N_GPU_LAYERS": -1,
            },
            Path("models"),
        )
        self.assertEqual(intent.ngl, -1)

    @patch("autoresearch.core.llama_runner.resolve_n_cpu_moe", return_value=(None, False))
    @patch("autoresearch.core.llama_runner.resolve_model_path")
    def test_from_config_numa(self, mock_path, _mock_resolve_n):
        mock_path.return_value = Path("models/model.gguf")
        intent, _ = ServerIntent.from_config(
            {
                "MODEL": "model.gguf",
                "CTX_SIZE": 4096,
                "FLASH_ATTN": "on",
                "BATCH_SIZE": 512,
                "UBATCH_SIZE": 128,
                "NUMA": "distribute",
            },
            Path("models"),
        )
        self.assertEqual(intent.numa, "distribute")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_traditional_speculative(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/qwen35-9b.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            spec_type="draft-dflash",
            spec_draft_model="qwen35-9b-dflash-Q4_K_M.gguf",
            spec_draft_n_max=2,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)
        self.assertIn("--spec-type", cmd)
        self.assertIn("draft-dflash", cmd)
        self.assertIn("--spec-draft-model", cmd)
        expected_draft_path = Path("models/qwen35-9b-dflash-Q4_K_M.gguf")
        self.assertIn(str(expected_draft_path), cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    def test_wait_for_server_success(self, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_time.side_effect = [0, 1]
        mock_res = MagicMock()
        mock_res.status = 200
        mock_res.__enter__.return_value = mock_res
        mock_urlopen.return_value = mock_res

        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.return_value = None

        self.assertTrue(runner._wait_for_server(18080))
        mock_urlopen.assert_called_once_with(mock_urlopen.call_args.args[0], timeout=2.0)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_for_server_crash(self, _mock_sleep, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_urlopen.side_effect = Exception("Not ready")

        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.side_effect = [None, 1]

        self.assertFalse(runner._wait_for_server(18080))

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_wait_for_server_handles_vram_sampler_cleanup(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = None

        self.assertFalse(runner._wait_for_server(18080))

    @patch("autoresearch.core.llama_runner.LlamaServerRunner._start_vram_sampler")
    @patch("autoresearch.core.llama_runner.candidate_ports", return_value=[18080])
    @patch("autoresearch.core.llama_runner.subprocess.Popen")
    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_enter_reports_vram_cleanup_without_touching_closed_log(
        self, mock_resolve, _mock_popen, _mock_ports, _mock_sampler
    ):
        mock_resolve.return_value = Path("/bin/llama-server")
        runner = LlamaServerRunner(self.intent)

        def reject_for_vram(_port):
            runner.vram_killed = True
            runner._cleanup_process()
            return False

        runner._wait_for_server = reject_for_vram
        with self.assertRaisesRegex(RuntimeError, "VRAM_LIMIT_EXCEEDED"):
            runner.__enter__()

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.time")
    @patch("time.sleep")
    def test_wait_for_server_backoff(self, mock_sleep, mock_time, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_urlopen.side_effect = Exception("Not ready")

        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.side_effect = [None, None, None, 1]

        self.assertFalse(runner._wait_for_server(18080))

        # Verify backoff values
        sleep_args = [args[0] for args, kwargs in mock_sleep.call_args_list]
        self.assertEqual(sleep_args, [0.05, 0.1, 0.2])

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    @patch("time.monotonic")
    def test_wait_for_server_deadline(self, mock_monotonic, mock_sleep, mock_urlopen, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_urlopen.side_effect = Exception("Not ready")
        # deadline = monotonic() + SERVER_HEALTH_TIMEOUT_SECONDS; first check is
        # below the deadline, the second crosses it.
        mock_monotonic.side_effect = [0.0, 0.0, 301.0]
        runner = LlamaServerRunner(self.intent)
        runner._server_proc = MagicMock()
        runner._server_proc.poll.return_value = None

        self.assertFalse(runner._wait_for_server(18080))

    @patch("autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True)
    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("subprocess.check_output")
    @patch("ctypes.CDLL")
    def test_vram_sampler(self, mock_cdll, mock_output, mock_resolve, _mock_prefer_gpu):
        import threading

        called_event = threading.Event()

        def check_output_side_effect(*args, **kwargs):
            called_event.set()
            return "1000\n"

        mock_cdll.side_effect = Exception("Mock NVML load failure")
        mock_resolve.return_value = Path("/bin/llama-server")
        mock_output.side_effect = check_output_side_effect

        runner = LlamaServerRunner(self.intent)
        runner._start_vram_sampler()

        # Robust event synchronization: wait until check_output gets called
        called_event.wait(5.0)

        runner._stop_event.set()
        runner._vram_thread.join()
        self.assertGreaterEqual(runner.peak_vram_mb, 1000)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_advanced_tuning(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            kv_cache_k="f16",
            kv_cache_v="q4_0",
            threads_batch=16,
            spec_draft_n_max=2,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)

        # Verify Key/Value KV cache parameters
        self.assertIn("--cache-type-k", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-k") + 1], "f16")
        self.assertIn("--cache-type-v", cmd)
        self.assertEqual(cmd[cmd.index("--cache-type-v") + 1], "q4_0")

        # Verify Threads Batch parameters
        self.assertIn("--threads-batch", cmd)
        self.assertEqual(cmd[cmd.index("--threads-batch") + 1], "16")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    @patch("autoresearch.core.llama_runner.gguf_has_mtp", return_value=True)
    def test_build_cmd_mtp_advanced_tuning(self, mock_mtp, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/Gemma-MTP.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            kv_cache_k="q8_0",
            kv_cache_v="q4_0",
            spec_draft_n_max=3,
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)

        # Verify MTP advanced spec settings
        self.assertIn("--spec-draft-n-max", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "3")
        self.assertIn("--spec-draft-type-k", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-type-k") + 1], "q8_0")
        self.assertIn("--spec-draft-type-v", cmd)
        self.assertEqual(cmd[cmd.index("--spec-draft-type-v") + 1], "q4_0")

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_extra_flags(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            no_mmap=True,
            jinja=True,
            reasoning_budget=1024,
            reasoning_budget_message="Thinking budget reached. Proceed to final answer now.",
        )
        runner = LlamaServerRunner(intent)
        cmd = runner._build_cmd(18080)

        self.assertIn("--no-mmap", cmd)
        self.assertIn("--jinja", cmd)
        self.assertIn("--reasoning-budget", cmd)
        self.assertEqual(cmd[cmd.index("--reasoning-budget") + 1], "1024")
        self.assertIn("--reasoning-budget-message", cmd)
        self.assertEqual(
            cmd[cmd.index("--reasoning-budget-message") + 1],
            "Thinking budget reached. Proceed to final answer now.",
        )
        self.assertNotIn("--reasoning-preserve", cmd)
        self.assertNotIn("--no-reasoning-preserve", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_reasoning_preserve_on(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            reasoning_preserve=True,
        )
        cmd = LlamaServerRunner(intent)._build_cmd(18080)
        self.assertIn("--reasoning-preserve", cmd)
        self.assertNotIn("--no-reasoning-preserve", cmd)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_build_cmd_reasoning_preserve_off(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            reasoning_preserve=False,
        )
        cmd = LlamaServerRunner(intent)._build_cmd(18080)
        self.assertIn("--no-reasoning-preserve", cmd)
        self.assertNotIn("--reasoning-preserve", cmd)

    def test_estimate_vram_mb(self):
        from autoresearch.core.llama_runner import (
            VRAM_DEFAULT_QUANT_FACTOR,
            VRAM_KB_PER_TOKEN_F16,
            VRAM_OVERHEAD_MB,
            VRAM_QUANT_FACTORS,
            estimate_vram_mb,
        )

        self.assertEqual(VRAM_KB_PER_TOKEN_F16, 80.0)
        self.assertEqual(VRAM_OVERHEAD_MB, 300.0)
        self.assertEqual(VRAM_DEFAULT_QUANT_FACTOR, 0.3)
        self.assertEqual(VRAM_QUANT_FACTORS["q4"], 0.28)

        # Test with 4 arguments (backward-compatibility check)
        v1 = estimate_vram_mb(Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0")
        self.assertGreater(v1, 4000)

        # Test with 5 arguments
        v2 = estimate_vram_mb(Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0", "q4_0")
        self.assertEqual(v1, v2)

        # Test default/none cache parameters
        v3 = estimate_vram_mb(Path("models/non-existent.gguf"), 2048)
        self.assertEqual(v1, v3)

    def test_estimate_vram_mb_includes_draft(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            draft = Path(tmp) / "draft.gguf"
            draft.write_bytes(b"x" * (10 * 1024 * 1024))  # 10 MiB
            base = estimate_vram_mb(Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0")
            with_draft = estimate_vram_mb(
                Path("models/non-existent.gguf"), 2048, "q4_0", "q4_0", draft_path=draft
            )
            self.assertAlmostEqual(with_draft - base, 10.0, places=1)

    def test_estimate_vram_mb_includes_speculative_workspace_by_draft_window(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        base = estimate_vram_mb(Path("models/non-existent.gguf"), 2048)
        mtp_two = estimate_vram_mb(
            Path("models/non-existent.gguf"),
            2048,
            spec_type="draft-mtp",
            spec_draft_n_max=2,
        )
        mtp_four = estimate_vram_mb(
            Path("models/non-existent.gguf"),
            2048,
            spec_type="draft-mtp",
            spec_draft_n_max=4,
        )

        self.assertAlmostEqual(mtp_two - base, 1024.0)
        self.assertAlmostEqual(mtp_four - mtp_two, 512.0)

    def test_estimate_vram_mb_moe_external_draft_skips_spec_workspace(self):
        """MoE expert-CPU offload + external draft: charge draft weights only.

        Flat speculative workspace (512 + 256*n) false-rejects DFlash on 8 GB
        when measured peaks are ~4 GB. Embedded MTP (no draft file) keeps workspace.
        """
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            draft = Path(tmp) / "dflash.gguf"
            model.write_bytes(b"m" * (100 * 1024 * 1024))
            draft.write_bytes(b"d" * (10 * 1024 * 1024))

            with (
                patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True),
                patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40),
            ):
                base = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=40)
                with_dflash = estimate_vram_mb(
                    model,
                    2048,
                    "q4_0",
                    "q4_0",
                    n_cpu_moe=40,
                    draft_path=draft,
                    spec_type="draft-dflash",
                    spec_draft_n_max=15,
                )
                embedded_mtp = estimate_vram_mb(
                    model,
                    2048,
                    "q4_0",
                    "q4_0",
                    n_cpu_moe=40,
                    spec_type="draft-mtp",
                    spec_draft_n_max=2,
                )

            self.assertAlmostEqual(with_dflash - base, 10.0, places=1)
            self.assertAlmostEqual(embedded_mtp - base, 1024.0)

    def test_estimate_vram_mb_n_cpu_moe_shrinks_weight(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=10 * 1024 * 1024 * 1024)
            ):
                full = estimate_vram_mb(model, 2048, "q4_0", "q4_0")
                vitriol = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            self.assertLess(vitriol, full * 0.4)
            self.assertGreater(vitriol, 1000.0)

    def test_preflight_vram_rejects_over_limit(self):
        from autoresearch.core.llama_runner import preflight_vram

        ok, est, reason = preflight_vram(
            Path("models/non-existent.gguf"),
            131072,
            kv_cache_k="q4_0",
            kv_cache_v="q4_0",
            vram_limit_mb=1.0,
        )
        self.assertFalse(ok)
        self.assertGreater(est, 1.0)
        self.assertIn("VRAM_PREFLIGHT", reason)

    def test_preflight_vram_for_intent_accounts_for_configured_ctx_and_mtp(self):
        from autoresearch.core.llama_runner import preflight_vram_for_intent

        intent = ServerIntent(
            model_path=Path("models/embedded-MTP.gguf"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
            spec_type="draft-mtp",
            spec_draft_n_max=4,
        )

        ok, est, reason = preflight_vram_for_intent(intent, vram_limit_mb=8000.0)

        self.assertFalse(ok)
        self.assertGreater(est, 8000.0)
        self.assertIn("VRAM_PREFLIGHT", reason)

    def test_preflight_vram_for_intent_only_counts_enabled_external_draft(self):
        from autoresearch.core.llama_runner import preflight_vram_for_intent

        with tempfile.TemporaryDirectory() as tmp:
            draft = Path(tmp) / "draft.gguf"
            draft.write_bytes(b"x" * (10 * 1024 * 1024))
            common = dict(
                model_path=Path("models/base.gguf"),
                ctx_size=2048,
                kv_cache="q4_0",
                flash_attn="on",
                spec_draft_model=str(draft),
                spec_draft_n_max=2,
            )

            disabled = preflight_vram_for_intent(ServerIntent(**common), 10000.0)[1]
            enabled = preflight_vram_for_intent(ServerIntent(**common, spec_type="draft"), 10000.0)[
                1
            ]

        self.assertAlmostEqual(enabled - disabled, 1034.0, places=1)

    def test_preflight_vram_passes_large_moe_with_n_cpu_moe(self):
        from autoresearch.core.llama_runner import preflight_vram

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=14 * 1024 * 1024 * 1024)
            ):
                ok, est, reason = preflight_vram(
                    model,
                    65536,
                    kv_cache_k="q4_0",
                    kv_cache_v="q4_0",
                    vram_limit_mb=7900.0,
                    n_cpu_moe=30,
                )
            self.assertTrue(ok, reason)
            self.assertLessEqual(est, 7900.0)

    def test_estimate_host_memory_ignores_n_cpu_moe(self):
        from autoresearch.core.llama_runner import estimate_host_memory_mb, estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            size = 10 * 1024 * 1024 * 1024
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=size)):
                host = estimate_host_memory_mb(model, 2048, "q4_0", "q4_0")
                vram_full = estimate_vram_mb(model, 2048, "q4_0", "q4_0")
                vram_off = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            self.assertAlmostEqual(host, vram_full, places=1)
            self.assertGreater(host, vram_off * 1.5)

    def test_preflight_host_rejects_12gb_on_16gb_unified(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "big.gguf"
            model.write_bytes(b"x")
            size = 12 * 1024 * 1024 * 1024
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=size)):
                ok, est, budget, reason = preflight_host_memory(
                    model,
                    2048,
                    kv_cache_k="q4_0",
                    kv_cache_v="q4_0",
                    ram_mb=16384.0,
                    unified=True,
                )
            self.assertFalse(ok)
            self.assertIn("HOST_MEMORY_PREFLIGHT", reason)
            self.assertLess(budget, 12000.0)
            self.assertGreater(est, budget)

    def test_preflight_host_passes_when_under_budget(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "small.gguf"
            model.write_bytes(b"x")
            size = 2 * 1024 * 1024 * 1024
            with patch("pathlib.Path.stat", return_value=MagicMock(st_size=size)):
                ok, est, budget, reason = preflight_host_memory(
                    model,
                    2048,
                    kv_cache_k="q4_0",
                    kv_cache_v="q4_0",
                    ram_mb=16384.0,
                    unified=True,
                )
            self.assertTrue(ok, reason)
            self.assertEqual(reason, "")
            self.assertLessEqual(est, budget)

    def test_preflight_host_fail_closed_unified_unknown_ram(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=None):
            ok, est, budget, reason = preflight_host_memory(
                Path("models/non-existent.gguf"),
                2048,
                ram_mb=None,
                unified=True,
            )
        self.assertFalse(ok)
        self.assertIn("ram_unknown", reason)

    def test_preflight_host_discrete_unknown_ram_passes(self):
        from autoresearch.core.llama_runner import preflight_host_memory

        with patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=None):
            ok, est, budget, reason = preflight_host_memory(
                Path("models/non-existent.gguf"),
                2048,
                ram_mb=None,
                unified=False,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_estimate_vram_offload_uses_gguf_block_count(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=10 * 1024 * 1024 * 1024)
            ):
                with patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True):
                    with patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40):
                        full = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=40)
                        half = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=20)
            self.assertLess(full, half)

    def test_estimate_vram_offload_falls_back_to_32_ref(self):
        from autoresearch.core.llama_runner import VRAM_MOE_NON_EXPERT_FRAC, estimate_vram_mb

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with patch(
                "pathlib.Path.stat", return_value=MagicMock(st_size=10 * 1024 * 1024 * 1024)
            ):
                with patch(
                    "autoresearch.core.llama_runner.gguf_is_moe",
                    side_effect=RuntimeError("no arch"),
                ):
                    # n=32 / fallback 32 → full expert offload → ~28% of file + kv + overhead
                    est = estimate_vram_mb(model, 2048, "q4_0", "q4_0", n_cpu_moe=32)
            file_mb = 10 * 1024
            self.assertAlmostEqual(
                est,
                file_mb * VRAM_MOE_NON_EXPERT_FRAC + 300.0 + (2048 * 80.0 / 1024.0) * 0.28,
                delta=50.0,
            )

    def test_dense_n_cpu_moe_rejected(self):
        from autoresearch.core.config import ConfigError, validate_config

        with self.assertRaises(ConfigError) as ctx:
            validate_config(
                {
                    "MODEL": "Bonsai-27B-Q1_0.gguf",
                    "CTX_SIZE": 65536,
                    "FLASH_ATTN": "on",
                    "BATCH_SIZE": 512,
                    "UBATCH_SIZE": 128,
                    "N_CPU_MOE": 32,
                }
            )
        self.assertIn("MoE-only", str(ctx.exception))

    def test_moe_n_cpu_moe_allowed(self):
        from autoresearch.core.config import validate_config

        with patch("autoresearch.core.config.is_dense_model", return_value=False):
            cfg = validate_config(
                {
                    "MODEL": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
                    "CTX_SIZE": 65536,
                    "FLASH_ATTN": "on",
                    "BATCH_SIZE": 512,
                    "UBATCH_SIZE": 128,
                    "N_CPU_MOE": 32,
                    "VRAM_LIMIT_MB": 7900,
                }
            )
        self.assertEqual(cfg["N_CPU_MOE"], 32)

    def test_ornith_moe_via_gguf_not_filename(self):
        from autoresearch.core import model_arch
        from autoresearch.core.config import validate_config

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Ornith-1.0-35B-UD-Q4_K_XL.gguf"
            model.touch()
            with patch.object(model_arch, "gguf_is_moe", return_value=True):
                self.assertTrue(model_arch.is_moe_model(model.name, models_dir=Path(tmp)))
        with patch("autoresearch.core.config.is_dense_model", return_value=False):
            cfg = validate_config(
                {
                    "MODEL": "Ornith-1.0-35B-UD-Q4_K_XL.gguf",
                    "CTX_SIZE": 65536,
                    "FLASH_ATTN": "on",
                    "BATCH_SIZE": 512,
                    "UBATCH_SIZE": 128,
                    "N_CPU_MOE": 32,
                    "VRAM_LIMIT_MB": 7900,
                }
            )
        self.assertEqual(cfg["N_CPU_MOE"], 32)

    def test_missing_gguf_treated_dense_for_n_cpu_moe(self):
        from autoresearch.core.config import ConfigError, validate_config

        with self.assertRaises(ConfigError) as ctx:
            validate_config(
                {
                    "MODEL": "Totally-Fake-MoE-A3B.gguf",
                    "CTX_SIZE": 65536,
                    "FLASH_ATTN": "on",
                    "BATCH_SIZE": 512,
                    "UBATCH_SIZE": 128,
                    "N_CPU_MOE": 32,
                }
            )
        self.assertIn("MoE-only", str(ctx.exception))

    def test_resolve_n_cpu_moe_auto_block_count(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", return_value=(True, 41)):
                n, auto = model_arch.resolve_n_cpu_moe(path, None)
            self.assertEqual(n, 41)
            self.assertTrue(auto)
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_n_cpu_moe_explicit_and_dense(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", return_value=(True, 41)):
                n, auto = model_arch.resolve_n_cpu_moe(path, 0)
            self.assertEqual(n, 0)
            self.assertFalse(auto)
            with patch.object(model_arch, "_gguf_arch_info", return_value=(False, 22)):
                n, auto = model_arch.resolve_n_cpu_moe(path, None)
            self.assertIsNone(n)
            self.assertFalse(auto)
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_n_cpu_moe_missing_file_skips_auto(self):
        from autoresearch.core import model_arch

        n, auto = model_arch.resolve_n_cpu_moe(Path("missing-moe.gguf"), None)
        self.assertIsNone(n)
        self.assertFalse(auto)
        n, auto = model_arch.resolve_n_cpu_moe(Path("missing-moe.gguf"), 32)
        self.assertEqual(n, 32)
        self.assertFalse(auto)

    def test_resolve_n_cpu_moe_moe_without_block_count_fails(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", return_value=(True, None)):
                with self.assertRaises(ValueError) as ctx:
                    model_arch.resolve_n_cpu_moe(path, None)
            self.assertIn("block_count", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_resolve_n_cpu_moe_unreadable_file_fails_auto(self):
        from autoresearch.core import model_arch

        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with patch.object(model_arch, "_gguf_arch_info", side_effect=OSError("bad gguf")):
                with self.assertRaises(ValueError) as ctx:
                    model_arch.resolve_n_cpu_moe(path, None)
            self.assertIn("cannot read GGUF", str(ctx.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_vram_sampler_kills_dense_over_limit(self):
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

        intent = ServerIntent(
            model_path=Path("Bonsai-27B-Q1_0.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
        )
        with patch(
            "autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")
        ):
            runner = LlamaServerRunner(intent, vram_limit_mb=100.0)
        proc = MagicMock()
        runner._server_proc = proc
        # Force nvidia-smi path (no NVML)
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="500,8192\n"):
                with patch(
                    "autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True
                ):
                    runner._start_vram_sampler()
                # Allow sampler thread to fire once
                import time

                time.sleep(0.35)
                runner._stop_event.set()
                if runner._vram_thread:
                    runner._vram_thread.join(timeout=1.0)
        self.assertTrue(runner.vram_killed)
        proc.kill.assert_called()

    def test_vram_sampler_kills_absolute_shared_with_low_dedicated(self):
        """Shared>ceil kills even when dedicated is under budget (WDDM/PCI-e thrash)."""
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

        intent = ServerIntent(
            model_path=Path("moe-offload-Q4_K_M.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=30,
        )
        with patch(
            "autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")
        ):
            with patch("autoresearch.core.llama_runner.is_dense_model", return_value=False):
                with patch(
                    "autoresearch.core.llama_runner.resolve_shared_vram_limit_mb",
                    return_value=2048.0,
                ):
                    runner = LlamaServerRunner(intent, vram_limit_mb=7900.0)
        proc = MagicMock()
        proc.pid = 4242
        runner._server_proc = proc
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="4800,8188\n"):
                with patch(
                    "autoresearch.core.llama_runner.detect_pid_gpu_shared_mb",
                    return_value=15000.0,
                ):
                    with patch(
                        "autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True
                    ):
                        runner._start_vram_sampler()
                    import time

                    time.sleep(0.45)
                    runner._stop_event.set()
                    if runner._vram_thread:
                        runner._vram_thread.join(timeout=1.0)
        self.assertTrue(runner.vram_killed)
        proc.kill.assert_called()

    def test_vram_sampler_kills_moe_over_limit(self):
        """MoE must die at VRAM_LIMIT too — skipping enabled WDDM shared spill."""
        from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent

        intent = ServerIntent(
            model_path=Path("POCKET-26B-Q4_K_M.gguf"),
            ctx_size=65536,
            kv_cache="q4_0",
            flash_attn="on",
            n_cpu_moe=30,
        )
        with patch(
            "autoresearch.core.llama_runner.resolve_llama_server", return_value=Path("llama-server")
        ):
            with patch("autoresearch.core.llama_runner.is_dense_model", return_value=False):
                runner = LlamaServerRunner(intent, vram_limit_mb=100.0)
        proc = MagicMock()
        runner._server_proc = proc
        with patch("ctypes.CDLL", side_effect=OSError("no nvml")):
            with patch("subprocess.check_output", return_value="500,8192\n"):
                with patch(
                    "autoresearch.core.llama_runner.should_prefer_gpu_build", return_value=True
                ):
                    runner._start_vram_sampler()
                import time

                time.sleep(0.35)
                runner._stop_event.set()
                if runner._vram_thread:
                    runner._vram_thread.join(timeout=1.0)
        self.assertTrue(runner.vram_killed)
        proc.kill.assert_called()

    def test_resolve_vram_limit_clamps_to_physical(self):
        from autoresearch.core.llama_runner import resolve_vram_limit_mb

        with patch("autoresearch.core.llama_runner.detect_total_vram_mb", return_value=8188.0):
            # physical − keepout(512) = 7676
            self.assertEqual(resolve_vram_limit_mb(8600), 7676.0)
            self.assertEqual(resolve_vram_limit_mb(7900), 7676.0)
            self.assertEqual(resolve_vram_limit_mb(7000), 7000.0)


class TestKvCalibration(unittest.TestCase):
    """GGUF-derived KV cache sizing (sparse-GQA fix, measured 2026-08)."""

    class _FakeField:
        def __init__(self, value):
            self._value = value

        def contents(self):
            return self._value

    def _kv_bytes(self, fields):
        from autoresearch.core.model_arch import gguf_kv_bytes_per_token_f16

        fake_fields = {k: self._FakeField(v) for k, v in fields.items()}

        class FakeReader:
            def __init__(self, _path):
                self.fields = fake_fields

        with patch("gguf.GGUFReader", FakeReader):
            return gguf_kv_bytes_per_token_f16(Path("m.gguf"))

    def test_dense_scalar_head_count_kv(self):
        # llama default path: scalar kv heads on every layer
        b = self._kv_bytes(
            {
                "general.architecture": "llama",
                "llama.block_count": 32,
                "llama.embedding_length": 4096,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
            }
        )
        self.assertEqual(b, 32 * 8 * (128 + 128))

    def test_sparse_gqa_per_layer_array(self):
        # LFM2.5-8B-A1B: head_count_kv is a per-layer array (8 on attn, 0 on conv)
        b = self._kv_bytes(
            {
                "general.architecture": "lfm2moe",
                "lfm2moe.block_count": 24,
                "lfm2moe.embedding_length": 2048,
                "lfm2moe.attention.head_count": 32,
                "lfm2moe.attention.head_count_kv": [
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    8,
                    0,
                    0,
                ],
            }
        )
        self.assertEqual(b, 48 * (64 + 64))

    def _kv_f16_mb(self, fields, ctx):
        from autoresearch.core.model_arch import gguf_kv_f16_mb

        fake_fields = {k: self._FakeField(v) for k, v in fields.items()}

        class FakeReader:
            def __init__(self, _path):
                self.fields = fake_fields

        with patch("gguf.GGUFReader", FakeReader):
            return gguf_kv_f16_mb(Path("m.gguf"), ctx)

    def test_gemma4_swa_charges_window_not_full_ctx(self):
        # 5 SWA layers + 1 full; window 1024; SWA dims 256; full dims 512
        pattern = [True, True, True, True, True, False]
        kv_heads = [8, 8, 8, 8, 8, 2]
        fields = {
            "general.architecture": "gemma4",
            "gemma4.block_count": 6,
            "gemma4.embedding_length": 2816,
            "gemma4.attention.head_count": 16,
            "gemma4.attention.head_count_kv": kv_heads,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.value_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.attention.value_length_swa": 256,
            "gemma4.attention.sliding_window": 1024,
            "gemma4.attention.sliding_window_pattern": pattern,
        }
        # SWA false-path (bytes/token × full ctx) must not apply
        self.assertIsNone(self._kv_bytes(fields))
        ctx = 65536
        mb = self._kv_f16_mb(fields, ctx)
        swa_cells = 5 * 8 * (256 + 256) * 1024
        full_cells = 1 * 2 * (512 + 512) * ctx
        expected = (swa_cells + full_cells) / (1024.0 * 1024.0)
        self.assertAlmostEqual(mb, expected, places=3)
        # Old bug: charge every layer at full ctx + full dims
        bogous = (5 * 8 * (512 + 512) * ctx + full_cells) / (1024.0 * 1024.0)
        self.assertLess(mb, bogous * 0.25)

    def test_non_swa_f16_mb_matches_bytes_per_token_times_ctx(self):
        fields = {
            "general.architecture": "llama",
            "llama.block_count": 32,
            "llama.embedding_length": 4096,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
        }
        b = self._kv_bytes(fields)
        mb = self._kv_f16_mb(fields, 65536)
        self.assertAlmostEqual(mb, 65536 * b / (1024.0 * 1024.0), places=6)

    @unittest.skipUnless(
        Path("models/FINAL-Bench/pocket-26b-gguf/POCKET-26B-Q4_K_M.gguf").exists(),
        "POCKET-26B GGUF not downloaded",
    )
    def test_real_pocket26_65k_moe_offload_fits_8gb(self):
        from autoresearch.core.llama_runner import estimate_vram_mb
        from autoresearch.core.model_arch import resolve_n_cpu_moe

        p = Path("models/FINAL-Bench/pocket-26b-gguf/POCKET-26B-Q4_K_M.gguf")
        n, _ = resolve_n_cpu_moe(p, None)
        # Measured claw peak ~4.5GB @65k; old estimator 8548MB false-rejected.
        est = estimate_vram_mb(p, 65536, "q4_0", "q4_0", n_cpu_moe=n)
        self.assertLess(est, 7900.0)
        self.assertGreater(est, 3500.0)

    @unittest.skipUnless(
        Path("models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf").exists(),
        "LFM2.5-8B-A1B GGUF not downloaded",
    )
    def test_real_lfm_file_matches_measured_kv(self):
        from autoresearch.core.llama_runner import estimate_vram_mb

        p = Path("models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf")
        # est 65k q4_0 ~ 5324MB vs measured load 5399 / peak 5638 (2026-08)
        est = estimate_vram_mb(p, 65536, "q4_0", "q4_0")
        self.assertLess(est, 5600.0)
        self.assertGreater(est, 5000.0)


class TestProcessGuardWiring(unittest.TestCase):
    """Issue #39: Process Guard integration in LlamaServerRunner (pre-flight + spawn)."""

    def setUp(self):
        self.intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=2048,
            kv_cache="q4_0",
            flash_attn="on",
            port=18080,
        )

    def test_sweep_leftover_processes_delegates_to_cleanup(self):
        with patch("autoresearch.core.llama_runner.cleanup_leftover_processes") as mock_cleanup:
            mock_cleanup.return_value = [123, 456]
            llama_runner.sweep_leftover_processes()
        mock_cleanup.assert_called_once_with()

    def test_sweep_leftover_processes_fail_open_on_error(self):
        with patch(
            "autoresearch.core.llama_runner.cleanup_leftover_processes",
            side_effect=OSError("boom"),
        ):
            llama_runner.sweep_leftover_processes()  # must not raise

    def test_sweep_leftover_processes_fail_open_when_nothing_matches(self):
        with patch(
            "autoresearch.core.llama_runner.cleanup_leftover_processes",
            return_value=[],
        ):
            llama_runner.sweep_leftover_processes()  # must not raise

    @patch("autoresearch.core.llama_runner.LlamaServerRunner._start_vram_sampler")
    @patch("autoresearch.core.llama_runner.candidate_ports", return_value=[18080])
    @patch("autoresearch.core.llama_runner.sweep_leftover_processes")
    @patch("autoresearch.core.llama_runner.enforce_single_load", return_value=[])
    @patch("autoresearch.core.llama_runner.ProcessGuard")
    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_enter_sweeps_then_spawns_via_guard(
        self, mock_resolve, mock_guard_cls, _mock_gate, mock_sweep, _mock_ports, _mock_sampler
    ):
        mock_resolve.return_value = Path("/bin/llama-server")
        guard = mock_guard_cls.return_value
        runner = LlamaServerRunner(self.intent)
        runner._wait_for_server = lambda port: True  # skip real health check
        runner.__enter__()
        mock_sweep.assert_called_once()
        mock_guard_cls.assert_called_once()
        guard.spawn.assert_called_once()
        self.assertIs(runner._guard, guard)
        self.assertIs(runner._server_proc, guard.spawn.return_value)
        runner._cleanup_all()
        guard.teardown.assert_called_once()
        self.assertIsNone(runner._guard)

    @patch("autoresearch.core.llama_runner.resolve_llama_server")
    def test_cleanup_all_tears_down_guard(self, mock_resolve):
        mock_resolve.return_value = Path("/bin/llama-server")
        runner = LlamaServerRunner(self.intent)
        guard = MagicMock()
        runner._guard = guard
        runner._server_proc = MagicMock()
        runner._cleanup_all()
        guard.teardown.assert_called_once()
        self.assertIsNone(runner._guard)


class TestVramHeadroomPreflight(unittest.TestCase):
    """Issue #10: dynamic VRAM headroom from free-at-start."""

    def test_effective_limit_caps_by_free_minus_headroom(self):
        limit = llama_runner.effective_vram_limit_mb(7900.0, free_vram_mb=6000.0, headroom_mb=512.0)
        self.assertEqual(limit, 5488.0)

    def test_effective_limit_uses_configured_when_free_unknown(self):
        limit = llama_runner.effective_vram_limit_mb(7900.0, free_vram_mb=None, headroom_mb=512.0)
        self.assertEqual(limit, 7900.0)

    def test_effective_limit_never_exceeds_configured(self):
        limit = llama_runner.effective_vram_limit_mb(4000.0, free_vram_mb=6000.0, headroom_mb=0.0)
        self.assertEqual(limit, 4000.0)

    def test_preflight_effective_rejects_and_records_both_budgets(self):
        with (
            patch.object(llama_runner, "detect_free_vram_mb", return_value=6000.0),
            patch.object(llama_runner, "detect_total_vram_mb", return_value=None),
        ):
            ok, est, reason = llama_runner.preflight_vram_effective(
                Path("models/non-existent.gguf"),
                131072,
                "q4_0",
                "q4_0",
                vram_limit_mb=7900.0,
                headroom_mb=512.0,
            )
        self.assertFalse(ok)
        self.assertIn("effective=5488MB", reason)
        self.assertIn("configured=7900MB", reason)
        self.assertIn("free=6000MB", reason)

    def test_preflight_effective_passes_when_configured_binds(self):
        # free - headroom far above configured -> configured wins, no rewrite
        with patch.object(llama_runner, "detect_free_vram_mb", return_value=20000.0):
            ok, est, reason = llama_runner.preflight_vram_effective(
                Path("models/non-existent.gguf"),
                2048,
                "q4_0",
                "q4_0",
                vram_limit_mb=7900.0,
                headroom_mb=512.0,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_skip_free_clamp_env_uses_configured_budget(self):
        """AUTORESEARCH_SKIP_FREE_CLAMP=1 ignores free−headroom (operator escape)."""
        with (
            patch.object(llama_runner, "detect_free_vram_mb", return_value=6000.0),
            patch.dict(os.environ, {"AUTORESEARCH_SKIP_FREE_CLAMP": "1"}, clear=False),
        ):
            self.assertTrue(llama_runner.skip_free_vram_clamp())
            ok, est, reason = llama_runner.preflight_vram_effective(
                Path("models/non-existent.gguf"),
                2048,
                "q4_0",
                "q4_0",
                vram_limit_mb=7900.0,
                headroom_mb=512.0,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_preflight_for_intent_uses_free_vram(self):
        intent = ServerIntent(
            model_path=Path("models/test-model.gguf"),
            ctx_size=131072,
            kv_cache="q4_0",
            flash_attn="on",
        )
        with (
            patch.object(llama_runner, "detect_free_vram_mb", return_value=5000.0),
            patch.object(llama_runner, "detect_total_vram_mb", return_value=None),
        ):
            ok, _, reason = llama_runner.preflight_vram_for_intent(
                intent, 7900.0, headroom_mb=512.0
            )
        self.assertFalse(ok)
        self.assertIn("effective=4488MB", reason)
        self.assertIn("configured=7900MB", reason)

    def test_preflight_moe_offload_skips_free_vram_clamp(self):
        """MoE n_cpu_moe>0 uses configured budget; free clamp would false-reject."""
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "moe.gguf"
            model.write_bytes(b"x")
            with (
                patch("pathlib.Path.stat", return_value=MagicMock(st_size=14 * 1024 * 1024 * 1024)),
                patch.object(llama_runner, "detect_free_vram_mb", return_value=5000.0),
                patch("autoresearch.core.llama_runner.gguf_is_moe", return_value=True),
                patch("autoresearch.core.llama_runner.gguf_block_count", return_value=40),
            ):
                ok, est, reason = llama_runner.preflight_vram_effective(
                    model,
                    2048,
                    "q4_0",
                    "q4_0",
                    vram_limit_mb=7900.0,
                    n_cpu_moe=40,
                    headroom_mb=512.0,
                )
        self.assertTrue(ok, f"expected pass, est={est} reason={reason!r}")
        self.assertEqual(reason, "")
        self.assertLess(est, 7900.0)


if __name__ == "__main__":
    unittest.main()
