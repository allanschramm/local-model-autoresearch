import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import bench_openvino


class TestBenchOpenVino(unittest.TestCase):
    def test_missing_dependency_is_actionable(self):
        with patch.object(
            bench_openvino, "_load_runtime", side_effect=RuntimeError("install openvino")
        ):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError, "install openvino"):
                    bench_openvino.benchmark(tmp, "hello", 2, "CPU")

    def test_reports_prefill_and_decode_tps(self):
        runtime = type(
            "Runtime",
            (),
            {
                "LLMPipeline": lambda *args: type(
                    "Pipe", (), {"generate": lambda self, prompt, **kwargs: "one two"}
                )()
            },
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(bench_openvino, "_load_runtime", return_value=runtime),
        ):
            with patch.object(
                bench_openvino.time, "perf_counter", side_effect=[1.0, 2.0, 3.0, 5.0]
            ):
                metrics = bench_openvino.benchmark(tmp, "hello world", 2, "CPU")
        self.assertEqual(metrics["prefill_tps"], 2.0)
        self.assertEqual(metrics["decode_tps"], 2.0)
        self.assertEqual(metrics["output_tokens"], 2.0)

    def test_missing_model_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            bench_openvino.benchmark(Path("missing-model"), "hello", 2, "CPU")


if __name__ == "__main__":
    unittest.main()
