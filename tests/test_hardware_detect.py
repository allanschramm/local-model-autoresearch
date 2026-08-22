"""Tests for detect_hardware_capabilities (issue #17).

All probes are mocked so the suite stays green on CPU-only / CI hosts.
"""

import unittest
from unittest.mock import MagicMock, mock_open, patch

from autoresearch import core
from autoresearch.core import hardware


def _nvidia_probe(has: bool):
    return (None, 0.0, has)


def _amd_probe(has: bool):
    return ("AMD Radeon RX 6600", 8.0, has) if has else (None, 0.0, False)


class TestDetectPhysicalCores(unittest.TestCase):
    def test_windows_reads_wmic_number_of_cores(self):
        proc = MagicMock(stdout="NumberOfCores\n8\n")
        with patch("autoresearch.core.hardware.sys.platform", "win32"):
            with patch("autoresearch.core.hardware.subprocess.run", return_value=proc) as run:
                self.assertEqual(hardware.detect_physical_cores(), 8)
        run.assert_called_once()

    def test_darwin_reads_sysctl_hw_physicalcpu(self):
        proc = MagicMock(stdout="10\n")
        with patch("autoresearch.core.hardware.sys.platform", "darwin"):
            with patch("autoresearch.core.hardware.subprocess.run", return_value=proc) as run:
                self.assertEqual(hardware.detect_physical_cores(), 10)
        run.assert_called_once()

    def test_linux_dedupes_physical_core_pairs(self):
        cpuinfo = (
            "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 0\n\n"
            "processor\t: 2\nphysical id\t: 0\ncore id\t\t: 1\n\n"
        )
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=cpuinfo)):
                self.assertEqual(hardware.detect_physical_cores(), 2)

    def test_falls_back_to_logical_os_cpu_count_on_probe_failure(self):
        with patch("autoresearch.core.hardware.sys.platform", "win32"):
            with patch("autoresearch.core.hardware.subprocess.run", side_effect=OSError("no wmic")):
                with patch("os.cpu_count", return_value=12):
                    self.assertEqual(hardware.detect_physical_cores(), 12)


class TestDetectSimdHints(unittest.TestCase):
    def test_linux_parses_cpuinfo_flags(self):
        cpuinfo = "processor\t: 0\nflags\t\t: fpu vme sse4_2 avx2 avx512f avx512_vnni\n"
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=cpuinfo)):
                self.assertEqual(
                    hardware.detect_simd_hints(),
                    ["avx512_vnni", "avx512f", "avx2", "sse4_2"],
                )

    def test_darwin_normalizes_mac_style_flags(self):
        proc = MagicMock(stdout="SSE4.2 AVX1.0 AVX2.0 AVX512F F16C")
        with patch("autoresearch.core.hardware.sys.platform", "darwin"):
            with patch("autoresearch.core.hardware.subprocess.run", return_value=proc) as run:
                self.assertEqual(
                    hardware.detect_simd_hints(),
                    ["avx512f", "avx2", "avx", "sse4_2", "f16c"],
                )
        run.assert_called_once()

    def test_windows_returns_empty_without_crash(self):
        with patch("autoresearch.core.hardware.sys.platform", "win32"):
            self.assertEqual(hardware.detect_simd_hints(), [])

    def test_linux_read_failure_returns_empty(self):
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", side_effect=OSError("no /proc/cpuinfo")):
                self.assertEqual(hardware.detect_simd_hints(), [])

    def test_returns_empty_when_no_relevant_flags(self):
        cpuinfo = "flags\t\t: fpu vme pse\n"
        with patch("autoresearch.core.hardware.sys.platform", "linux"):
            with patch("builtins.open", mock_open(read_data=cpuinfo)):
                self.assertEqual(hardware.detect_simd_hints(), [])


class TestGetSystemInfo(unittest.TestCase):
    def test_consumes_detect_hardware_capabilities(self):
        with patch(
            "autoresearch.core.hardware.detect_hardware_capabilities",
            return_value={"has_gpu": False, "physical_cores": 6, "ram_mb": 24576.0},
        ):
            with patch("autoresearch.core.hardware.detect_nvidia", return_value=(None, 0.0, False)):
                with patch(
                    "autoresearch.core.hardware.detect_amd", return_value=(None, 0.0, False)
                ):
                    with patch(
                        "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                    ):
                        with patch("os.cpu_count", return_value=12):
                            with patch(
                                "autoresearch.core.hardware.detect_simd_hints", return_value=[]
                            ):
                                info = hardware.get_system_info()

        self.assertEqual(info["ram_mb"], 24576.0)
        self.assertEqual(info["ram_gb"], 24.0)
        self.assertEqual(info["physical_cores"], 6)
        self.assertEqual(info["logical_cores"], 12)
        self.assertFalse(info["has_gpu"])

    def test_has_gpu_true_keeps_cuda_meaning(self):
        with patch(
            "autoresearch.core.hardware.detect_hardware_capabilities",
            return_value={"has_gpu": True, "physical_cores": 8, "ram_mb": 32768.0},
        ):
            with patch(
                "autoresearch.core.hardware.detect_nvidia", return_value=("RTX 4060", 8.0, True)
            ):
                with patch(
                    "autoresearch.core.hardware.detect_amd", return_value=(None, 0.0, False)
                ):
                    with patch(
                        "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                    ):
                        with patch("os.cpu_count", return_value=16):
                            with patch(
                                "autoresearch.core.hardware.detect_simd_hints", return_value=[]
                            ):
                                info = hardware.get_system_info()

        self.assertTrue(info["has_gpu"])
        self.assertTrue(info["has_cuda"])
        self.assertEqual(info["vram_gb"], 8.0)
        self.assertEqual(info["memory_class"], "discrete_gpu")

    def test_has_gpu_true_for_amd_radeon(self):
        with patch(
            "autoresearch.core.hardware.detect_hardware_capabilities",
            return_value={"has_gpu": True, "physical_cores": 6, "ram_mb": 16384.0},
        ):
            with patch("autoresearch.core.hardware.detect_nvidia", return_value=(None, 0.0, False)):
                with patch(
                    "autoresearch.core.hardware.detect_amd",
                    return_value=("AMD Radeon RX 6600", 8.0, True),
                ):
                    with patch(
                        "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                    ):
                        with patch("os.cpu_count", return_value=12):
                            with patch(
                                "autoresearch.core.hardware.detect_simd_hints", return_value=[]
                            ):
                                info = hardware.get_system_info()

        self.assertTrue(info["has_gpu"])
        self.assertTrue(info["has_rocm"])
        self.assertEqual(info["vram_gb"], 8.0)
        self.assertEqual(info["memory_class"], "discrete_gpu")
        self.assertIn("AMD Radeon RX 6600", info["gpu_name"])


class TestDetectHardwareCapabilities(unittest.TestCase):
    def test_returns_expected_dict_shape(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(True)):
            with patch("autoresearch.core.hardware.detect_amd", return_value=_amd_probe(False)):
                with patch(
                    "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                ):
                    with patch("autoresearch.core.hardware.detect_physical_cores", return_value=8):
                        with patch(
                            "autoresearch.core.hardware.detect_host_ram_mb", return_value=32768.0
                        ):
                            caps = hardware.detect_hardware_capabilities()

        self.assertEqual(set(caps), {"has_gpu", "physical_cores", "ram_mb"})
        self.assertIsInstance(caps["has_gpu"], bool)
        self.assertIsInstance(caps["physical_cores"], int)
        self.assertIsInstance(caps["ram_mb"], float)
        self.assertTrue(caps["has_gpu"])
        self.assertEqual(caps["physical_cores"], 8)
        self.assertEqual(caps["ram_mb"], 32768.0)

    def test_has_gpu_true_via_amd_rocm(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(False)):
            with patch("autoresearch.core.hardware.detect_amd", return_value=_amd_probe(True)):
                with patch(
                    "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                ):
                    with patch("autoresearch.core.hardware.detect_physical_cores", return_value=6):
                        with patch(
                            "autoresearch.core.hardware.detect_host_ram_mb", return_value=16384.0
                        ):
                            caps = hardware.detect_hardware_capabilities()

        self.assertTrue(caps["has_gpu"])

    def test_has_gpu_true_via_metal_on_mac(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(False)):
            with patch("autoresearch.core.hardware.detect_amd", return_value=_amd_probe(False)):
                with patch(
                    "autoresearch.core.hardware.detect_apple_metal", return_value=(True, "chip")
                ):
                    with patch("autoresearch.core.hardware.detect_physical_cores", return_value=8):
                        with patch(
                            "autoresearch.core.hardware.detect_host_ram_mb", return_value=32768.0
                        ):
                            caps = hardware.detect_hardware_capabilities()

        self.assertTrue(caps["has_gpu"])

    def test_has_gpu_false_on_cpu_only(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(False)):
            with patch("autoresearch.core.hardware.detect_amd", return_value=_amd_probe(False)):
                with patch(
                    "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                ):
                    with patch("autoresearch.core.hardware.detect_physical_cores", return_value=4):
                        with patch(
                            "autoresearch.core.hardware.detect_host_ram_mb", return_value=16384.0
                        ):
                            caps = hardware.detect_hardware_capabilities()

        self.assertFalse(caps["has_gpu"])

    def test_degrades_to_defaults_when_probes_fail(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(False)):
            with patch("autoresearch.core.hardware.detect_amd", return_value=_amd_probe(False)):
                with patch(
                    "autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)
                ):
                    with patch(
                        "autoresearch.core.hardware.detect_physical_cores", return_value=None
                    ):
                        with patch(
                            "autoresearch.core.hardware.detect_host_ram_mb", return_value=None
                        ):
                            caps = hardware.detect_hardware_capabilities()

        self.assertEqual(
            caps,
            {"has_gpu": False, "physical_cores": None, "ram_mb": None},
        )

    def test_exported_from_autoresearch_core(self):
        self.assertIs(core.detect_hardware_capabilities, hardware.detect_hardware_capabilities)
        self.assertIn("detect_hardware_capabilities", core.__all__)


if __name__ == "__main__":
    unittest.main()


class TestGpuTemp(unittest.TestCase):
    def test_nvidia_temp(self):
        proc = MagicMock(returncode=0, stdout="42\n")
        with patch("autoresearch.core.hardware.subprocess.run", return_value=proc):
            self.assertEqual(hardware.detect_gpu_temp_c(), 42.0)

    def test_probe_fail_returns_none(self):
        with patch("autoresearch.core.hardware.subprocess.run", side_effect=OSError("no smi")):
            self.assertIsNone(hardware.detect_gpu_temp_c())

    def test_wait_disabled_skips_detect(self):
        with patch("autoresearch.core.hardware.detect_gpu_temp_c") as detect:
            self.assertIsNone(hardware.wait_gpu_near_idle(idle_c=40.0, enabled=False))
            detect.assert_not_called()
