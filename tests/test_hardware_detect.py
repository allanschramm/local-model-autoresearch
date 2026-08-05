"""Tests for detect_hardware_capabilities (issue #17).

All probes are mocked so the suite stays green on CPU-only / CI hosts.
"""

import unittest
from unittest.mock import MagicMock, mock_open, patch

from autoresearch import core
from autoresearch.core import hardware


def _nvidia_probe(has: bool):
    return (None, 0.0, has)


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


class TestDetectHardwareCapabilities(unittest.TestCase):
    def test_returns_expected_dict_shape(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(True)):
            with patch("autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)):
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

    def test_has_gpu_true_via_metal_on_mac(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(False)):
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
            with patch("autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)):
                with patch("autoresearch.core.hardware.detect_physical_cores", return_value=4):
                    with patch(
                        "autoresearch.core.hardware.detect_host_ram_mb", return_value=16384.0
                    ):
                        caps = hardware.detect_hardware_capabilities()

        self.assertFalse(caps["has_gpu"])

    def test_degrades_to_defaults_when_probes_fail(self):
        with patch("autoresearch.core.hardware.detect_nvidia", return_value=_nvidia_probe(False)):
            with patch("autoresearch.core.hardware.detect_apple_metal", return_value=(False, None)):
                with patch("autoresearch.core.hardware.detect_physical_cores", return_value=None):
                    with patch("autoresearch.core.hardware.detect_host_ram_mb", return_value=None):
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
