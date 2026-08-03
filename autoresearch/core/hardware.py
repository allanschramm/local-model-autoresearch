"""Host hardware detection for fit gates (Win / macOS / Linux).

Shared by check_hardware recommendations and harness host-memory preflight.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

MEMORY_CLASS_UNIFIED = "unified_memory"
MEMORY_CLASS_DISCRETE = "discrete_gpu"

# Unified: reserve at least 6 GB (or 20% of RAM) for OS/IDE.
UNIFIED_HEADROOM_FLOOR_MB = 6144.0
UNIFIED_HEADROOM_RATIO = 0.20
# Discrete: smaller host reserve (MoE experts on system RAM).
DISCRETE_HEADROOM_FLOOR_MB = 4096.0
DISCRETE_HEADROOM_RATIO = 0.15


def classify_memory_class(*, has_cuda: bool, has_metal: bool = False) -> str:
    """Discrete only when NVIDIA CUDA VRAM is present; else one shared host pool."""
    del has_metal  # API clarity for Darwin callers
    if has_cuda:
        return MEMORY_CLASS_DISCRETE
    return MEMORY_CLASS_UNIFIED


def detect_host_ram_mb() -> float | None:
    """Total physical RAM in MiB, or None if undetectable."""
    try:
        import psutil

        return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except ImportError:
        pass

    system = sys.platform
    if system == "win32":
        try:
            res = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip().isdigit()]
            if lines:
                return float(int(lines[0])) / (1024.0 * 1024.0)
        except Exception:
            pass
        return None

    if system == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            raw = (res.stdout or "").strip()
            if raw.isdigit():
                return float(int(raw)) / (1024.0 * 1024.0)
        except Exception:
            pass
        return None

    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    return float(int(parts[1])) / 1024.0  # kB → MiB
    except Exception:
        pass
    return None


def detect_nvidia() -> tuple[str | None, float, bool]:
    """Return (gpu_name, vram_gb, has_cuda)."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                return parts[0], round(float(parts[1]) / 1024.0, 1), True
    except Exception:
        pass
    return None, 0.0, False


def detect_free_vram_mb() -> float | None:
    """Free VRAM in MiB on the first NVIDIA GPU, or None if unavailable.

    Used for dynamic Trial headroom (issue #10): effective budget = free-at-start
    minus a safety margin, so Trials on a dirty GPU fail early instead of
    spuriously killing mid-eval.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        raw = (res.stdout or "").strip()
        if res.returncode == 0 and raw:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if lines:
                return float(lines[0])
    except Exception:
        pass
    return None


def has_discrete_nvidia() -> bool:
    return detect_nvidia()[2]


def detect_apple_metal() -> tuple[bool, str | None]:
    """On macOS, Metal is the GPU backend (Intel + Apple Silicon)."""
    if sys.platform != "darwin":
        return False, None
    chip = None
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        brand = (res.stdout or "").strip()
        if brand:
            chip = brand
    except Exception:
        pass
    machine = platform.machine().lower()
    if not chip:
        chip = f"macOS ({machine})"
    return True, chip


def is_unified_memory_host() -> bool:
    return not has_discrete_nvidia()


def resolve_host_headroom_mb(
    ram_mb: float,
    *,
    unified: bool | None = None,
    override_mb: float | int | None = None,
) -> float:
    """OS/IDE reserve subtracted from total RAM for host-memory budget."""
    if override_mb is not None:
        return float(override_mb)
    env = os.environ.get("AUTORESEARCH_HOST_HEADROOM_MB")
    if env:
        return float(env)
    try:
        from autoresearch.core import config as cfg

        baseline = cfg.DEFAULTS.get("HOST_MEMORY_HEADROOM_MB")
        if baseline is not None:
            return float(baseline)
    except Exception:
        pass

    if unified is None:
        unified = is_unified_memory_host()
    if unified:
        return max(UNIFIED_HEADROOM_FLOOR_MB, UNIFIED_HEADROOM_RATIO * float(ram_mb))
    return max(DISCRETE_HEADROOM_FLOOR_MB, DISCRETE_HEADROOM_RATIO * float(ram_mb))


def host_memory_budget_mb(
    ram_mb: float | None = None,
    *,
    unified: bool | None = None,
    headroom_mb: float | int | None = None,
) -> float | None:
    """Return usable host budget in MiB, or None if RAM unknown."""
    if ram_mb is None:
        ram_mb = detect_host_ram_mb()
    if ram_mb is None or ram_mb <= 0:
        return None
    if unified is None:
        unified = is_unified_memory_host()
    headroom = resolve_host_headroom_mb(ram_mb, unified=unified, override_mb=headroom_mb)
    return max(0.0, float(ram_mb) - float(headroom))


def model_pool_gb(info: dict[str, Any]) -> float:
    """Reported capacity GB: dedicated VRAM, or total unified RAM (not a safe fill target)."""
    if info.get("memory_class") == MEMORY_CLASS_DISCRETE:
        return float(info.get("vram_gb") or 0.0)
    return float(info.get("ram_gb") or 0.0)


def get_system_info() -> dict[str, Any]:
    ram_mb = detect_host_ram_mb()
    ram_gb = round(ram_mb / 1024.0, 1) if ram_mb else 0.0
    gpu_name, vram_gb, has_cuda = detect_nvidia()
    has_metal, chip = detect_apple_metal()

    if has_cuda and gpu_name:
        display_gpu = gpu_name
    elif has_metal:
        display_gpu = "Apple / macOS (Metal)"
    else:
        display_gpu = "Não detectada (CPU)"

    memory_class = classify_memory_class(has_cuda=has_cuda, has_metal=has_metal)

    return {
        "ram_gb": ram_gb,
        "ram_mb": ram_mb,
        "vram_gb": vram_gb if has_cuda else 0.0,
        "gpu_name": display_gpu,
        "has_cuda": has_cuda,
        "has_metal": has_metal,
        "memory_class": memory_class,
        "chip": chip,
        "platform": sys.platform,
        "detection_complete": ram_gb > 0.0,
    }
