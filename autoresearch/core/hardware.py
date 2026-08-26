"""Host hardware detection for fit gates (Win / macOS / Linux).

Shared by check_hardware recommendations and harness host-memory preflight.
`detect_hardware_capabilities()` is the source-of-truth probe for the Search
loop (has_gpu / physical_cores / ram_mb).
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


def _spawn_kwargs() -> dict:
    """Windows: console probes (wmic/powershell/nvidia-smi/typeperf) must never pop a terminal."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def classify_memory_class(
    *, has_cuda: bool, has_rocm: bool = False, has_metal: bool = False
) -> str:
    """Discrete when NVIDIA CUDA or AMD ROCm/Radeon VRAM is present; else shared host pool."""
    del has_metal  # API clarity for Darwin callers
    if has_cuda or has_rocm:
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
                **_spawn_kwargs(),
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
            **_spawn_kwargs(),
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                return parts[0], round(float(parts[1]) / 1024.0, 1), True
    except Exception:
        pass
    return None, 0.0, False


def detect_gpu_temp_c() -> float | None:
    """First GPU temperature in C, or None if no sensor."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            **_spawn_kwargs(),
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            return float(line.strip())
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["rocm-smi", "--showtemp"],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            import re

            match = re.search(r"([\d.]+)\s*[cC]", res.stdout)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return None


def wait_gpu_near_idle(
    *,
    idle_c: float | None,
    delta_c: float = 8.0,
    timeout_s: float = 90.0,
    poll_s: float = 2.0,
    enabled: bool = True,
) -> float | None:
    """Poll until GPU temp is near the captured idle, or timeout."""
    import time

    if not enabled:
        return None
    current = detect_gpu_temp_c()
    if idle_c is None or current is None:
        return current
    deadline = time.monotonic() + timeout_s
    while current > idle_c + delta_c and time.monotonic() < deadline:
        time.sleep(poll_s)
        nxt = detect_gpu_temp_c()
        if nxt is None:
            return current
        current = nxt
    return current


def detect_amd() -> tuple[str | None, float, bool]:
    """Return (gpu_name, vram_gb, has_rocm_or_amd)."""
    system = sys.platform
    if system == "win32":
        try:
            res = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\000*' "
                        "-ErrorAction SilentlyContinue | Where-Object { $_.DriverDesc -like '*AMD*' -or $_.DriverDesc -like '*Radeon*' } "
                        "| ForEach-Object { [string]$_.DriverDesc + '|' + [string]$_.'HardwareInformation.qwMemorySize' }"
                    ),
                ],
                capture_output=True,
                text=True,
                check=False,
                **_spawn_kwargs(),
            )
            if res.returncode == 0 and res.stdout.strip():
                line = res.stdout.strip().splitlines()[0]
                if "|" in line:
                    desc, size_str = line.split("|", 1)
                    desc = desc.strip()
                    vram_bytes = int(size_str.strip()) if size_str.strip().isdigit() else 0
                    vram_gb = (
                        round(vram_bytes / (1024.0 * 1024.0 * 1024.0), 1) if vram_bytes > 0 else 0.0
                    )
                    if desc and vram_gb > 0:
                        return desc, vram_gb, True
        except Exception:
            pass
    elif system == "linux":
        try:
            res = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return "AMD Radeon (ROCm)", 8.0, True
        except Exception:
            pass
    return None, 0.0, False


def _nvidia_smi_memory_mb(query: str) -> float | None:
    """First-GPU nvidia-smi memory.* field in MiB, or None if unavailable."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
            **_spawn_kwargs(),
        )
        raw = (res.stdout or "").strip()
        if res.returncode == 0 and raw:
            lines = [line.strip() for line in raw.splitlines() if line.strip()]
            if lines:
                return float(lines[0])
    except Exception:
        pass
    return None


def detect_free_vram_mb() -> float | None:
    """Free VRAM in MiB on the first NVIDIA GPU, or None if unavailable.

    Used for dynamic Trial headroom (issue #10): effective budget = free-at-start
    minus a safety margin, so Trials on a dirty GPU fail early instead of
    spuriously killing mid-eval.
    """
    return _nvidia_smi_memory_mb("memory.free")


def detect_total_vram_mb() -> float | None:
    """Dedicated GPU VRAM total in MiB (first NVIDIA GPU), or None if unavailable."""
    return _nvidia_smi_memory_mb("memory.total")


def detect_used_total_vram_mb() -> tuple[float, float | None]:
    """``(used_mb, total_mb)`` for GPU 0.

    Raises ``FileNotFoundError`` when nvidia-smi is missing; other subprocess/
    parse errors propagate so callers can retry without treating them as absent.
    """
    res = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        text=True,
        **_spawn_kwargs(),
    )
    parts = [p.strip() for p in (res.strip().splitlines() or [""])[0].split(",")]
    used = float(parts[0] or 0.0)
    total = float(parts[1]) if len(parts) > 1 and parts[1] else None
    return used, total


def detect_pid_gpu_shared_mb(pid: int) -> float | None:
    """WDDM Shared GPU memory (MiB) for a process, or None if unavailable.

    MoE + ``--no-mmap`` can park host tensors in this bucket (PCI-e / CUDA
    pinned) instead of normal RAM; Shared→pagefile freezes the PC even when
    dedicated VRAM is only ~4–5 GB. Uses PDH ``typeperf`` Shared Usage.
    """
    if sys.platform != "win32" or pid <= 0:
        return None
    try:
        res = subprocess.run(
            [
                "typeperf",
                r"\GPU Process Memory(*)\Shared Usage",
                "-sc",
                "1",
            ],
            capture_output=True,
            text=True,
            check=False,
            **_spawn_kwargs(),
        )
        if res.returncode != 0:
            return None
        header: list[str] | None = None
        data: list[str] | None = None
        for line in (res.stdout or "").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("Exiting") or raw.startswith("The command"):
                continue
            cols = [p.strip().strip('"') for p in raw.split(",")]
            if not cols:
                continue
            joined = " ".join(cols).lower()
            if "pdh-csv" in joined or "gpu process memory" in joined:
                header = cols
                continue
            if header is not None and data is None and len(cols) == len(header):
                data = cols
        if not header or not data:
            return None
        needle = f"pid_{int(pid)}_"
        total = 0.0
        matched = False
        for idx, col in enumerate(header):
            if needle not in col:
                continue
            try:
                total += float(data[idx])
                matched = True
            except (ValueError, IndexError):
                continue
        if not matched:
            return None
        return total / (1024.0 * 1024.0)
    except Exception:
        return None


def detect_physical_cores() -> int | None:
    """Physical core count, or None if undetectable (best-effort).

    Windows reads `wmic cpu get NumberOfCores`; macOS reads
    `sysctl -n hw.physicalcpu`; Linux dedupes (physical id, core id) pairs from
    /proc/cpuinfo. Falls back to `os.cpu_count()` (logical count) when a
    platform-specific read fails, so logical-vs-physical is best-effort there.
    """
    system = sys.platform
    if system == "win32":
        try:
            res = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True,
                text=True,
                check=False,
                **_spawn_kwargs(),
            )
            lines = [line.strip() for line in res.stdout.splitlines() if line.strip().isdigit()]
            if lines:
                return int(lines[0])
        except Exception:
            pass
    elif system == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                check=False,
            )
            raw = (res.stdout or "").strip()
            if raw.isdigit():
                return int(raw)
        except Exception:
            pass
    else:
        try:
            pairs = set()
            phys = core = None
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("physical id"):
                        phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif line.strip() == "" and phys is not None and core is not None:
                        pairs.add((phys, core))
                        phys = core = None
            if pairs:
                return len(pairs)
        except Exception:
            pass
    return os.cpu_count()


_SIMD_ALIASES: dict[str, tuple[str, ...]] = {
    "avx512_vnni": ("avx512vnni",),
    "avx512f": ("avx512f",),
    "avx2": ("avx2", "avx20"),
    "avx": ("avx", "avx10"),
    "sse4_2": ("sse42",),
    "fma": ("fma",),
    "f16c": ("f16c",),
    "neon": ("neon", "asimd"),
}


def _normalize_simd(flag: str) -> str:
    return flag.replace(".", "").replace("_", "").lower()


def detect_simd_hints() -> list[str]:
    """Best-effort CPU SIMD flags from stdlib probes. Empty when unavailable.

    Linux reads the `/proc/cpuinfo` `flags` line; macOS reads
    `sysctl machdep.cpu.features` (normalizing the mac-style `AVX1.0`/`AVX2.0`
    names). Windows has no stdlib probe — returns []. Never raises.
    """
    system = sys.platform
    raw_flags: set[str] = set()
    if system == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("flags"):
                        raw_flags = set(line.split(":", 1)[1].split())
                        break
        except Exception:
            return []
    elif system == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features"],
                capture_output=True,
                text=True,
                check=False,
            )
            raw_flags = set((res.stdout or "").split())
        except Exception:
            return []
    else:
        return []

    normalized = {_normalize_simd(f) for f in raw_flags}
    return [
        name for name, aliases in _SIMD_ALIASES.items() if any(a in normalized for a in aliases)
    ]


def has_discrete_nvidia() -> bool:
    return detect_nvidia()[2]


def has_discrete_amd() -> bool:
    return detect_amd()[2]


def has_discrete_gpu() -> bool:
    return has_discrete_nvidia() or has_discrete_amd()


def detect_hardware_capabilities() -> dict[str, Any]:
    """Source-of-truth hardware probe for the Search loop (issue #17).

    Returns has_gpu / physical_cores / ram_mb. Never raises: each probe
    degrades to a safe default on failure. Reuses the existing NVIDIA/AMD/Metal
    probe path and the host RAM probe; adds a stdlib physical-core read.
    """
    _, _, has_cuda = detect_nvidia()
    _, _, has_rocm = detect_amd()
    has_metal, _ = detect_apple_metal()
    return {
        "has_gpu": bool(has_cuda or has_rocm or has_metal),
        "physical_cores": detect_physical_cores(),
        "ram_mb": detect_host_ram_mb(),
    }


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
    return not has_discrete_gpu()


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
    """Rich host facts for check_hardware diagnostics.

    Consumes the shared `detect_hardware_capabilities()` for the overlapping
    fields (ram_mb / has_gpu / physical_cores) so the Search-loop source of
    truth stays the single probe path; keeps its own GPU-name / VRAM / Metal /
    chip probes and adds best-effort SIMD hints.
    """
    caps = detect_hardware_capabilities()
    ram_mb = caps["ram_mb"]
    has_gpu = caps["has_gpu"]
    physical_cores = caps["physical_cores"]
    ram_gb = round(ram_mb / 1024.0, 1) if ram_mb else 0.0
    gpu_name, vram_gb, has_cuda = detect_nvidia()
    amd_name, amd_vram_gb, has_rocm = detect_amd()
    has_metal, chip = detect_apple_metal()

    if has_cuda and gpu_name:
        display_gpu = gpu_name
        active_vram = vram_gb
    elif has_rocm and amd_name:
        display_gpu = f"{amd_name} (AMD)"
        active_vram = amd_vram_gb
    elif has_metal:
        display_gpu = "Apple / macOS (Metal)"
        active_vram = 0.0
    else:
        display_gpu = "Não detectada (CPU)"
        active_vram = 0.0

    memory_class = classify_memory_class(has_cuda=has_cuda, has_rocm=has_rocm, has_metal=has_metal)

    return {
        "ram_gb": ram_gb,
        "ram_mb": ram_mb,
        "physical_cores": physical_cores,
        "logical_cores": os.cpu_count(),
        "has_gpu": has_gpu,
        "vram_gb": active_vram,
        "gpu_name": display_gpu,
        "has_cuda": has_cuda,
        "has_rocm": has_rocm,
        "has_metal": has_metal,
        "memory_class": memory_class,
        "chip": chip,
        "simd_hints": detect_simd_hints(),
        "platform": sys.platform,
        "detection_complete": ram_gb > 0.0,
    }
