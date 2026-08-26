"""Permanent RAM circuit breaker — never let a model process push the machine into swap.

Protocol (operator-adopted 2026-08-24 codacus session; made permanent and
always-on 2026-08-25 after a raw llama-cli probe ballooned to a 22.4 GB
working set and thrashed the pagefile): kill the spawned llama process tree
when either

- system free RAM drops below FREE_RAM_FLOOR_MB (default 500 MiB), or
- the watched process's own working set exceeds physical RAM minus
  RAM_WATCHDOG_RESERVE_MB (default 4096 MiB).

Floor history: 2500 MiB came from the codacus memory-envelope incident
(machine already thrashing at that point); the operator set 500 MiB on
2026-08-25. The single-process balloon guard is the RSS cap
(physical - reserve); the free-RAM floor only needs to catch combined
near-death.

Both rules are relative to the machine's physical RAM, so the breaker scales
to any hardware. It is a hard guarantee: no model process launched through
the harness may ever pagefile-thrash the host.

Integration point (permanent):

- ``autoresearch/core/llama_runner.py`` — harness server launches run an
  in-process watchdog thread (the runner process outlives the server).

``scripts/model_up.py`` alias launches are exempt (operator decision
2026-08-26): aliases are hand-tuned trusted recipes, and the detached
watchdog's probe subprocesses popped terminal windows on Windows 11.

Memory reading is cross-platform and dependency-free: ctypes
(GlobalMemoryStatusEx / GetProcessMemoryInfo) on Windows, ``/proc/meminfo``
on Linux, ``psutil`` when available as a richer fallback. If readers are
unavailable the watchdog logs a degradation warning once and never guesses —
it must not kill the wrong process; the preflight (file-size based) still
guards the launch.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable

# Defaults (match the operator protocol; overridable per integration).
DEFAULT_FLOOR_MB = (
    500  # runtime kill: free RAM below this = swap-spill risk (operator-set 2026-08-25)
)
DEFAULT_POLL_S = 1.0
DEFAULT_RESERVE_MB = 4096
DEFAULT_WORKSPACE_MB = 1024
DEFAULT_PREFLIGHT_MARGIN_MB = 512  # launch gate only; watchdog is the real guard

_STILL_ACTIVE = 259  # Windows STILL_ACTIVE exit code


class CircuitBreakerError(RuntimeError):
    """Raised by :func:`preflight_ram` when a launch would endanger the host."""


# --------------------------------------------------------------------------
# Memory readers (ctypes / /proc / psutil fallback)
# --------------------------------------------------------------------------


def _win_memory_status() -> tuple[float | None, float | None]:
    """Return (avail_phys_mb, total_phys_mb) on Windows, else (None, None)."""
    if os.name != "nt":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(ms)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        if not ok:
            return None, None
        return ms.ullAvailPhys / (1024 * 1024), ms.ullTotalPhys / (1024 * 1024)
    except Exception:
        return None, None


def _proc_meminfo() -> tuple[float | None, float | None]:
    """Linux /proc/meminfo: (avail_mb, total_mb) or (None, None)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = dict(line.split(":", 1) for line in fh)

        def _kb(key: str) -> float:
            return float(info.get(key, "0 kB").strip().split()[0]) / 1024.0

        total = _kb("MemTotal")
        avail = _kb("MemAvailable") or _kb("MemFree")
        return avail, total
    except Exception:
        return None, None


def free_ram_mb() -> float | None:
    """Free physical RAM in MiB, or None if undetectable."""
    if os.name == "nt":
        avail, _ = _win_memory_status()
        if avail is not None:
            return avail
    avail, _ = _proc_meminfo()
    if avail is not None:
        return avail
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().available) / (1024 * 1024)
    except Exception:
        return None


def physical_ram_mb() -> float | None:
    """Total physical RAM in MiB, or None if undetectable."""
    if os.name == "nt":
        _, total = _win_memory_status()
        if total is not None:
            return total
    _, total = _proc_meminfo()
    if total is not None:
        return total
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total) / (1024 * 1024)
    except Exception:
        return None


def process_rss_mb(pid: int) -> float | None:
    """Working-set (resident) size of ``pid`` in MiB, or None if unavailable."""
    if pid is None or pid <= 0:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_INFORMATION = 0x0400
            PROCESS_VM_READ = 0x0010
            kernel32 = ctypes.windll.kernel32

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            handle = kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, int(pid)
            )
            if not handle:
                return None
            try:
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                if not kernel32.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    return None
                return counters.WorkingSetSize / (1024 * 1024)
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return None
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return float(psutil.Process(pid).memory_info().rss) / (1024 * 1024)
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                return code.value == _STILL_ACTIVE
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def kill_process_tree(pid: int) -> None:
    """Force-kill ``pid`` and its whole tree. Safe when already dead."""
    if pid is None or pid <= 0:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return
        try:
            os.killpg(os.getpgid(pid), 9)  # type: ignore[attr-defined]
        except (OSError, ProcessLookupError):
            os.kill(pid, 9)
    except Exception:
        pass


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def preflight_ram(
    model_bytes: int,
    margin_mb: float = DEFAULT_PREFLIGHT_MARGIN_MB,
    workspace_mb: float = DEFAULT_WORKSPACE_MB,
) -> float:
    """Refuse a launch when free RAM cannot hold the model + workspace + margin.

    The margin is intentionally small (512 MiB default): the *runtime* watchdog
    (FREE_RAM_FLOOR_MB, default 500) is the hard anti-swap guarantee during the
    run. The preflight only prevents obviously-impossible launches (free RAM
    below the file size) and launch-then-instant-kill loops.

    Returns free MiB on success; raises :class:`CircuitBreakerError` otherwise.
    """
    free = free_ram_mb()
    if free is None:
        # Cannot measure: do not block (worst case the runtime watchdog is
        # also degraded) — but still refuse for absurdly large models.
        return 0.0
    need_mb = model_bytes / (1024 * 1024) + workspace_mb + margin_mb
    if free < need_mb:
        raise CircuitBreakerError(
            f"RAM_PREFLIGHT free={free:.0f}MB < need={need_mb:.0f}MB "
            f"(model={model_bytes / (1024 * 1024):.0f}MB + workspace={workspace_mb:.0f}MB "
            f"+ margin={margin_mb:.0f}MB) — refusing launch to prevent swap spill"
        )
    return free


# --------------------------------------------------------------------------
# Runtime watchdog
# --------------------------------------------------------------------------


def start_ram_watchdog(
    pid: int,
    floor_mb: float = DEFAULT_FLOOR_MB,
    poll_s: float = DEFAULT_POLL_S,
    reserve_mb: float = DEFAULT_RESERVE_MB,
    on_kill: Callable[[str], None] | None = None,
    log: Callable[[str], None] = lambda msg: print(msg, flush=True),
) -> threading.Thread:
    """Start a daemon thread that kills ``pid``'s tree on RAM breach.

    Breach when free RAM < ``floor_mb`` OR the process RSS exceeds
    ``physical - reserve_mb``. Exits when the pid is gone. Returns the thread.
    """
    killed = threading.Event()

    def _loop() -> None:
        warned = False
        while not killed.is_set():
            try:
                if not _pid_alive(pid):
                    return
                free = free_ram_mb()
                phys = physical_ram_mb()
                rss = process_rss_mb(pid)
                reason: str | None = None
                if free is not None and free < floor_mb:
                    reason = f"system free RAM {free:.0f} MiB < floor {floor_mb:.0f} MiB"
                elif rss is not None and phys is not None and rss > phys - reserve_mb:
                    reason = (
                        f"process RSS {rss:.0f} MiB > physical {phys:.0f} MiB "
                        f"- reserve {reserve_mb:.0f} MiB"
                    )
                if reason and not killed.is_set():
                    log(
                        f"[RAM] CIRCUIT BREAKER: {reason} (pid {pid}) — killing "
                        "to prevent pagefile swap spill"
                    )
                    kill_process_tree(pid)
                    killed.set()
                    if on_kill is not None:
                        try:
                            on_kill(reason)
                        except Exception:
                            pass
                    return
                if free is None and not warned:
                    log("[RAM] watchdog: memory readers unavailable — monitoring degraded")
                    warned = True
            except Exception as exc:  # pragma: no cover - defensive
                if not warned:
                    log(f"[RAM] watchdog error: {exc}")
                    warned = True
            time.sleep(max(0.05, poll_s))

    thread = threading.Thread(target=_loop, daemon=True, name="ram-circuit-breaker")
    thread.start()
    return thread


# --------------------------------------------------------------------------
# Detached watchdog CLI (used by scripts/model_up.py)
# --------------------------------------------------------------------------


def _watch_main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="circuit-breaker watch",
        description="Kill the given PID tree if the machine would spill to swap.",
    )
    parser.add_argument("pid", type=int)
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR_MB)
    parser.add_argument("--poll", type=float, default=DEFAULT_POLL_S)
    parser.add_argument("--reserve", type=float, default=DEFAULT_RESERVE_MB)
    parser.add_argument("--log", default=None, help="append kill/exit lines to this file")
    args = parser.parse_args(argv)

    if args.log:

        def log(msg: str) -> None:
            with open(args.log, "a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    else:

        def log(msg: str) -> None:
            print(msg, flush=True)

    thread = start_ram_watchdog(
        args.pid, floor_mb=args.floor, poll_s=args.poll, reserve_mb=args.reserve, log=log
    )
    thread.join()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "watch":
        print(
            "usage: python -m autoresearch.core.circuit_breaker watch <pid> [--floor N] [--poll S] [--reserve N] [--log PATH]"
        )
        return 2
    return _watch_main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
