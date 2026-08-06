"""Single-load gate (issue #41): at most one full server intent at a time.

Refuses to start a second full llama-server / SGLang server while one is
already live on a harness port. A speculative draft model is loaded inside the
SAME server process (``--spec-draft-model`` on the same command line), so it
never registers as a second server and cannot trip the gate.

Detection reuses the Process Guard pre-flight surface (issue #37): a "full
server" is a process that BOTH matches a target server name AND listens on a
harness port. LM Studio / Ollama / off-port user servers never match.

Escape hatch: explicit allow-multi via ``allow_multi=True`` (model-up
``--allow-multi``) or the ``AUTORESEARCH_ALLOW_MULTI_SERVERS`` env var
(truthy: 1/true/yes/on) — for intentional multi small-model experiments.

    allow_multi = resolve_allow_multi()
    assert_single_load(allow_multi=allow_multi)   # raises SingleLoadError
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from autoresearch.core.process_guard import (
    HARNESS_PORTS,
    TARGET_PROCESS_NAMES,
    listeners_on_ports,
    processes_by_name,
)

ALLOW_MULTI_ENV = "AUTORESEARCH_ALLOW_MULTI_SERVERS"

_TRUTHY = {"1", "true", "yes", "on"}


class SingleLoadError(RuntimeError):
    """A full server is already live and the single-load gate is not bypassed."""


def resolve_allow_multi(allow_multi: bool | None = None) -> bool:
    """Resolve the escape hatch: explicit arg wins, else the env var."""
    if allow_multi is not None:
        return bool(allow_multi)
    return os.environ.get(ALLOW_MULTI_ENV, "").strip().lower() in _TRUTHY


def live_full_server_pids(
    ports: Sequence[int] = HARNESS_PORTS,
    process_names: Sequence[str] = TARGET_PROCESS_NAMES,
) -> list[int]:
    """PIDs of live full servers (name+port intersection, sorted).

    A speculative draft lives inside the same server process and never adds a
    PID, so it cannot be seen as a second full server.
    """
    return sorted(listeners_on_ports(ports) & processes_by_name(process_names))


def assert_single_load(
    allow_multi: bool | None = None,
    ports: Sequence[int] = HARNESS_PORTS,
    process_names: Sequence[str] = TARGET_PROCESS_NAMES,
) -> list[int]:
    """Refuse a second full server while one is live (default).

    Raises SingleLoadError when a live full server exists and allow-multi is
    not set. Returns the live PIDs (empty when the start is allowed).
    """
    if resolve_allow_multi(allow_multi):
        return []
    live = live_full_server_pids(ports, process_names)
    if live:
        raise SingleLoadError(
            f"single-load gate: {len(live)} full server(s) already live "
            f"(pids={live}); stop it or set {ALLOW_MULTI_ENV}=1 / --allow-multi"
        )
    return live


def enforce_single_load(allow_multi: bool | None = None) -> None:
    """Runner-facing gate: refuse a second full server while one is live.

    Fail-open, mirroring the pre-flight orphan sweep (#38): a detection
    failure (e.g. subprocess tooling unavailable or mocked out in tests) logs
    and moves on — the gate only refuses on a positive live-server detection.
    """
    try:
        assert_single_load(allow_multi=allow_multi)
    except SingleLoadError:
        raise
    except Exception as exc:
        print(f"  [single-load] gate skipped: {exc}")
