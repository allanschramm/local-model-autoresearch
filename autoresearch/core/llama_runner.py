"""
Llama Server Runner

Encapsulates the lifecycle of a llama.cpp server process, including:
- ServerIntent parsing & command building
- Hardware locality optimization (MTP, VITRIOL/MoE)
- Port discovery & binding
- Health checking
- VRAM usage sampling
- Safe teardown
"""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LLAMA_SERVER_HELP_CACHE = None
_MODEL_SEARCH_SKIP = frozenset({".cache", "aliases", "huggingface"})

import autoresearch.core.config as config
from autoresearch.core.config import ConfigError, is_dense_model, validate_config
from autoresearch.core.hardware import (
    detect_free_vram_mb,
    detect_pid_gpu_shared_mb,
    detect_total_vram_mb,
    detect_used_total_vram_mb,
)
from autoresearch.core.model_arch import (
    gguf_block_count,
    gguf_is_moe,
    gguf_kv_f16_mb,
    resolve_n_cpu_moe,
)
from autoresearch.core.process_guard import ProcessGuard, cleanup_leftover_processes
from autoresearch.core.single_load import enforce_single_load, resolve_allow_multi


def resolve_model_path(models_dir: Path, ref: str | Path) -> Path:
    """Resolve a model ref under models_dir (flat or nested LM Studio layout).

    Order: absolute → models_dir/ref if present → rglob basename (skip .cache/aliases).
    Missing refs return models_dir/ref for the caller to fail later.
    """
    models_dir = Path(models_dir)
    ref_path = Path(ref)
    if ref_path.is_absolute():
        return ref_path

    direct = models_dir / ref_path
    if direct.exists():
        return direct

    name = ref_path.name
    matches: list[Path] = []
    for path in models_dir.rglob(name):
        if any(part in _MODEL_SEARCH_SKIP for part in path.parts):
            continue
        if path.is_file() or path.is_dir():
            matches.append(path)
    if not matches:
        return direct
    matches.sort(key=lambda p: (len(p.relative_to(models_dir).parts), str(p).lower()))
    return matches[0]


ROOT_DIR = Path(__file__).resolve().parent
LLAMA_CPP_ROOT = Path(os.environ.get("AUTORESEARCH_LLAMA_CPP_ROOT", "./llama.cpp"))
IS_WINDOWS = os.name == "nt"


def _exe(name: str) -> str:
    return f"{name}.exe" if IS_WINDOWS else name


def _parse_bool_env(var_name: str, default: bool = False) -> bool:
    val = os.environ.get(var_name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _is_gpu_working(binary_name: str, probe_flag: str) -> bool:
    tool_path = shutil.which(binary_name)
    if not tool_path:
        return False
    try:
        res = subprocess.run(
            [tool_path, probe_flag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return res.returncode == 0
    except Exception:
        return False


def should_prefer_gpu_build() -> bool:
    if _parse_bool_env("AUTORESEARCH_PREFER_CPU"):
        return False
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return False
    if sys.platform == "darwin":
        return platform.machine() == "arm64"
    return _is_gpu_working("nvidia-smi", "-L") or _is_gpu_working("rocm-smi", "-i")


def _build_dir_candidates(root: Path, build_dir: str, exe: str) -> tuple[Path, ...]:
    base = root / build_dir / "bin"
    return (base / exe, base / "Release" / exe, base / "Debug" / exe)


def _candidate_binary(root: Path, name: str) -> tuple[Path, ...]:
    exe = _exe(name)
    cuda_paths = _build_dir_candidates(root, "build-cuda", exe)
    rocm_paths = _build_dir_candidates(root, "build-rocm", exe)
    cpu_paths = _build_dir_candidates(root, "build-cpu", exe)
    generic_paths = _build_dir_candidates(root, "build", exe)

    if should_prefer_gpu_build():
        return cuda_paths + rocm_paths + cpu_paths + generic_paths
    return cpu_paths + generic_paths + cuda_paths + rocm_paths


def _binary_candidates(name: str) -> tuple[Path, ...]:
    roots = (
        LLAMA_CPP_ROOT,
        Path.cwd() / "llama.cpp",
        ROOT_DIR.parent.parent / "llama.cpp",
        ROOT_DIR.parent.parent.parent / "llama.cpp",
    )
    seen: list[Path] = []
    for root in roots:
        for candidate in _candidate_binary(root, name):
            if candidate not in seen:
                seen.append(candidate)
    on_path = shutil.which(_exe(name)) or shutil.which(name)
    if on_path:
        seen.append(Path(on_path))
    return tuple(seen)


LLAMA_SERVER_CANDIDATES = _binary_candidates("llama-server")


@dataclass(frozen=True)
class ServerIntent:
    """A pure data object describing high-level benchmark intent."""

    model_path: Path
    ctx_size: int
    kv_cache: str
    flash_attn: str
    port: int = 18080
    batch_size: int = 512
    ubatch_size: int = 128
    threads: int = 8
    parallel: int = 1
    ngl: int = 999
    numa: str | None = None
    kv_cache_k: str | None = None
    kv_cache_v: str | None = None
    threads_batch: int | None = None
    spec_draft_n_max: int = 1
    no_mmap: bool = False
    mlock: bool = False
    jinja: bool = False
    reasoning_budget: int | None = None
    reasoning_budget_message: str | None = None
    reasoning: str | None = None
    cont_batching: bool = False
    host: str = "127.0.0.1"
    spec_type: str | None = None
    spec_draft_model: str | None = None
    n_cpu_moe: int | None = None
    n_cpu_moe_auto: bool = False

    @classmethod
    def from_config(cls, cfg: dict, models_dir: Path, **overrides) -> tuple["ServerIntent", dict]:
        """Build ServerIntent from config dict. Caller converts non-dict to dict first.

        Returns (intent, norm_dict) where norm_dict holds all config fields
        (server + non-server) for callers that need remaining params.
        """
        merged = dict(cfg)
        merged.update({k: v for k, v in overrides.items() if v is not None})
        merged = validate_config(merged)
        norm = {
            str(k).lower(): v
            for k, v in merged.items()
            if v is not None and isinstance(k, (str, bytes))
        }

        model_fn = norm.get("model", "g4-opt-it-Q4_K_M.gguf")
        kv_cache = norm.get("kv_cache") or norm.get("kv") or "q4_0"
        k_val = norm.get("kv_cache_k") or norm.get("kv_k") or kv_cache
        v_val = norm.get("kv_cache_v") or norm.get("kv_v") or kv_cache
        draft_ref = (
            norm.get("spec_draft_model")
            or norm.get("draft_model")
            or norm.get("spec_draft_model_file")
        )
        draft_path = str(resolve_model_path(models_dir, draft_ref)) if draft_ref else None
        model_path = resolve_model_path(models_dir, model_fn)
        # Baseline None is dropped from norm; read raw merged for explicit 0 vs auto.
        raw_n_cpu_moe = merged.get("N_CPU_MOE", merged.get("n_cpu_moe"))
        try:
            resolved_n_cpu_moe, n_cpu_moe_auto = resolve_n_cpu_moe(model_path, raw_n_cpu_moe)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if n_cpu_moe_auto:
            print(
                f"  [VITRIOL] auto n-cpu-moe={resolved_n_cpu_moe} "
                f"from GGUF block_count for {model_path.name}"
            )

        intent = cls(
            model_path=model_path,
            ctx_size=norm.get("ctx_size", 131072),
            kv_cache=kv_cache,
            flash_attn=norm.get("flash_attn", "on"),
            port=norm.get("port", 18080),
            host=norm.get("host", "127.0.0.1"),
            ngl=norm.get("n_gpu_layers", norm.get("ngl", 999)),
            numa=norm.get("numa"),
            batch_size=norm.get("batch_size", 512),
            ubatch_size=norm.get("ubatch_size", 128),
            threads=norm.get("threads", 12),
            parallel=norm.get("parallel", 1),
            kv_cache_k=k_val,
            kv_cache_v=v_val,
            threads_batch=norm.get("threads_batch"),
            spec_draft_n_max=norm.get("spec_draft_n_max", 1),
            no_mmap=norm.get("no_mmap", False),
            mlock=norm.get("mlock", False),
            jinja=norm.get("jinja", False),
            reasoning_budget=norm.get("reasoning_budget"),
            reasoning_budget_message=norm.get("reasoning_budget_message"),
            reasoning=norm.get("reasoning"),
            cont_batching=norm.get("cont_batching", False),
            spec_type=norm.get("spec_type"),
            spec_draft_model=draft_path,
            n_cpu_moe=resolved_n_cpu_moe,
            n_cpu_moe_auto=n_cpu_moe_auto,
        )

        return intent, norm


# VRAM Estimation Constants (calibrated for f16 KV cache and typical systems)
VRAM_KB_PER_TOKEN_F16 = 80.0
"""Calibrated memory consumption per context token at f16 precision (in kilobytes)."""

VRAM_OVERHEAD_MB = 300.0
"""Typical baseline VRAM overhead for CUDA runtime and system operations (in megabytes)."""

# Residual risk: estimator can under-read peak. Measured dense 65k coding overshoot:
# est 7584 MB → peak 7931 MB (qwen3.5-9b, 2026-07-27). Runtime VRAM sampler is the
# final kill guard; keep the preflight margin conservative for new model arches.

VRAM_SPECULATIVE_BASE_MB = 512.0
VRAM_SPECULATIVE_PER_DRAFT_TOKEN_MB = 256.0
"""Conservative speculative-decoding runtime/workspace allowance."""

VRAM_DEFAULT_QUANT_FACTOR = 0.3
"""Fallback quantization multiplier for unknown/default KV cache types."""

VRAM_QUANT_FACTORS = {
    "f16": 1.0,
    "f32": 1.0,
    "q8": 0.55,
    "q5": 0.38,
    "q4": 0.28,
    "turbo4": 0.18,
    "turbo3": 0.14,
    "turbo2": 0.10,
}
"""KV cache quantization type memory usage scaling factors relative to f16."""

# VITRIOL preflight: MoE expert tensors dominate file size; --n-cpu-moe keeps them on CPU.
VRAM_MOE_NON_EXPERT_FRAC = 0.28
VRAM_MOE_OFFLOAD_LAYER_REF = 32.0

DEFAULT_VRAM_LIMIT_MB = 7900.0
# Safety margin subtracted from free VRAM at Trial start (issue #10).
DEFAULT_VRAM_HEADROOM_MB = 512.0
# Keep this much dedicated VRAM free so WDDM CUDA Sysmem Fallback never arms.
DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB = 512.0
# Task Manager "Shared GPU" kill — WDDM/PCI-e host maps (MoE+NO_MMAP), not
# dedicated-full CUDA Sysmem Fallback. Dedicated can stay ~4–5 GB while Shared
# climbs to 10+ GB and pagefile/SSD freezes the PC. Absolute ceil; do not wait
# for dedicated to fill.
DEFAULT_SHARED_VRAM_LIMIT_MB = 2048.0


def dedicated_vram_kill_ceil(
    limit_mb: float,
    total_mb: float | None,
    keepout_mb: float = DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB,
) -> float:
    """Kill ceiling: ``min(limit, physical − keepout)`` when total is known."""
    ceil = float(limit_mb)
    if total_mb is not None and total_mb > 0:
        return min(ceil, max(1024.0, float(total_mb) - keepout_mb))
    return ceil


def resolve_shared_vram_limit_mb(limit: float | int | None = None) -> float:
    """Resolve Shared-GPU kill ceiling (env AUTORESEARCH_SHARED_VRAM_LIMIT_MB)."""
    if limit is not None:
        return float(limit)
    env = os.environ.get("AUTORESEARCH_SHARED_VRAM_LIMIT_MB")
    if env:
        return float(env)
    return float(config.DEFAULTS.get("SHARED_VRAM_LIMIT_MB", DEFAULT_SHARED_VRAM_LIMIT_MB))


def resolve_vram_limit_mb(limit: float | int | None = None) -> float:
    """Resolve VRAM budget: explicit arg > env AUTORESEARCH_VRAM_LIMIT_MB > config default.

    Hard clamp: never above ``physical − keepout`` so Trials stay below the
    WDDM CUDA Sysmem Fallback zone (Shared GPU / pagefile thrash).
    """
    if limit is not None:
        configured = float(limit)
    else:
        env = os.environ.get("AUTORESEARCH_VRAM_LIMIT_MB")
        if env:
            configured = float(env)
        else:
            configured = float(config.DEFAULTS.get("VRAM_LIMIT_MB", DEFAULT_VRAM_LIMIT_MB))
    total = detect_total_vram_mb()
    if total is None or total <= 0:
        return configured
    keepout = float(
        config.DEFAULTS.get("PHYSICAL_VRAM_KEEPOUT_MB", DEFAULT_PHYSICAL_VRAM_KEEPOUT_MB)
    )
    env_keep = os.environ.get("AUTORESEARCH_PHYSICAL_VRAM_KEEPOUT_MB")
    if env_keep:
        keepout = float(env_keep)
    safe_ceil = dedicated_vram_kill_ceil(configured, total, keepout)
    # When total known, kill_ceil always applies keepout — clamp only if over.
    if configured > safe_ceil:
        print(
            f"  [vram] clamp VRAM_LIMIT_MB {configured:.0f} -> {safe_ceil:.0f}MB "
            f"(physical {total:.0f} − keepout {keepout:.0f}; stay out of Shared spill zone)",
            flush=True,
        )
        return float(safe_ceil)
    return configured


def resolve_vram_headroom_mb(headroom_mb: float | int | None = None) -> float:
    """Resolve safety margin: explicit arg > env AUTORESEARCH_VRAM_HEADROOM_MB > config > default."""
    if headroom_mb is not None:
        return float(headroom_mb)
    env = os.environ.get("AUTORESEARCH_VRAM_HEADROOM_MB")
    if env:
        return float(env)
    val = config.DEFAULTS.get("VRAM_HEADROOM_MB")
    if val is not None:
        return float(val)
    return float(DEFAULT_VRAM_HEADROOM_MB)


def skip_free_vram_clamp() -> bool:
    """Operator escape: AUTORESEARCH_SKIP_FREE_CLAMP=1 skips dense free-at-start clamp.

    Default remains free−headroom (issue #10). Opt-in only — runtime VRAM monitoring
    stays the kill guard. Use when WDDM desktop reservation would false-reject a Trial
    known to fit physical VRAM (same spirit as MoE n_cpu_moe>0 configured-only budget).
    """
    raw = (os.environ.get("AUTORESEARCH_SKIP_FREE_CLAMP") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def effective_vram_limit_mb(
    configured_mb: float,
    free_vram_mb: float | None = None,
    headroom_mb: float | int | None = None,
) -> float:
    """Effective Trial budget: min(configured, free-at-start - headroom).

    Free VRAM is measured at Trial start; the headroom absorbs measurement noise
    and concurrent-process drift so dirty-GPU Trials fail early instead of
    spuriously. Unknown free -> configured unchanged.
    """
    if free_vram_mb is None or free_vram_mb <= 0:
        return float(configured_mb)
    headroom = resolve_vram_headroom_mb(headroom_mb)
    return min(float(configured_mb), max(0.0, float(free_vram_mb) - headroom))


def estimate_vram_mb(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    base_kv_cache: str = "q4_0",
    draft_path: Path | str | None = None,
    n_cpu_moe: int | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
) -> float:
    try:
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
    except Exception:
        model_size_mb = 4000.0

    if n_cpu_moe is not None and int(n_cpu_moe) > 0:
        layer_ref = VRAM_MOE_OFFLOAD_LAYER_REF
        try:
            if model_path.is_file() and gguf_is_moe(model_path):
                layer_ref = float(gguf_block_count(model_path))
        except Exception:
            pass
        offload = min(1.0, float(n_cpu_moe) / layer_ref)
        expert_frac = 1.0 - VRAM_MOE_NON_EXPERT_FRAC
        model_size_mb = model_size_mb * (VRAM_MOE_NON_EXPERT_FRAC + expert_frac * (1.0 - offload))

    draft_mb = 0.0
    if draft_path:
        try:
            draft_mb = Path(draft_path).stat().st_size / (1024 * 1024)
        except Exception:
            draft_mb = 0.0

    try:
        c_size = int(ctx_size)
    except Exception:
        c_size = 16384

    kv_est_mb = _kv_est_mb(
        c_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        base_kv_cache=base_kv_cache,
        model_path=model_path,
    )

    spec_enabled = bool(spec_type and spec_type.lower() != "none" and spec_draft_n_max > 0)
    # MoE expert-CPU offload + external draft file: charge draft weights only.
    # Flat workspace (512 + 256*n) false-rejects DFlash (measured ~4 GB peaks on 8 GB).
    # Embedded MTP / dense targets keep the conservative workspace allowance.
    moe_external_draft = (
        n_cpu_moe is not None and int(n_cpu_moe) > 0 and bool(draft_path) and draft_mb > 0
    )
    spec_workspace_mb = (
        0.0
        if (not spec_enabled) or moe_external_draft
        else VRAM_SPECULATIVE_BASE_MB + VRAM_SPECULATIVE_PER_DRAFT_TOKEN_MB * spec_draft_n_max
    )

    # Baseline system/CUDA overhead (+ draft weights and speculative workspace).
    return model_size_mb + draft_mb + kv_est_mb + VRAM_OVERHEAD_MB + spec_workspace_mb


def preflight_vram(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    draft_path: Path | str | None = None,
    vram_limit_mb: float | None = None,
    n_cpu_moe: int | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
) -> tuple[bool, float, str]:
    """Return (ok, estimate_mb, reason). reason non-empty when rejected."""
    limit = resolve_vram_limit_mb(vram_limit_mb)
    est = estimate_vram_mb(
        model_path,
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        draft_path=draft_path,
        n_cpu_moe=n_cpu_moe,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
    )
    if est > limit:
        return False, est, f"VRAM_PREFLIGHT est={est:.0f}MB > limit={limit:.0f}MB"
    return True, est, ""


def preflight_vram_effective(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    draft_path: Path | str | None = None,
    vram_limit_mb: float | None = None,
    n_cpu_moe: int | None = None,
    spec_type: str | None = None,
    spec_draft_n_max: int = 0,
    headroom_mb: float | int | None = None,
    free_vram_mb: float | None = None,
) -> tuple[bool, float, str]:
    """Headroom wrapper around preflight_vram (issue #10).

    Effective budget = min(configured, free VRAM at Trial start - headroom).
    Exception: MoE with `n_cpu_moe > 0` uses configured only — OS-reserved VRAM
    otherwise false-rejects expert-CPU offload (measured peaks far below free).
    Runtime VRAM monitoring remains the OOM kill guard.
    The reject reason records both configured and effective budgets.
    """
    configured = resolve_vram_limit_mb(vram_limit_mb)
    moe_offload = n_cpu_moe is not None and int(n_cpu_moe) > 0
    skip_clamp = skip_free_vram_clamp()
    if moe_offload or skip_clamp:
        effective = float(configured)
        if skip_clamp and not moe_offload:
            print(
                f"  [vram-preflight] SKIP_FREE_CLAMP=1 — using configured={configured:.0f}MB "
                "(runtime monitor remains kill guard)",
                flush=True,
            )
    else:
        if free_vram_mb is None:
            free_vram_mb = detect_free_vram_mb()
        effective = effective_vram_limit_mb(configured, free_vram_mb, headroom_mb)
    ok, est, reason = preflight_vram(
        model_path,
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        draft_path=draft_path,
        vram_limit_mb=effective,
        n_cpu_moe=n_cpu_moe,
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max,
    )
    if ok or effective == configured:
        return ok, est, reason
    headroom = resolve_vram_headroom_mb(headroom_mb)
    return (
        False,
        est,
        f"VRAM_PREFLIGHT est={est:.0f}MB > effective={effective:.0f}MB "
        f"(configured={configured:.0f}MB free={free_vram_mb:.0f}MB headroom={headroom:.0f}MB)",
    )


def resolve_spec_estimate_args(
    model_name: str,
    spec_type: str | None,
    spec_draft_n_max: int,
    draft_path: Path | str | None,
) -> tuple[str | None, bool, Path | str | None]:
    """Resolve effective speculative args for VRAM estimation (MTP-aware).

    Mirrors ServerIntent MTP inference: embedded-MTP GGUFs with no explicit
    SPEC_TYPE estimate with speculative workspace; external drafts only count
    when speculation is enabled.
    """
    if spec_type is None and "MTP" in model_name.upper():
        spec_type = "mtp"
    spec_enabled = bool(spec_type and spec_type.lower() != "none" and spec_draft_n_max > 0)
    return spec_type, spec_enabled, draft_path if spec_enabled else None


def preflight_vram_for_intent(
    intent: "ServerIntent",
    vram_limit_mb: float | None = None,
    headroom_mb: float | int | None = None,
) -> tuple[bool, float, str]:
    spec_type, _, draft_path = resolve_spec_estimate_args(
        intent.model_path.name,
        intent.spec_type,
        intent.spec_draft_n_max,
        intent.spec_draft_model,
    )
    return preflight_vram_effective(
        intent.model_path,
        intent.ctx_size,
        kv_cache_k=intent.kv_cache_k or intent.kv_cache,
        kv_cache_v=intent.kv_cache_v or intent.kv_cache,
        draft_path=draft_path,
        vram_limit_mb=vram_limit_mb,
        n_cpu_moe=intent.n_cpu_moe,
        spec_type=spec_type,
        spec_draft_n_max=intent.spec_draft_n_max,
        headroom_mb=headroom_mb,
    )


def _kv_est_mb(
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    base_kv_cache: str = "q4_0",
    model_path: Path | None = None,
) -> float:
    """KV cache estimate in MiB.

    Prefers GGUF-derived total f16 MiB (`gguf_kv_f16_mb`) — sliding-window
    arches (gemma4) only charge `min(ctx, window)` with SWA dims. Falls back to
    flat VRAM_KB_PER_TOKEN_F16. head_count_kv=0 (recurrent) → 0.
    """
    try:
        c_size = int(ctx_size)
    except Exception:
        c_size = 16384
    base_kv = base_kv_cache if base_kv_cache is not None else "q4_0"
    k_type = kv_cache_k if kv_cache_k is not None else base_kv
    v_type = kv_cache_v if kv_cache_v is not None else base_kv

    def get_quant_factor(q_type: Any) -> float:
        if q_type is None or not isinstance(q_type, str):
            return VRAM_DEFAULT_QUANT_FACTOR
        q = q_type.lower()
        for key, factor in VRAM_QUANT_FACTORS.items():
            if key in q:
                return factor
        return VRAM_DEFAULT_QUANT_FACTOR

    kf = get_quant_factor(k_type)
    vf = get_quant_factor(v_type)

    kv_base_mb = c_size * VRAM_KB_PER_TOKEN_F16 / 1024.0
    if model_path is not None:
        try:
            swa_mb = gguf_kv_f16_mb(model_path, c_size)
            if swa_mb is not None:
                kv_base_mb = float(swa_mb)
        except Exception:
            pass
    return (kv_base_mb / 2.0) * kf + (kv_base_mb / 2.0) * vf


def estimate_host_memory_mb(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    base_kv_cache: str = "q4_0",
    draft_path: Path | str | None = None,
) -> float:
    """Full GGUF + draft + KV + overhead. Never shrinks for MoE CPU offload."""
    try:
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
    except Exception:
        model_size_mb = 4000.0

    draft_mb = 0.0
    if draft_path:
        try:
            draft_mb = Path(draft_path).stat().st_size / (1024 * 1024)
        except Exception:
            draft_mb = 0.0

    kv_est_mb = _kv_est_mb(
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        base_kv_cache=base_kv_cache,
        model_path=model_path,
    )
    return model_size_mb + draft_mb + kv_est_mb + VRAM_OVERHEAD_MB


def preflight_host_memory(
    model_path: Path,
    ctx_size: int,
    kv_cache_k: str | None = None,
    kv_cache_v: str | None = None,
    draft_path: Path | str | None = None,
    headroom_mb: float | int | None = None,
    ram_mb: float | None = None,
    unified: bool | None = None,
) -> tuple[bool, float, float, str]:
    """Return (ok, estimate_mb, budget_mb, reason).

    Fail closed on unified hosts when RAM cannot be detected.
    On discrete NVIDIA with unknown RAM: warn via empty reason and pass (VRAM gate remains).
    """
    from autoresearch.core.hardware import (
        MEMORY_CLASS_DISCRETE,
        MEMORY_CLASS_UNIFIED,
        detect_host_ram_mb,
        host_memory_budget_mb,
        is_unified_memory_host,
        resolve_host_headroom_mb,
    )

    if unified is None:
        unified = is_unified_memory_host()
    mem_class = MEMORY_CLASS_UNIFIED if unified else MEMORY_CLASS_DISCRETE

    if ram_mb is None:
        ram_mb = detect_host_ram_mb()

    est = estimate_host_memory_mb(
        model_path,
        ctx_size,
        kv_cache_k=kv_cache_k,
        kv_cache_v=kv_cache_v,
        draft_path=draft_path,
    )

    if ram_mb is None or ram_mb <= 0:
        if unified:
            return (
                False,
                est,
                0.0,
                f"HOST_MEMORY_PREFLIGHT ram_unknown class={mem_class} est={est:.0f}MB",
            )
        return True, est, 0.0, ""

    budget = host_memory_budget_mb(ram_mb, unified=unified, headroom_mb=headroom_mb)
    if budget is None:
        if unified:
            return (
                False,
                est,
                0.0,
                f"HOST_MEMORY_PREFLIGHT ram_unknown class={mem_class} est={est:.0f}MB",
            )
        return True, est, 0.0, ""

    headroom = resolve_host_headroom_mb(ram_mb, unified=unified, override_mb=headroom_mb)
    if est > budget:
        return (
            False,
            est,
            budget,
            (
                f"HOST_MEMORY_PREFLIGHT est={est:.0f}MB > budget={budget:.0f}MB "
                f"(ram={ram_mb:.0f} headroom={headroom:.0f} class={mem_class})"
            ),
        )
    return True, est, budget, ""


def preflight_host_memory_for_intent(
    intent: "ServerIntent",
    headroom_mb: float | int | None = None,
) -> tuple[bool, float, float, str]:
    if headroom_mb is None:
        headroom_mb = config.DEFAULTS.get("HOST_MEMORY_HEADROOM_MB")
    return preflight_host_memory(
        intent.model_path,
        intent.ctx_size,
        kv_cache_k=intent.kv_cache_k or intent.kv_cache,
        kv_cache_v=intent.kv_cache_v or intent.kv_cache,
        draft_path=intent.spec_draft_model,
        headroom_mb=headroom_mb,
    )


LLAMA_BENCH_CANDIDATES = _binary_candidates("llama-bench")


def resolve_llama_server() -> Path:
    for candidate in LLAMA_SERVER_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-server not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_SERVER_CANDIDATES)
    )


def engine_version_tag(server: Path) -> str:
    """Engine identity for Trial evidence: `engine@tag` for versioned fork
    releases (`llama.cpp-releases/<engine>/<tag>/`), `""` for the stock
    submodule.

    AGENTS.md runtime policy: alternate engines land as versioned prebuilt
    releases under `llama.cpp-releases/<engine>/<tag>/`, selected via
    `AUTORESEARCH_LLAMA_CPP_ROOT`. Trials must preserve engine/tag so rows
    from different forks stay distinguishable.
    """
    parts = server.parts
    for i, part in enumerate(parts):
        if part == "llama.cpp-releases" and i + 2 < len(parts):
            return f"{parts[i + 1]}@{parts[i + 2]}"
    return ""


def resolve_llama_bench() -> Path:
    for candidate in LLAMA_BENCH_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-bench not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_BENCH_CANDIDATES)
    )


LLAMA_CLI_CANDIDATES = _binary_candidates("llama-cli")


def resolve_llama_cli() -> Path:
    for candidate in LLAMA_CLI_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-cli not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_CLI_CANDIDATES)
    )


LLAMA_PERPLEXITY_CANDIDATES = _binary_candidates("llama-perplexity")


def resolve_llama_perplexity() -> Path:
    for candidate in LLAMA_PERPLEXITY_CANDIDATES:
        if candidate.exists():
            return candidate.absolute()
    raise FileNotFoundError(
        "llama-perplexity not found. Expected one of: "
        + ", ".join(str(path) for path in LLAMA_PERPLEXITY_CANDIDATES)
    )


def candidate_ports(preferred: int) -> list[int]:
    return list(dict.fromkeys((preferred, preferred + 1, preferred + 2, 18080, 28080)))


def sweep_leftover_processes() -> None:
    """Best-effort pre-flight orphan sweep (ADR 0010 decision 2).

    Kills leftover harness processes holding a harness port before a server
    binds. Never blocks startup: a sweep failure (e.g. subprocess tooling
    unavailable or mocked out) logs and moves on — the port bind would surface
    EADDRINUSE on its own.
    """
    try:
        killed = cleanup_leftover_processes()
    except Exception as exc:
        print(f"  [process-guard] pre-flight orphan sweep skipped: {exc}")
        return
    if killed:
        print(f"  [process-guard] pre-flight killed leftover harness procs: {sorted(killed)}")


class LlamaServerRunner:
    def __init__(
        self,
        intent: ServerIntent,
        log_path: Path | None = None,
        vram_limit_mb: float | None = None,
    ):
        self.intent = intent
        self.log_path = log_path
        self.vram_limit_mb = resolve_vram_limit_mb(vram_limit_mb)
        self.shared_vram_limit_mb = resolve_shared_vram_limit_mb()

        self.port: int | None = None
        self.peak_vram_mb: float = 0.0
        self.peak_shared_mb: float = 0.0
        self.vram_killed: bool = False

        self._server_proc: subprocess.Popen[str] | None = None
        self._server_log: Any = None
        self._stop_event = threading.Event()
        self._vram_thread: threading.Thread | None = None
        self._guard: ProcessGuard | None = None

        self.llama_server = resolve_llama_server()

    def _build_cmd(self, target_port: int) -> list[str]:
        cache_type_k = (
            self.intent.kv_cache_k if self.intent.kv_cache_k is not None else self.intent.kv_cache
        )
        cache_type_v = (
            self.intent.kv_cache_v if self.intent.kv_cache_v is not None else self.intent.kv_cache
        )

        cmd = [
            str(self.llama_server),
            "--model",
            str(self.intent.model_path),
            "--host",
            str(self.intent.host),
            "--port",
            str(target_port),
            "--ctx-size",
            str(self.intent.ctx_size),
            "--batch-size",
            str(self.intent.batch_size),
            "--ubatch-size",
            str(self.intent.ubatch_size),
            "--threads",
            str(self.intent.threads),
            "--parallel",
            str(self.intent.parallel),
            "--n-gpu-layers",
            str(self.intent.ngl),
        ]

        if self.intent.numa:
            cmd += ["--numa", self.intent.numa]

        cmd += [
            "--cache-type-k",
            cache_type_k,
            "--cache-type-v",
            cache_type_v,
            "--flash-attn",
            self.intent.flash_attn,
        ]

        if self.intent.threads_batch is not None:
            cmd += ["--threads-batch", str(self.intent.threads_batch)]

        if self.intent.no_mmap:
            cmd += ["--no-mmap"]
        if self.intent.mlock:
            cmd += ["--mlock"]
        if self.intent.jinja:
            cmd += ["--jinja"]
        if self.intent.reasoning_budget is not None:
            cmd += ["--reasoning-budget", str(self.intent.reasoning_budget)]
        if self.intent.reasoning_budget_message is not None:
            cmd += ["--reasoning-budget-message", self.intent.reasoning_budget_message]
        if self.intent.reasoning is not None:
            cmd += ["--reasoning", str(self.intent.reasoning)]
        if self.intent.cont_batching:
            cmd += ["--cont-batching"]

        # MTP/Speculative Optimization: Detect MTP models and enable speculative decoding.
        spec_type_val = self.intent.spec_type
        if (
            spec_type_val is None
            and "MTP" in self.intent.model_path.name.upper()
            and self.intent.spec_draft_n_max > 0
        ):
            global _LLAMA_SERVER_HELP_CACHE
            if _LLAMA_SERVER_HELP_CACHE is None:
                try:
                    _LLAMA_SERVER_HELP_CACHE = subprocess.check_output(
                        [str(self.llama_server), "--help"], stderr=subprocess.STDOUT, text=True
                    )
                except Exception:
                    _LLAMA_SERVER_HELP_CACHE = "mtp"
            if "mtp" in _LLAMA_SERVER_HELP_CACHE:
                spec_type_val = "mtp"
            else:
                spec_type_val = "draft-mtp"
            print(
                f"  [MTP] Multi-Token Prediction detected for {self.intent.model_path.name}. Auto-selected spec-type: {spec_type_val}"
            )

        if (
            spec_type_val is not None
            and spec_type_val.lower() != "none"
            and self.intent.spec_draft_n_max > 0
        ):
            cmd += [
                "--spec-type",
                spec_type_val,
                "--spec-draft-n-max",
                str(self.intent.spec_draft_n_max),
                "--spec-draft-type-k",
                cache_type_k,
                "--spec-draft-type-v",
                cache_type_v,
            ]
            if self.intent.spec_draft_model:
                draft_path = Path(self.intent.spec_draft_model)
                if not draft_path.is_absolute():
                    draft_path = self.intent.model_path.parent / draft_path
                cmd += ["--spec-draft-model", str(draft_path)]

        # VITRIOL: MoE expert offload only (resolved in from_config). Dense: no flag.
        if self.intent.n_cpu_moe is not None:
            print(
                f"  [VITRIOL] MoE Expert Streaming: --n-cpu-moe {self.intent.n_cpu_moe} "
                f"for {self.intent.model_path.name}."
            )
            cmd += ["--n-cpu-moe", str(self.intent.n_cpu_moe)]

        return cmd

    def __enter__(self):
        self._start_vram_sampler()

        server_env = os.environ.copy()
        llama_lib_dir = str(self.llama_server.parent)
        lib_path_var = "PATH" if IS_WINDOWS else "LD_LIBRARY_PATH"
        existing = server_env.get(lib_path_var, "")
        server_env[lib_path_var] = (
            f"{llama_lib_dir}{os.pathsep}{existing}" if existing else llama_lib_dir
        )
        # Avoid CUDA pinned host maps that WDDM counts as Shared GPU (pagefile freeze).
        server_env["GGML_CUDA_NO_PINNED"] = "1"

        # Single-load gate (#41): refuse a second full server while one is
        # live. The pre-flight orphan sweep would kill a live sibling on a
        # harness port, so it only runs when the gate passes and allow-multi
        # is off.
        allow_multi = resolve_allow_multi()
        enforce_single_load(allow_multi=allow_multi)
        if not allow_multi:
            sweep_leftover_processes()
        self._guard = ProcessGuard()

        startup_tail: list[str] = []
        for port in candidate_ports(self.intent.port):
            cmd = self._build_cmd(port)
            print(f"Starting server: {' '.join(cmd)}")

            if self.log_path:
                self._server_log = open(self.log_path, "w+", encoding="utf-8")
            else:
                self._server_log = tempfile.NamedTemporaryFile(
                    mode="w+",
                    encoding="utf-8",
                    prefix="autoresearch-llama-server-",
                    suffix=".log",
                    delete=True,
                )

            self._server_proc = self._guard.spawn(
                cmd,
                stdout=self._server_log,
                stderr=subprocess.STDOUT,
                env=server_env,
                text=True,
            )

            self.port = port
            if self._wait_for_server(port):
                return self

            if self.vram_killed:
                self._cleanup_all()
                raise RuntimeError("VRAM_LIMIT_EXCEEDED")

            # If wait failed, grab the tail before cleaning up
            self._server_log.flush()
            if hasattr(self._server_log, "name"):
                log_content = Path(self._server_log.name).read_text(
                    encoding="utf-8", errors="replace"
                )
                startup_tail = log_content.splitlines()[-50:]

            self._cleanup_process()
            print(f"Failed to start on port {port}, trying next...")

        self._cleanup_all()
        print("FAIL: Server crashed during startup.")
        if startup_tail:
            print("Tail of startup log:")
            print("\n".join(startup_tail))
        raise RuntimeError("Failed to start llama-server on any candidate port.")

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self._cleanup_all()

    def _wait_for_server(self, port: int) -> bool:
        delay = 0.05
        while True:
            if self._server_proc is None or self._server_proc.poll() is not None:
                return False
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                with urllib.request.urlopen(req) as response:
                    if response.status == 200:
                        return True
            except Exception:
                time.sleep(delay)
                delay = min(delay * 2, 0.4)

    @property
    def is_cpu_mode(self) -> bool:
        return (
            "build-cpu" in str(self.llama_server).lower()
            or (self.intent is not None and self.intent.ngl == 0)
            or not should_prefer_gpu_build()
        )

    def _start_vram_sampler(self) -> None:
        if self.is_cpu_mode:
            return

        import ctypes

        # Load NVML library using ctypes
        nvml = None
        device = None

        class nvmlMemory_t(ctypes.Structure):
            _fields_ = [
                ("total", ctypes.c_uint64),
                ("free", ctypes.c_uint64),
                ("used", ctypes.c_uint64),
            ]

        try:
            nvml_name = "nvml.dll" if IS_WINDOWS else "libnvidia-ml.so.1"
            nvml = ctypes.CDLL(nvml_name)
            nvml.nvmlInit_v2()
            device = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(device))
            print("  [VRAM] NVML initialized successfully. High-frequency 20ms sampling enabled.")
        except Exception:
            nvml = None
            print(
                "  [VRAM] NVML initialization failed. Falling back to subprocess nvidia-smi (200ms)."
            )

        dense = is_dense_model(self.intent.model_path)
        limit = self.vram_limit_mb
        shared_limit = self.shared_vram_limit_mb
        shared_tick = 0

        def _maybe_kill(current: float, total_mb: float | None = None) -> None:
            if current > self.peak_vram_mb:
                self.peak_vram_mb = current
            # Kill ANY arch (dense or MoE) over limit — MoE used to skip and WDDM
            # spilled into shared GPU memory / pagefile, freezing the host.
            ceil = dedicated_vram_kill_ceil(limit, total_mb)
            if current > ceil and not self.vram_killed:
                self.vram_killed = True
                kind = "dense" if dense else "moe"
                print(
                    f"  [VRAM] LIMIT EXCEEDED used={current:.0f}MB > limit={ceil:.0f}MB "
                    f"({kind}={self.intent.model_path.name}) — killing "
                    "(no Windows shared-GPU / pagefile spill)",
                    flush=True,
                )
                self._cleanup_process()
                self._stop_event.set()

        def _maybe_kill_shared() -> None:
            """Kill absolute Shared GPU overshoot (dedicated can stay ~4–5 GB)."""
            if self.vram_killed or self._server_proc is None or self._server_proc.pid is None:
                return
            shared = detect_pid_gpu_shared_mb(int(self._server_proc.pid))
            if shared is None:
                return
            if shared > self.peak_shared_mb:
                self.peak_shared_mb = shared
            if shared > shared_limit and not self.vram_killed:
                self.vram_killed = True
                kind = "dense" if dense else "moe"
                print(
                    f"  [VRAM] SHARED GPU EXCEEDED shared={shared:.0f}MB > "
                    f"limit={shared_limit:.0f}MB ({kind}={self.intent.model_path.name}) — killing "
                    "(WDDM/PCI-e Shared→pagefile freeze)",
                    flush=True,
                )
                self._cleanup_process()
                self._stop_event.set()

        def sampler() -> None:
            nonlocal nvml, shared_tick
            while not self._stop_event.is_set():
                if nvml is not None and device is not None:
                    try:
                        mem_info = nvmlMemory_t()
                        nvml.nvmlDeviceGetMemoryInfo(device, ctypes.byref(mem_info))
                        current = float(mem_info.used) / (1024.0 * 1024.0)
                        total_mb = float(mem_info.total) / (1024.0 * 1024.0)
                        _maybe_kill(current, total_mb)
                        shared_tick += 1
                        # typeperf is slow — sample Shared ~1/s while NVML is 20ms.
                        if shared_tick % 50 == 0:
                            _maybe_kill_shared()
                        self._stop_event.wait(0.02)
                        continue
                    except Exception:
                        nvml = None
                try:
                    current, total_mb = detect_used_total_vram_mb()
                    _maybe_kill(current, total_mb)
                    _maybe_kill_shared()
                except FileNotFoundError:
                    # VRAM sampling unavailable on non-GPU host
                    break
                except (subprocess.CalledProcessError, ValueError):
                    pass
                self._stop_event.wait(0.2)

        self._vram_thread = threading.Thread(target=sampler, daemon=True)
        self._vram_thread.start()

    def _cleanup_process(self):
        if self._server_proc:
            self._server_proc.kill()
            self._server_proc.wait()
            self._server_proc = None
        if self._server_log:
            self._server_log.close()
            self._server_log = None

    def _cleanup_all(self):
        self._stop_event.set()
        self._cleanup_process()
        if self._guard:
            self._guard.teardown()
            self._guard = None
