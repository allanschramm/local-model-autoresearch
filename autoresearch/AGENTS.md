# `autoresearch/` — Core Autotuning Package

## Purpose
Core codebase containing search strategy optimization logic, llama.cpp server wrappers, API client integrations, and evaluation benchmark harnesses (Nexus, Claw, Coding).

## Ownership
Repository developers.

## Local Contracts
- Do not modify internal evaluation logic or benchmarks under `autoresearch/benchmarks/` without authorization.
- `autoresearch/core/config.py` owns the mutable Baseline (`ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS`) and validation. File is **gitignored**; seed from `config.py.example`. Before the first Trial on a model, seed `SAMPLER_DEFAULTS` from `docs/models/<card>.md` Recommended settings for the job (agentic/general vs coding). The ignored `.autoresearch_state.json` stores visited memory only. `TPS_FLOOR` (default 20.0) is the only Trial reject gate for throughput — set per model (MoE on 8GB often needs 15–18).
- Baseline `VRAM_LIMIT_MB` is the single VRAM budget for autoloop preflight and runtime monitoring; there is no CLI override.
- VRAM preflight estimates main GPU-resident weights + configured `CTX_SIZE` KV + enabled external draft weights + conservative speculative/MTP workspace (`512 + 256 * SPEC_DRAFT_N_MAX` MiB) before server start. Backend/version-specific allocations can still exceed the estimate, so runtime VRAM monitoring remains the final kill guard.
- Host-memory gate lives in `autoresearch/core/hardware.py` + `preflight_host_memory` in `llama_runner.py`: full GGUF (+ draft + KV), no MoE shrink. Unified headroom `max(6144, 0.20×RAM)` MiB; discrete `max(4096, 0.15×RAM)`. Override `HOST_MEMORY_HEADROOM_MB` / `AUTORESEARCH_HOST_HEADROOM_MB`.
- Do not add hardcoded user or absolute directory paths in the source files.
- Model paths: `resolve_model_path(models_dir, ref)` owns flat + nested (`publisher/model/*.gguf`) lookup. Config Baseline keeps basenames (and `draft/...` for drafts).
- Architecture class (MoE vs dense): `autoresearch/core/model_arch.py` reads local GGUF metadata (`expert_count` / `*moe*` arch). No filename heuristics. Model cards must mirror that GGUF truth.
- MoE `N_CPU_MOE=None` → `resolve_n_cpu_moe` sets `--n-cpu-moe` to GGUF `block_count` (replaces old `override-tensor .*exps.*=CPU`). Explicit `0` = full GPU; `N>0` = manual.
- **Trial order (hard gate):** GGUF arch classify → resolve `N_CPU_MOE` → VRAM preflight → **host-memory preflight** (full GGUF + KV, no MoE shrink) → TPS floor → eval. MoE `N_CPU_MOE=0` that exceeds physical VRAM is `MODEL_REJECTED` (set `None` for auto offload). Dense never uses expert offload. Host gate rejects oversized unified-RAM loads (`HOST_MEMORY_PREFLIGHT`) before Metal/CUDA start.
- **Use the harness, not raw binaries**: Run `benchmark_search.py` or `autoloop.py` for evaluation. Do not invoke `llama-server` or `llama-bench` directly — the harness resolves paths (supporting both `build-cuda` and `build-cpu`), translates config flags to CLI args, manages server lifecycle, monitors VRAM, and logs results.
- **Status recompute after every Trial write (issue #5)**: `run.py` / `autoloop.py` refresh the whole store's statuses after each write via `autoresearch/core/recompute.py` (pure, idempotent) so a new `on_front` point demotes rows it dominates to `dominated`. `scripts/recompute_status.py` runs the same refresh retroactively. Stored status is the global-by-hardware+budget front (ADR 0006); incomplete/rejected and fingerprint-less legacy rows never compete.
- **Perplexity-Guided Tuning Guard**: When using `--perplexity-val` to maximize throughput (TPS), enforce a strict quality ceiling: any candidate configuration resulting in more than a 1% increase in perplexity (PPL) compared to the baseline must be discarded.

## Work Guidance
- Implement mock classes for system hardware calls (like GPU VRAM) to ensure code remains testable across environments.
- Keep dependencies minimal and avoid adding new third-party libraries.

## Verification
- Run `pytest` on tests checking core runners (`test_llama_runner.py`, `test_llama_client.py`).
- Run `pytest tests/test_search_strategy.py` and `pytest tests/test_state.py` for core optimization loop and state verification.

## Child DOX Index
None
