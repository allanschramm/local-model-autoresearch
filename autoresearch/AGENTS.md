# `autoresearch/` — Core Autotuning Package

## Purpose
Core codebase containing search strategy optimization logic, llama.cpp server wrappers, API client integrations, and evaluation benchmark harnesses (Nexus, Claw, Coding).

## Ownership
Repository developers.

## Local Contracts
- Do not modify internal evaluation logic or benchmarks under `autoresearch/benchmarks/` without authorization.
- `autoresearch/core/config.py` owns the mutable Baseline (`ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS`) and validation. File is **gitignored**; seed from `config.py.example`. Before the first Trial on a model, seed `SAMPLER_DEFAULTS` from `docs/models/<card>.md` Recommended settings for the job (agentic/general vs coding). The ignored `.autoresearch_state.json` stores visited memory only. `TPS_FLOOR` (default 20.0) is the only Trial reject gate for throughput — set per model (MoE on 8GB often needs 15–18).
- Baseline `VRAM_LIMIT_MB` is the single VRAM budget for autoloop preflight and runtime monitoring; there is no CLI override.
- VRAM preflight estimates main GPU-resident weights + configured `CTX_SIZE` KV + enabled external draft weights + conservative speculative/MTP workspace (`512 + 256 * SPEC_DRAFT_N_MAX` MiB) before server start. **Exception (workspace):** MoE with `n_cpu_moe > 0` and an external draft file (e.g. `draft-dflash`) counts draft weights only — no flat speculative workspace (false-rejects DFlash on 8 GB when measured peaks are ~4 GB). Embedded MTP (no draft file) and dense targets still use the workspace term. **Exception (free clamp):** MoE with `n_cpu_moe > 0` uses the configured `VRAM_LIMIT_MB` only — does not clamp to `free−headroom` (OS-reserved VRAM false-rejects expert-CPU offload). Dense still uses free-at-start headroom (issue #10). **Operator escape:** `AUTORESEARCH_SKIP_FREE_CLAMP=1` skips the free clamp for dense too (configured budget only; runtime VRAM monitor remains the kill guard — WDDM desktop reservation false-rejects). Runtime VRAM monitoring remains the final kill guard.
- Host-memory gate lives in `autoresearch/core/hardware.py` + `preflight_host_memory` in `llama_runner.py`: full GGUF (+ draft + KV), no MoE shrink. Unified headroom `max(6144, 0.20×RAM)` MiB; discrete `max(4096, 0.15×RAM)`. Override `HOST_MEMORY_HEADROOM_MB` / `AUTORESEARCH_HOST_HEADROOM_MB`.
- Do not add hardcoded user or absolute directory paths in the source files.
- Model paths: `resolve_model_path(models_dir, ref)` owns flat + nested (`publisher/model/*.gguf`) lookup. Config Baseline keeps basenames (and `draft/...` for drafts).
- Architecture class (MoE vs dense): `autoresearch/core/model_arch.py` reads local GGUF metadata (`expert_count` / `*moe*` arch). No filename heuristics. Model cards must mirror that GGUF truth.
- MoE `N_CPU_MOE=None` → `resolve_n_cpu_moe` sets `--n-cpu-moe` to GGUF `block_count` (replaces old `override-tensor .*exps.*=CPU`). Explicit `0` = full GPU; `N>0` = manual.
- GPU/CPU split: `N_GPU_LAYERS` in Baseline (`-1` = Auto, `0` = CPU-only, `N>0` = N layers on GPU) and `NUMA` (`None` / `distribute` / `isolate`) are validated in `validate_config` and surface through autoloop `CORE_PASSTHROUGH` (issue #16). CPU-only Baseline is first-class via `N_GPU_LAYERS=0`.
- **Trial order (hard gate):** GGUF arch classify → resolve `N_CPU_MOE` → VRAM preflight → **host-memory preflight** (full GGUF + KV, no MoE shrink) → TPS floor → eval. MoE `N_CPU_MOE=0` that exceeds physical VRAM is `MODEL_REJECTED` (set `None` for auto offload). Dense never uses expert offload. Host gate rejects oversized unified-RAM loads (`HOST_MEMORY_PREFLIGHT`) before Metal/CUDA start.
- **Use the harness, not raw binaries**: Run `benchmark_search.py` or `autoloop.py` for evaluation. Do not invoke `llama-server` or `llama-bench` directly — the harness resolves paths (supporting both `build-cuda` and `build-cpu`), translates config flags to CLI args, manages server lifecycle, monitors VRAM, and logs results. **Claw/agentic loop:** merge `reasoning_content` when `content` is empty (same as coding-10) and pass `max_tokens≥2048` for agentic tiers — thinking models otherwise get blank graders + mid-loop HTTP 400.
- **Status recompute after every Trial write (issue #5)**: `run.py` / `autoloop.py` refresh the whole store's statuses after each write via `autoresearch/core/recompute.py` (pure, idempotent) so a new `on_front` point demotes rows it dominates to `dominated`. `scripts/recompute_status.py` runs the same refresh retroactively. Stored status is the global-by-hardware+budget front (ADR 0006); incomplete/rejected and fingerprint-less legacy rows never compete. **Budget bucket** = `round(VRAM_LIMIT_MB/1024)` from `config_json` (intended limit); peak `memory_gb` is legacy fallback only — same-Fingerprint Trials with different peaks must still merge.
- **Perplexity-Guided Tuning Guard**: When using `--perplexity-val` to maximize throughput (TPS), enforce a strict quality ceiling: any candidate configuration resulting in more than a 1% increase in perplexity (PPL) compared to the baseline must be discarded.
- **Single-load gate (issue #41)**: `autoresearch/core/single_load.py` refuses a second full server while one is live on a harness port (`SingleLoadError`, reuses the #37 detection surface). Wired into `LlamaServerRunner.__enter__` and `SGLangServerRunner.start` before the orphan sweep; a speculative draft on the same server never counts. Fail-open like the sweep: a detection failure logs and proceeds. Bypass via `AUTORESEARCH_ALLOW_MULTI_SERVERS=1` (also skips the sweep so a live sibling is not killed).

## Work Guidance
- Implement mock classes for system hardware calls (like GPU VRAM) to ensure code remains testable across environments.
- Keep dependencies minimal and avoid adding new third-party libraries.

## Verification
- Run `pytest` on tests checking core runners (`test_llama_runner.py`, `test_llama_client.py`).
- Run `pytest tests/test_search_strategy.py` and `pytest tests/test_state.py` for core optimization loop and state verification.

## Child DOX Index
None
