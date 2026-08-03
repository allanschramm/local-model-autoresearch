# `scripts/` — Utilities and Runner Scripts

## Purpose
Operator scripts for running setup health checks, monitoring GPU metrics, discovering candidate models, and managing the server daemon.

## Ownership
Repository operators and developers.

## Local Contracts
- Scripts must be runnable from the repository root.
- OpenVINO GenAI remains optional. `bench_openvino.py` imports it only while running and exits nonzero with an actionable install message when unavailable.
- `model_up.py` (global `model-up`) must work from any cwd: resolve `--model` and draft path flags (`--spec-draft-model` / `-md` / `--model-draft`) to absolute paths, and spawn `llama-server` with `cwd=REPO_ROOT`.
- `setup-check.sh` is the canonical readiness verification script (supports GPU acceleration and CPU-only builds). It must import-check every package listed in root `requirements.txt` (including `gguf`).
- Claude Code hard-gates live under [`.claude/hooks/`](../.claude/hooks/) (not `scripts/`). See [docs/discovery/agent-shell-hard-gates.md](../docs/discovery/agent-shell-hard-gates.md).
- **Rollback:** [docs/discovery/agent-shell-hard-gates.md](../docs/discovery/agent-shell-hard-gates.md) §3.

## Work Guidance
- Use `serve-config.py` as the preferred CLI helper to start/stop the llama-server daemon based on the mutable Baseline in `config.py`. It runs VRAM + host-memory preflight before spawn and exits 2 on reject.
- Use `build-llamacpp.py` (`python scripts/build-llamacpp.py --cpu` or `--cuda`) to build runtime binaries for local inference.
- Use `check_hardware.py` to diagnose hardware on **Windows, macOS, and Linux**: RAM, NVIDIA VRAM (`discrete_gpu`) or unified memory / no discrete NVIDIA (`unified_memory`, including Apple Silicon Metal). Report memory class + model pool; give conservative GGUF/context guidance. Dense must fit the detected pool (VRAM or unified RAM with OS headroom); Metal uses `-ngl 99`, true CPU-only uses `-ngl 0`; never suggest partial dense offload. Agents treat this script as local fit authority over whichllm/llmfit rankings.
- Use `verify_setup.py` to validate local API server health and benchmark real-time TPS. Its default port matches `serve-config.py` (18080).
- Use `lcb_only.py` to re-measure LiveCodeBench (10 tasks) against current Baseline — gambiarra when coding HE/MBPP/BC already logged but LCB cache failed.
- Use `recompute_status.py` to refresh Trial statuses in a `results.tsv` after the fact (Pareto Set recompute, issue #5). Write paths (`run.py`, `autoloop.py`) already recompute after every Trial; the script covers retroactive refreshes. Idempotent, no GPU.
- Use `rank_results.py` to print Pareto / Day / Night / claw / coding rankings from `results.tsv`. Agents must use this CLI for model rankings — no ad-hoc temp filter scripts.
- Use `measure_vram_peak.py` to measure REAL peak VRAM for the Baseline model, bypassing the preflight gate (calibration for the VRAM estimator; uses the harness `LlamaServerRunner` sampler, writes no results row).
- Maintain helper commands documented in README.md.

## Verification
- Test script changes locally by executing them.
- `rank_results.py`: `.\venv\Scripts\python.exe -m pytest tests/test_rank_results.py`
- Ensure `bash scripts/setup-check.sh` passes before declaring environment readiness.
- Git pre-commit (Ruff/pytest) is repo-root owned — see root `AGENTS.md` Verification + `.pre-commit-config.yaml`. Not under `scripts/hooks` (removed; Claude gates live in `.claude/hooks/`).

## Child DOX Index
- [bench_openvino.py](bench_openvino.py) — optional OpenVINO GenAI CPU/iGPU benchmark with separate prefill/decode TPS output.
- [run_pytest_hook.py](run_pytest_hook.py) — pre-commit local entry for venv pytest.
- [lcb_only.py](lcb_only.py) — LCB-only remeasure helper (`scripts/lcb_only.py`).
- [recompute_status.py](recompute_status.py) — store-wide Pareto status recompute over a results.tsv (issue #5; `autoresearch/core/recompute.py` owns the logic).
- [rank_results.py](rank_results.py) — Pareto / Day / Night / claw / coding ranking over `results.tsv` (ADR 0006/0008).
- [measure_vram_peak.py](measure_vram_peak.py) — real peak-VRAM measurement for the Baseline model, gate bypass (estimator calibration, issue #10).
- Hard-gates (Claude): [../.claude/hooks/](../.claude/hooks/) + [../docs/discovery/agent-shell-hard-gates.md](../docs/discovery/agent-shell-hard-gates.md).
- Git pre-commit: [../.pre-commit-config.yaml](../.pre-commit-config.yaml) + [../pyproject.toml](../pyproject.toml).
