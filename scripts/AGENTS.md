# `scripts/` — Utilities and Runner Scripts

## Purpose
Operator scripts for running setup health checks, monitoring GPU metrics, discovering candidate models, and managing the server daemon.

## Ownership
Repository operators and developers.

## Local Contracts
- Scripts must be runnable from the repository root.
- `model_up.py` (global `model-up`) must work from any cwd: resolve `--model` and draft path flags (`--spec-draft-model` / `-md` / `--model-draft`) to absolute paths, and spawn `llama-server` with `cwd=REPO_ROOT`.
- `setup-check.sh` is the canonical readiness verification script (supports GPU acceleration and CPU-only builds). It must import-check every package listed in root `requirements.txt` (including `gguf`).
- `hooks/block-adhoc-eval.ps1` — shell policy (python allowlist, config-only Baseline, cwd, no gate rewrite).
- `hooks/block-gate-tamper.ps1` — deny Edit/Write/Delete on gate wiring paths.
- **Rollback:** [docs/discovery/agent-shell-hard-gates.md](../docs/discovery/agent-shell-hard-gates.md) §3.

## Work Guidance
- Use `serve-config.py` as the preferred CLI helper to start/stop the llama-server daemon based on the mutable Baseline in `config.py`. It runs VRAM + host-memory preflight before spawn and exits 2 on reject.
- Use `build-llamacpp.py` (`python scripts/build-llamacpp.py --cpu` or `--cuda`) to build runtime binaries for local inference.
- Use `check_hardware.py` to diagnose hardware on **Windows, macOS, and Linux**: RAM, NVIDIA VRAM (`discrete_gpu`) or unified memory / no discrete NVIDIA (`unified_memory`, including Apple Silicon Metal). Report memory class + model pool; give conservative GGUF/context guidance. Dense must fit the detected pool (VRAM or unified RAM with OS headroom); Metal uses `-ngl 99`, true CPU-only uses `-ngl 0`; never suggest partial dense offload. Agents treat this script as local fit authority over whichllm/llmfit rankings.
- Use `verify_setup.py` to validate local API server health and benchmark real-time TPS. Its default port matches `serve-config.py` (18080).
- Use `lcb_only.py` to re-measure LiveCodeBench (10 tasks) against current Baseline — gambiarra when coding HE/MBPP/BC already logged but LCB cache failed.
- Use `rank_results.py` to print Pareto / Day / Night / claw / coding rankings from `results.tsv`. Agents must use this CLI for model rankings — no ad-hoc temp filter scripts.
- Maintain helper commands documented in README.md.

## Verification
- Test script changes locally by executing them.
- `rank_results.py`: `.\venv\Scripts\python.exe -m pytest tests/test_rank_results.py`
- Ensure `bash scripts/setup-check.sh` passes before declaring environment readiness.

## Child DOX Index
- [hooks/block-adhoc-eval.ps1](hooks/block-adhoc-eval.ps1) — shell hard-gate.
- [hooks/block-gate-tamper.ps1](hooks/block-gate-tamper.ps1) — gate-file hard-gate.
- [lcb_only.py](lcb_only.py) — LCB-only remeasure helper (`scripts/lcb_only.py`).
- [rank_results.py](rank_results.py) — Pareto / Day / Night / claw / coding ranking over `results.tsv` (ADR 0006/0008).
