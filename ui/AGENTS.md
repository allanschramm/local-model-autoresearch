# `ui/` — Operator Dashboard Contract

## Purpose
Read-only localhost dashboard for monitoring Baseline, recent Trials, and Trial server log during Search. Humans watch; agents start the UI.

## Ownership
Repository developers. Port and panels owned here — not by `autoresearch/` runners.

## Local Contracts
- Bind **127.0.0.1:18765** only. UI language **pt-BR**.
- Start: `.\venv\Scripts\python.exe -m ui` (or `./venv/bin/python -m ui`).
- Poll `/api/status` (~2.5s). Payload includes `baseline`, `trials` (last 50), `run_state`, `log_tail`.
- Baseline from `autoresearch/core/config.py` via import+reload — never `.autoresearch_state.json`.
- Trials via `autoresearch.runners.run.read_rows` + `classify.is_on_front` (`keep` ≡ `on_front`). Never write TSV.
- Log path pinned: `autoresearch/runners/llama_server.log`. `Em execução` when mtime within ~10s; else `Idle`. Missing log → pt-BR empty, no crash.
- **stdlib-only** (`ui/requirements.txt`). No process control (no start/stop server, autoloop, or Search from the UI).

## Work Guidance
- Agent starts the dashboard when the human wants to monitor a Trial/Search session.
- Human monitors the browser; agent does not treat the dashboard as a control plane.
- Keep panels additive and poll-friendly; prefer small helpers (`trial_reader.py`, `run_log.py`).

## Verification
- Smoke: `python -m ui` → GET `/` (200, `lang=pt-BR`) and GET `/api/status` (JSON with baseline/trials/run_state).

## Child DOX Index
None — `ui/` is a leaf.
