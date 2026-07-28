# Session 2026-07-06 — Windows `model-up` shim

## Goal
Make `models/aliases/<name>/config.yaml` recipes callable from Windows shells (PowerShell and `cmd.exe`), same idea as the WSL PATH launcher.

## Hardware
- Windows workstation in the repo workspace
- Shell: Git Bash / Windows PowerShell available

## Setup
- Repo root: local checkout
- Alias configs under gitignored `models/aliases/`
- Launcher logic in `scripts/model_up.py` (bootstraps repo root into `sys.path`)
- Windows shims live next to the local alias tree (not tracked)

## Commands
```bat
model-up <name>
model-up list
model-up status
model-down
```

## Findings
- Alias YAML remains source of truth for local recipes.
- Thin Windows wrapper enough; no second alias system.
- Launcher parses the limited YAML shape and splits flag strings with `shlex`.

## Errors
- `pytest` missing in default Python env during first test attempt — use project venv.
- Some sandbox path probes were flaky; implementation stayed in new files only.

## Decisions
- One Python launcher + thin shell shims.
- Preserve `models/aliases/<name>/config.yaml` layout (machine-local).
- Put launcher on PATH; Python resolves repo root dynamically (no hardcoded user checkout path).
