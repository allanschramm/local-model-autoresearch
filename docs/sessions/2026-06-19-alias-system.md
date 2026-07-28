# Session 2026-06-19 (part 2) — Alias System + Launcher

## Goal
Global launcher so any WSL terminal can start a configured local server without `cd` into the repo. Detached process; OpenAI-compatible endpoint for agent harnesses.

## Design decision

Rejected: navigate to `models/aliases/<name>/` and run a local script.

Chosen: single launcher on PATH (e.g. `~/.local/bin/model-up` / historical `qwen-up`), reads `models/aliases/<name>/config.yaml` via a resolved absolute path (not CWD-dependent). Alias YAML stays machine-local under gitignored `models/`.

## Files created (shape)

| Path | Role |
|---|---|
| PATH launcher script | Python; detach via `subprocess.Popen(start_new_session=True)` |
| `models/aliases/<name>/config.yaml` | Per-recipe flags + model basename (local) |
| Skill under `.agents/skills/local-model-alias/` | How to add a new alias |

## Subcommands

```bash
model-up                # default alias
model-up <name>         # named alias
model-up list
model-up status
model-up stop
```

## Validation

1. `list` returned configured aliases.
2. Default start → server ready on loopback OpenAI port.
3. New shell `status` → process still alive after prior shell exit.
4. `curl POST /v1/chat/completions` → OK.
5. `stop` → clean kill + PID file removed.

Bugs fixed during validation:
- YAML flag string must be `shlex.split()` into argv for llama-server.
- `status` must split `/proc/PID/cmdline` on `\0`, not spaces.

## Errors

Non-login `bash -c` lacks `~/.local/bin` on PATH — use `bash -lc` or interactive shell.

## Thinking mode note

Qwen3.6 defaults may fill `max_tokens` with reasoning and leave `content` empty. Raise `max_tokens` or set `--chat-template-kwargs '{"enable_thinking": false}'` in alias `flags:`.

## Agent harness integration (shape)

Point any OpenAI-compatible client at `http://127.0.0.1:<port>/v1` with the model id registered by the server.

## Follow-ups

- Optional default `enable_thinking: false` for harness UX.
- Optional MTP-on alias (`--spec-type mtp`) as a separate Trial recipe.
