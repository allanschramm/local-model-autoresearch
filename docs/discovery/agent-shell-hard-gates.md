# Agent shell hard-gates

**Date:** 2026-07-29 (updated: 2026-08-03 — added `block-git-commit.ps1` git-commit guardrail)  
**Audience:** operators + next coding agent + anyone who clones  
**Purpose:** What is installed, what it blocks, how a **human** turns it off. Clone-and-use — no OS admin rituals.

> **Agent rule:** If the user asks to disable / remove / revert these gates, **do not silently strip them**. Teach the steps in §3, wait for explicit “do it” / “desliga” / “remove”, then apply.  
> Editing **wiring** under `.claude/hooks/**` or `.claude/settings.json` requires explicit user unlock — hooks deny those paths.

---

## 1. Inventory (live)

| Piece | Path | Role |
|---|---|---|
| Shell policy | `.claude/hooks/block-adhoc-eval.ps1` | PreToolUse Bash\|PowerShell: cwd check; python allowlist; deny Baseline CLI overrides / `-c` / raw llama / shell rewrite of gates |
| Gate-file policy | `.claude/hooks/block-gate-tamper.ps1` | PreToolUse Edit\|Write\|Delete: deny wiring paths |
| Git-commit guardrail | `.claude/hooks/block-git-commit.ps1` | PreToolUse Bash\|PowerShell: deny `git commit` / `git push` unless a fresh human token (`.claude/hooks/.git-commit-allow`, TTL 30 min) exists |
| Pi git-commit guard | `.pi/extensions/git-commit-guard.ts` | pi `tool_call` hook: block `git commit` / `git push` in the bash tool unless the user confirms in the TUI (headless → always block) |
| Post-tool audit | `.claude/hooks/audit-post-tool.ps1` | PostToolUse Bash\|PowerShell\|Edit\|Write: append `.claude/hooks-audit.log` (fail-open; `*.log` gitignored) |
| Claude wiring | `.claude/settings.json` | `allow` / `ask` / `deny` + `disableBypassPermissionsMode` + PreToolUse + PostToolUse (Claude-only) |
| Claude local | `.claude/settings.local.json` | Machine-only allow extras (gitignored). Keep empty or narrow — never `python.exe *` / CLI soup |
| Contract text | `AGENTS.md` + `scripts/AGENTS.md` | DOX pointers |
| Host memory | harness `HOST_MEMORY_PREFLIGHT` | Rejects oversized GGUF+KV vs RAM−headroom on serve / Trial (not a Claude permission rule) |

**Out of scope (by design):** OS ACL / `icacls` / chmod lockdowns / enterprise managed hooks / Cursor project hooks. Clone users get Claude Code in-repo settings + `.claude/hooks/` only.

---

## 2. What the live gate blocks / allows

**Shell — blocked:**

- `python -c` / `--command`
- Any `python`/`py` not hitting allowlist entrypoints
- Scratch scripts e.g. `python .scratch\matrix.py`
- Direct `llama-cli` / `llama-server` / `llama-bench`
- Baseline overrides on `benchmark_search.py` (`--model`, `--threads`, `--n-cpu-moe`, batch/KV/sampler flags, etc.); edit `autoresearch/core/config.py` instead
- `cwd` (or `cd` to absolute path) outside `workspace_roots` / `CLAUDE_PROJECT_DIR`
- Shell rewrite of gate paths (`Set-Content`, redirects, `Remove-Item`, …)
- `git commit` / `git push` (incl. `--amend`, `--force`) without a fresh permission token — see §3.6 for the grant command

**Shell — python allowlist:**

- `benchmark_search.py`
- `autoloop.py`
- `-m pytest` / `-m unittest`
- `scripts\*.py` (operator scripts)

**Shell — allowed without python:** e.g. `nvidia-smi`, `git status`, listing files (still subject to cwd + gate-rewrite rules).

**Edit/Write/Delete — blocked paths:**

- `.claude/settings.json`
- anything under `.claude/hooks/`

**Editable:** `docs/discovery/agent-shell-hard-gates.md` (this file), `AGENTS.md`, app code, `config.py`, etc.

### Claude Code permissions (`.claude/settings.json`)

Evaluation order: **deny → ask → allow**. Hooks still exit 2 independently of allow/ask.

| Tier | Purpose | Examples |
|---|---|---|
| `allow` | Onboarding without prompt spam | `check_hardware.py`, `verify_setup.py`, `pytest`, `nvidia-smi`, `hf models/repos/file-size`, read-only `git` |
| `ask` | Human confirms before disk/VRAM burn | `hf download`, `serve-config`, `model_up`, `benchmark_search`, `autoloop` |
| `deny` | Hardblock | `.env` / `.env.*` (Read/Edit/Write — pedagogical; repo may have none), `rm`, path-tolerant `python -c` / `py -c`, raw `llama-cli\|server\|bench`, gate wiring |

`disableBypassPermissionsMode: "disable"` — YOLO cannot skip deny floors.

**Not in permissions:** “model/ctx too big for this rig”. That is harness `HOST_MEMORY_PREFLIGHT` / VRAM preflight when spawn goes through `serve-config` / `benchmark_search` / `autoloop`. Deny+hook force that path (no raw llama).

**Local extras:** `.claude/settings.local.json` (gitignored). Keep allow empty or narrow. Do not re-add `Bash(venv/Scripts/python.exe *)` or Baseline CLI soup — hooks block soup anyway; broad allow trains bad habits.

**PostToolUse:** `audit-post-tool.ps1` logs successful Bash/PowerShell/Edit/Write to `.claude/hooks-audit.log`. Fail-open (never blocks). Use the log to tighten allow/ask later — same idea as S2D3 “comanda na saída”.

**Hook script location:** Anthropic examples use `.claude/hooks/`; this repo follows that convention (`${CLAUDE_PROJECT_DIR}/.claude/hooks/...` in settings).

---

## 3. How to DISABLE everything (rollback playbook)

Use when the user wants to “voltar atrás”. Do **one layer at a time**; restart Claude Code after file changes.

### 3.1 Fastest — Claude session only (keeps files)

1. Start Claude without project settings, or temporarily move/rename `.claude/settings.json` yourself.
2. Or use a session that does not load project hooks (see Claude Code docs). Prefer §3.2 for a durable off.

### 3.2 Full repo rollback (remove project enforcement)

**Explicit unlock** before an agent can delete wiring — otherwise Edit/Write on these paths is denied.

From repo root, after explicit user OK:

```powershell
Remove-Item -Force .claude/settings.json -ErrorAction SilentlyContinue
# optional:
# Remove-Item -Recurse -Force .claude/hooks -ErrorAction SilentlyContinue
```

Then strip AGENTS.md hard-gate bullets (Edit tool).  
Restart Claude Code.  
Smoke: `python -c "print(1)"` via agent should no longer be project-denied.

### 3.3 Git rollback

```powershell
git log --oneline -- .claude/settings.json .claude/hooks
# revert or checkout prior revision of those paths
```

### 3.4 Claude permissions / yolo

- Removing `.claude/settings.json` drops `permissions.deny` + hooks together (current file shape).
- Yolo / `bypassPermissions` does **not** replace unwiring hooks. To run without gates, use §3.1–3.2.

### 3.5 OS ACL / enterprise

**Do not use for this repo.** Not supported. See §7.

### 3.6 Git-commit guardrail (permission + rollback)

`block-git-commit.ps1` blocks `git commit` / `git push` unless a **human-created token** exists. Grant (PowerShell, repo root):

```powershell
New-Item -ItemType File -Force .claude/hooks/.git-commit-allow | Out-Null
```

- Token auto-expires after 30 minutes (TTL). Re-create it for each batch of commits you approve.
- Agents cannot create the token: `.claude/hooks/**` is denied to Edit/Write/Shell rewrite, and the token is gitignored (`git status` stays clean).
- Commit flow: you say “commit” → you create the token → agent runs `git commit` → hook allows.
- **Rollback / disable:** `Remove-Item -Force .claude/hooks/.git-commit-allow` removes the token; deleting `block-git-commit.ps1` + its PreToolUse entries in `.claude/settings.json` removes the guardrail (wiring edits require explicit unlock).

### 3.7 Pi agent guardrail (interactive confirm)

`.pi/extensions/git-commit-guard.ts` blocks `git commit` / `git push` in the **pi** agent's bash tool:

- TUI session → `ctx.ui.confirm` prompts you for every commit/push. Deny = blocked.
- Headless / no UI → always blocked (no silent commits).
- No token file needed. Non-git commands (`git status`, `git log`, `git diff`) pass.
- **Rollback / disable:** delete `.pi/extensions/git-commit-guard.ts` (or the whole `.pi/extensions/` dir) and restart pi.

---

## 4. Script for the next agent — “teach me to disable”

1. Confirm soft (rename settings) vs hard (delete files) vs git revert.  
2. Point to **this doc §3**.  
3. List inventory §1.  
4. Wait for explicit go-ahead / unlock.  
5. Delete or revert `.claude/settings.json` (+ optional `.claude/hooks`).  
6. Restart Claude Code + verify.  
7. Offer to strip AGENTS.md bullets in the same change set.

---

## 5. Threat model (short)

In-repo hooks = strong friction, not a vault. Residual: user disables hooks; obfuscation; tools that omit `path` in payload. Clone-and-use wins over per-machine lockdowns.

| Control | Claude Code (this repo) |
|---|---|
| Block shell | `PreToolUse` exit 2 (`block-adhoc-eval.ps1`) |
| Block git commit/push | Claude Code: `PreToolUse` exit 2 (`block-git-commit.ps1`) — fresh human token required. Pi: `tool_call` block (`.pi/extensions/git-commit-guard.ts`) — user confirm required |
| Block file edit | `permissions.deny` + PreToolUse (`block-gate-tamper.ps1`) |
| Audit after tool | `PostToolUse` (`audit-post-tool.ps1`) → `.claude/hooks-audit.log` |
| Soft ask | `permissions.ask` (download / serve / Trial) |
| Sandbox OS | not native Windows — rely on allow/ask/deny + hooks |

---

## 6. Yolo vs human

- Yolo skips approval prompts; it does not reliably skip exit-2 / deny hooks.  
- Wiring paths are denied to Edit/Write/Shell rewrite → agents **teach** §3; humans apply rollback.  
- **No OS ACL / icacls.** Anyone who clones gets the same in-repo hooks with zero machine setup.

---

## 7. What clone users get (portable)

On clone + open in Claude Code:

1. Project `.claude/settings.json` loads allow/ask/deny + Pre/Post hooks from `.claude/hooks/`.  
2. No Windows/Linux permission commands required.

If hooks do not fire: restart Claude Code; confirm project trust / that settings were not renamed away.

**Explicitly not part of the product:** Cursor project hooks, `icacls`, chmod lockdowns, enterprise managed hooks, or any per-machine admin ritual.

---

## 8. Sources / Verification

- Claude Code Hooks / Settings: https://code.claude.com/docs/en/hooks , https://code.claude.com/docs/en/settings , https://code.claude.com/docs/en/permissions — 2026-07-29  
- Smoke (2026-07-21): deny `python -c`, scratch `.py`, llama-cli, Baseline CLI overrides, foreign cwd, Set-Content gate; allow config-driven `benchmark_search.py`, `-m pytest`, `nvidia-smi`; deny Write gate files; allow Write `README.md`.  
- Smoke (2026-07-29): hooks under `.claude/hooks/`; Allow/Ask/Deny + PostToolUse audit; pedagogical `.env` + `rm` + path-tolerant `*python* -c *`; live demo: Bash tool required for Pre-hook (chat-only “I ran it” does not fire hooks).
- Smoke (2026-08-03): `block-git-commit.ps1` — deny `git commit`/`git push` without token; allow `git status`/`git log --grep commit`; allow with fresh token; deny with 61-min-old (stale) token. All six cases exit as expected (Claude payloads).
- Smoke (2026-08-03): `.pi/extensions/git-commit-guard.ts` — block commit/push with no UI; allow `git status`/`git log --grep commit`; block push on user deny; allow commit on user allow; non-bash tools untouched (node type-strip run with fake pi).

---

## Open questions

- None open for Claude-only wiring under `.claude/hooks/`.
