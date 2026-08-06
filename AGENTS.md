# AutoResearch Agent Work Contract

<!-- Scope: repo development agents. Research loop agents → read program.md -->

## Purpose
This repository implements an autonomous multi-objective Search that benchmarks local LLM runtime flags and maintains a Pareto frontier (ctx × TPS × agentic × coding) so each rig can pick a strong point for its hardware and usage profile.

## Ownership
Repository-wide agent guidelines are owned by the repository developers.

## Local Contracts
- **DO NOT USE PYTHON TO EDIT FILES.** Use the Edit tool for targeted changes. Use Write only for new files or full rewrites the user explicitly requested. Never run a Python script to read-modify-write a data file — it corrupts headers, drops rows, and destroys formatting.
- Respond terse like smart caveman. All technical substance stay. Only fluff die.
- Loop agents: Strictly forbidden from editing code. If error/crash occurs, stop immediately, report error, warn user.
- Results local-only: Never push results, tweaks, or run branches to remote repository. Keep all benchmark runs offline.
- Model downloads: Always use `hf` CLI tool to download models, never web download scripts or browser. Land main GGUFs under `models/<publisher>/<model>/` (LM Studio nested layout). Config keeps basename only; harness resolves via `resolve_model_path`. Drafts stay in `models/draft/`. See [models/README.md](models/README.md).
- Parallel processes: NEVER run multiple validations, benchmarks, or command tasks in parallel. Always run one command/task at a time sequentially.
- Architecture: Never overengineer. Keep it simple. Less is more. Reduce lines of code. Simplify instead of complicate.
- Docs always: Update relevant docs (model cards, ADRs, config comments) whenever any codebase/model/config improvement is found or applied.
- Config surface: Agents and the Search loop change Baseline only via `autoresearch/core/config.py`. Do not drive Trials with CLI flag soup. Never edit `program.md` or harness code from the Search loop. Before the first Trial on a model, seed `SAMPLER_DEFAULTS` from that model's card Recommended settings (job profile: agentic/general vs coding).
- **Hard gate (hooks):** Claude Code shell allowlist + Baseline CLI-override rejection + gate-file protection + git-commit guardrail; pi agent git-commit guard extension. Scripts: `.claude/hooks/block-adhoc-eval.ps1`, `.claude/hooks/block-gate-tamper.ps1`, `.claude/hooks/audit-post-tool.ps1`, `.claude/hooks/block-git-commit.ps1`; extension `.pi/extensions/git-commit-guard.ts`. Wiring: `.claude/settings.json`. Trial loop = edit `config.py` → `benchmark_search.py` / `autoloop.py`. **Disable playbook:** [docs/discovery/agent-shell-hard-gates.md](docs/discovery/agent-shell-hard-gates.md) §3 (teach human; wiring edits require unlock).
- Context size: CTX_SIZE default is 131072. User may lower it to trade context for speed. Code minimum is 2048 (llama.cpp practical floor). Always use the user-configured value.
- No timeouts: Never set execution timeouts on commands unless explicitly told to. Benchmarks and model tests run until completion.
- No hardcoded machine paths: Do not commit absolute user or checkout paths in scripts, docs, configs, or durable notes. Resolve them dynamically or keep them repo-relative.
- Ask first, ship never: When user asks "can we do X?", answer yes/no only. Do not implement unless user explicitly says "do it" / "implement" / "go ahead".
- Never assume. When uncertain whether a file is scratch, a decision is right, or a path is safe — ask the user or yourself explicitly before acting.
- NEVER commit and/or push without explicit user command. Wait for "commit", "commit and push", or equivalent. Do not infer intent. Mechanically enforced: Claude Code by `.claude/hooks/block-git-commit.ps1` (one-shot token `.claude/hooks/.git-commit-allow`, TTL 30 min), pi by `.pi/extensions/git-commit-guard.ts` (TUI confirm per commit/push, headless → blocked). See [docs/discovery/agent-shell-hard-gates.md](docs/discovery/agent-shell-hard-gates.md) §3.6–3.7.
- **Atomic commit = no leftovers**: When the user asks for an "atomic commit" / "commit atômico", every current git change that belongs in the repo (modified + untracked) must end up committed — do not leave files sitting unstaged/uncommitted. Split into **multiple logical commits** when the working tree mixes concerns (preferred). "Entire working tree" means cover everything, **not** dump everything into one commit. Only leave paths out when the user explicitly excludes them.
- **Never edit upstream/vendor runtimes:** `llama.cpp/` (submodule), local `claw-eval/`, `VITRIOL/`, and extracted `llama.cpp-releases/` are read-only for agents. No Edit/Write/Delete/patch inside them. Work around via env (`PYTHONUTF8=1`), harness code owned by this repo, or ask the user.

## Work Guidance
- Use `/caveman lite|full|ultra|wenyan` for communication style constraint.
- Prioritize test-driven sanity. Verify logic changes using the test suite.
- Maintain mutable Baseline in local `autoresearch/core/config.py` (gitignored; seed from [config.py.example](autoresearch/core/config.py.example)). Visited memory lives in `.autoresearch_state.json`.
- `program.md` and evaluation harnesses are fixed unless the user explicitly requests a change.

## Verification
- Test with `pytest` via project venv. Ensure the full collected test suite passes.
- **Git pre-commit (this repo only):** `pre-commit` + Ruff + pytest when owned Python changes. Install once: `uv pip install --python ./venv/Scripts/python.exe -r requirements.txt` then `.\venv\Scripts\pre-commit.exe install`. Manual: `.\venv\Scripts\pre-commit.exe run --all-files`. Config: `.pre-commit-config.yaml` + `pyproject.toml`. Does **not** wire hooks into `llama.cpp` / vendor clones / parent monorepo.
- Inspect `results.tsv` to ensure it is not polluted or modified by agent logic.

## Pre-Task Reading
- Before starting any task, read the frontmatter of **every available tool and skill** — tool names, descriptions, parameter schemas, and skill files. Know what each can do before choosing one.
- Do not rely on memory. Re-read tool/skill descriptions each session. Tool schemas and skill instructions change.

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

- **Read the benchmark-run docs before any Trial (hard)**: Before running any Trial/benchmark, read how to run benchmarks the right way first — `docs/discovery/good-enough-tuning.md` (default recipe: seed Baseline from card → smoke → hill-climb speed → complete the Objective Vector), `docs/discovery/discover-models.md` (end-to-end workflow), and the harness rules in `autoresearch/AGENTS.md` (use the harness, never raw binaries). Never launch a Trial from memory or an assumed procedure; `--dry-run` first and read the `[cli-bench]` command before burning GPU hours.

- **`results.tsv` is ground truth**: Measured Trial rows beat docs, memory, and heuristics. Leaderboards / Day-Night ranks / “incomplete” claims must be derived from TSV (or explicitly marked missing in TSV) via `scripts/rank_results.py` — never ad-hoc temp filter scripts. Never omit a measured model because it is weak, deleted from disk, dominated, or “not worth keeping” — data stays valuable; docs are a secondary view of that data.
- **Tracked docs = scores + basenames only**: Do not write which GGUFs are present, kept, or deleted on this machine into tracked docs or session logs. Machine-private notes live under gitignored `models/` (e.g. `REMOVED.md`, `aliases/REMOVED.md`) — not a parallel session tree.
- **No 0-trial model deletion on cleanup (hard)**: When performing disk cleanup under `models/`, models with 0 trials in `results.tsv` must NEVER be deleted. Only delete exact file duplicates or benchmarked models (>0 trials) that are dominated on the Pareto Frontier.
- **Quants are distinct Trials**: Different GGUF basenames = different points (e.g. deepreinforce `ornith-1.0-9b-Q4_K_M` ≠ Unsloth `Ornith-1.0-9B-UD-Q4_K_XL`; MTP ≠ non-MTP). Do not call non-UD “legacy” or skip claw/coding because a sibling quant already has a complete vector.
- **Seed sampler from the model card before Trials**: Before the first Trial on a model, copy publisher-recommended sampling (TEMP / TOP_P / TOP_K / MIN_P / penalties) from `docs/models/<card>.md` § Recommended settings into `SAMPLER_DEFAULTS`. Pick the profile for the job — agentic/general vs coding (they often differ). Do not start Search with the template `TEMP=0.4` when the card says otherwise. Engine knobs may still hill-climb after; sampler stays at recommended until an explicit quality pass.
- **Per-model Baseline, never a leftover (hard)**: `config.py` holds ONE model's Baseline. Before the first Trial on a model, read `docs/models/<card>.md` § Recommended settings and reseed the FULL Baseline (ENGINE + SAMPLER) from the card — never run a Trial with whatever baseline the previous model left in `config.py`. Then check the card's engine/KV types against the runtime: `turbo3`/`turbo4` KV cache types exist ONLY in the TurboQuant release (`AUTORESEARCH_LLAMA_CPP_ROOT=llama.cpp-releases/turboquant/tqp-v0.3.0`); stock `llama.cpp/` builds reject them (`Unsupported cache type: turbo3` → llama-cli exit 1). Record engine/release tag in Trial evidence. When in doubt, run `--dry-run` and read the `[cli-bench]` command before burning a Trial.
- **Fair testing across models**: Always keep exactly 10 tasks per dataset for direct-coding evaluations (never 5 tasks) to guarantee fair model comparisons. Claw-Eval quick smoke (5 tasks) is exempt — it is observational smoke, not a cross-model score.
- **README language**: README.md must always be in pt-BR. Agent-facing docs (docs/, AGENTS.md, GOLDEN-RULES.md, CONTEXT.md, program.md) stay in English.
- **Teach HTML = zero meta leak (hard):** Never put conversation thoughts, curriculum decisions, or design/planning labels into student-facing HTML (`lessons/`, `index.html`, `reference/`). Forbidden patterns include titles/asides like “nota rápida”, “simplificado”, “desta aula”, “já vimos X”, “hoje o foco…”, “(rascunho)” on next-lesson links, or agent↔instructor scaffolding. Teachable concept contrast is OK. Draft-status UI on the guide is the only status exception. Full rule: [teach/AGENTS.md](teach/AGENTS.md).
- **Agentic coding migration**: HumanEval+/MBPP+/LiveCodeBench/BigCodeBench are the coding Objective Vector axis (10 tasks/dataset). Prefer long-horizon agentic targets for future coding-agent quality once adapters exist; Claw full remains the agentic axis today.
- **Pareto Search (ADR 0006 + 0009)**: Keep surface is a Pareto Set on configured `CTX_SIZE` × TPS × agentic (Claw full) × coding (coding-10). Neighbors stay per model; global front ranks models for a hardware+budget. Day/Night are selection lenses (Day → TPS floor `TPS ≥ DAY_TPS_FLOOR` then max IQ, default floor 50.0 TPS — [ADR 0009](docs/adr/0009-day-profile-tps-floor.md); Night → `CTX ≥ NIGHT_CTX_FLOOR` then max `min(agentic, coding)`). Status: `on_front` | `dominated` | `incomplete` | `rejected`. See [CONTEXT.md](CONTEXT.md), [docs/adr/0006-pareto-frontier-search.md](docs/adr/0006-pareto-frontier-search.md), [docs/discovery/pareto-selection.md](docs/discovery/pareto-selection.md). Trial-a-Trial agent contract (Profile pick → edit Baseline → Trial → status; merge by Fingerprint) lives in the same [pareto-selection.md](docs/discovery/pareto-selection.md) § Agent contract. Autoloop entry point (issue #8): `autoloop.py --profile day|night` starts from the pick (loads its `config_json` as Baseline); `--dry-run` prints the plan without running benchmarks; Neighbor acceptance inside rounds = joins/improves the per-model Pareto Set (`improves_set`), scalar keep only for incomplete vectors. CPU preflight (issue #19): Baseline `N_GPU_LAYERS` Auto on a GPU-less host is seeded to `0` via `write_baseline` (through `SearchState.update_baseline`) before any Trial; `NUMA` ∈ `SEARCH_SPACE` (`None`/`distribute`/`isolate`); `N_GPU_LAYERS==0` drops the speculative knob `SPEC_DRAFT_N_MAX` from the active Search Space.
- **Agentic + coding axes**: Claw-Eval full = agentic axis; Claw quick = smoke. Direct-coding uses exactly 10 tasks per dataset and is the coding axis (required for a complete Objective Vector / `on_front`).
- **Good-enough speed path (default for new models)**: Cheap Trials first — `--validation` smoke, then TPS exploration. Complete the Objective Vector (Claw full + coding-10) on the same Fingerprint before treating a point as `on_front`. Details: [docs/discovery/good-enough-tuning.md](docs/discovery/good-enough-tuning.md). Grid CLI sweeps are unsupported; use config-only Trials.
- **No eval-score floor on membership**: Low Claw/coding must not short-circuit as `MODEL_REJECTED`. TPS Floor does not gate Pareto membership (legacy knob until removed). Day selection uses a Day TPS floor then max IQ ([ADR 0009](docs/adr/0009-day-profile-tps-floor.md)).
- **Model Markdown Links (hard)**: Never wrap GGUF model basenames in markdown links pointing to `results.tsv`. Keep model basenames formatted as inline code blocks (`model-name.gguf`).
- **"Todos os Modelos" Ranking Reports**: When the user requests a ranking of all models (Day/Night/TPS), show the full dataset table of measured models from `results.tsv` sorted by TPS/score with their Pareto status, alongside the strict Pareto-filtered pick table, so non-dominated and dominated measured models are clearly visible.
- **Skill Precedence over User Directives (hard)**: When a Skill or Slash Command is explicitly invoked by the user (e.g. `/implement`, `/code-review`, `/learn`), the skill's instructions take total precedence over passive rules. Execute the complete skill workflow to the letter from start to finish without pausing or stopping for passive user confirmations unless the skill itself asks.
- **config.py is the only mutable Baseline (local)**: Seed with `cp autoresearch/core/config.py.example autoresearch/core/config.py`. Agents and Search edit `ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS` there. File is gitignored — do not commit machine Baseline. `program.md` and harnesses stay fixed. Do not drive Trials with CLI flag soup. `.autoresearch_state.json` is visited memory only.
- **Every requested Trial edits `config.py` first**: For each user-requested test/run, set the Baseline in `config.py` (then invoke harness). Never pass the experiment knobs as CLI flags.
- **No ad-hoc eval scripts**: Do not invent one-off Python/`python -c` Trial loops. Hooks deny them. Use harness CLIs only.
- **Agent harness configs are machine-local**: `.agents/`, `.claude/`, `.cursor/`, `.pi/`, and `docs/agents/` are gitignored — skills, hooks, and harness wiring stay on the operator machine, not the clone. Hard-gates still apply where present locally. Do **not** require OS ACL (`icacls`), chmod lockdowns, or enterprise managed hooks for normal users.
- **Detect hardware before any model download**: Run `scripts/check_hardware.py` (Windows / macOS / Linux). Classify `discrete_gpu` (NVIDIA VRAM) vs `unified_memory` (Apple Silicon / no discrete NVIDIA — one RAM pool). Explain the numbers to the user and confirm. On unified memory, GGUF + context must leave clear headroom for OS/IDE — do **not** fill most of RAM (e.g. reject ~12 GB GGUF on 16 GB Mac). **Never** download blind if detection is incomplete — guide About This Mac / `sysctl`, Task Manager, `nvidia-smi`.
- **Harness host-memory hard gate**: Even if an agent skips docs, `benchmark_search` / validation / autoloop / `serve-config` reject Trials when full-GGUF host estimate exceeds `RAM − headroom` (`HOST_MEMORY_PREFLIGHT` → `MODEL_REJECTED`). No llama.cpp changes.
- **Single-load gate (issue #41)**: At most one full server intent at a time. A second full `llama-server`/SGLang start is refused while one is live on a harness port (`autoresearch/core/single_load.py` → `SingleLoadError`). A speculative draft rides the same server and never counts. Escape hatch: `model-up --allow-multi` and/or `AUTORESEARCH_ALLOW_MULTI_SERVERS=1` (intentional multi small-model experiments). Fail-open on detection tooling failure (logs + proceeds, like the orphan sweep). Stop live `model-up` before Trials, or Trials refuse instead of sweeping it away.
- **Prefer llmfit over whichllm**: Always prefer `llmfit` for candidate discovery and hardware sizing. Keep `whichllm` only as an optional fallback (fewer models, outdated, poor performance on unified RAM). Note: neither is final fit authority — especially on `unified_memory`. Local `check_hardware` + conservative headroom win; discard unsafe #1 picks and explain why.
- **Dense = no shared-memory offload**: Never partially offload dense GGUFs (layers to CPU / Windows shared GPU memory). That path freezes the whole PC. Only MoE may use expert offload (`--n-cpu-moe`). Dense must fit in **physical** VRAM on discrete GPUs, or in the **unified RAM pool** (with OS headroom) on Mac/UMA — cut `CTX_SIZE` / KV quant / drop draft or reject — never “spill and hope”.
- **MoE initial `N_CPU_MOE`**: Baseline `None` → harness auto-sets `--n-cpu-moe` to GGUF `block_count` (full expert CPU offload). Set `0` only when the MoE fits physical VRAM (full GPU). Explicit `N>0` remains a manual override.
- **llama.cpp runtime policy:** Keep only official `llama.cpp/` as a llama.cpp source clone/submodule. **Release first**: default Trials use a prebuilt release under `llama.cpp-releases/upstream/<tag>/`; build from source only to fix something urgent no release covers. Alternate llama.cpp engines or architecture forks use versioned prebuilt releases under gitignored `llama.cpp-releases/<engine>/<tag>/`; do not clone/build their source locally. Select any runtime through `AUTORESEARCH_LLAMA_CPP_ROOT` and preserve engine/tag in the Trial evidence. VITRIOL is a separate study repository, not a llama.cpp fork; model-card “VITRIOL split” still means stock `--n-cpu-moe`.
- **Virtual environment execution**: ALWAYS use the project's dedicated virtual environment (`.\venv\Scripts\python.exe` / `.\venv\Scripts\pytest.exe` on Windows, `./venv/bin/python` on Linux/macOS) for all python scripts, tests, and tool commands. NEVER run system-global `python` or `pip`, and NEVER install packages globally.
- **Never commit alias registry**: `model-up` alias names, ports, and `models/aliases/*/config.yaml` are machine-local (`/models/` gitignored). Tracked docs use GGUF basenames + benchmark scores only — not which alias the user runs.
- **Upstream/vendor runtimes are read-only**: Never modify `llama.cpp/` (the only llama.cpp clone/submodule), local `claw-eval/`, `VITRIOL/`, or extracted `llama.cpp-releases/`. UTF-8 / Windows mock issues → `PYTHONUTF8=1` (or harness-owned env injection), never patch third-party files.

## Child DOX Index
- [autoresearch/AGENTS.md](autoresearch/AGENTS.md) — Core autotuning package (config, runners, benchmarks).
- `.agents/`, `.claude/`, `.cursor/`, `.pi/`, `docs/agents/` — machine-local harness (gitignored; not in clone).
- [.pre-commit-config.yaml](.pre-commit-config.yaml) — Git pre-commit: Ruff + pytest (this repo only).
- [pyproject.toml](pyproject.toml) — Ruff / pytest config for owned Python.
- [docs/discovery/agent-shell-hard-gates.md](docs/discovery/agent-shell-hard-gates.md) — Hard-gate inventory + disable playbook (local wiring).
- [models/README.md](models/README.md) — Shared GGUF store layout (nested LM Studio + basename resolve).
- [docs/AGENTS.md](docs/AGENTS.md) — Durable documentation contract.
  - [docs/models/AGENTS.md](docs/models/AGENTS.md) — Per-model GGUF specs (architecture, quant, settings).
  - [docs/adr/AGENTS.md](docs/adr/AGENTS.md) — Architecture decision records contract + index (ADRs 0001–0009).
  - [docs/discovery/AGENTS.md](docs/discovery/AGENTS.md) — User-facing guides: model selection, good-enough speed path, whichllm/llmfit, quantization, CPU inference, agent onboarding, MTP/TPS, inference engines.
  - [docs/sessions/AGENTS.md](docs/sessions/AGENTS.md) — Empirical session logs (reproducibility evidence).
 - [docs/architecture.html](docs/architecture.html) — Interactive architecture diagram.
 - [docs/llamacpp-toolset.md](docs/llamacpp-toolset.md) — llama.cpp binary reference (build, bench, server, quantize).
- [scripts/AGENTS.md](scripts/AGENTS.md) — Operator scripts (setup, monitoring, server daemon).
- [tests/AGENTS.md](tests/AGENTS.md) — Unit and integration test suite.
- [teach/AGENTS.md](teach/AGENTS.md) — Course materials; arc contract [teach/SPEC.md](teach/SPEC.md) (Day/Night + Agent Harness).
- External sources (**agent read-only — never edit**):
 - [llama.cpp/](llama.cpp/) — upstream runtime (**git submodule**).
 - [claw-eval/](claw-eval/) — Claw-Eval harness (**local vendor tree**, gitignored; not a submodule).
 - `llama.cpp-releases/` — versioned prebuilt alternate runtimes (**gitignored**, never source clones); select with `AUTORESEARCH_LLAMA_CPP_ROOT`.
 - `VITRIOL/` — separate Randozart MoE DMA study clone (**gitignored**; not a llama.cpp fork or default Trial engine). See [docs/models/vitriol-technique.md](docs/models/vitriol-technique.md).
## Agent skills

### Issue tracker

GitHub Issues via `gh` (`allanschramm/local-model-autotuning`). Local skill config (gitignored): `docs/agents/issue-tracker.md`.

### Triage labels

Canonical roles = label strings: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. Local skill config (gitignored): `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `CONTEXT.md` + `docs/adr/`. Local skill config (gitignored): `docs/agents/domain.md`.

