# Discover Models for Your Hardware

End-to-end workflow: **find models that fit your rig, filter for coding quality, run the autoloop on the Pareto-optimal pick.** Start it with `autoloop.py --profile day|night` to continue from the current Day/Night pick off `results.tsv` (issue #8).

## Step 0 — Detect local hardware (`check_hardware`)

```bash
# Windows
.\venv\Scripts\python.exe scripts\check_hardware.py

# macOS / Linux
./venv/bin/python scripts/check_hardware.py
```

Read `memory_class`: **`discrete_gpu`** (NVIDIA VRAM) vs **`unified_memory`** (Apple Silicon / no discrete NVIDIA — one RAM pool shared with the OS). Note the **model pool** GB. Explain and confirm with the user **before** any download. This script is the local fit authority.

## Step 1 — Find candidates with `whichllm` or `llmfit`

```bash
# Option A: whichllm (Python / uvx)
uvx whichllm@latest

# Option B: llmfit (Rust CLI/TUI)
llmfit
# or plan a specific model:
llmfit plan "qwen 3.5 9b"
```

Auto-detects GPU/CPU/RAM. Outputs a ranked list and memory footprint breakdown. See [`whichllm-reference.md`](./whichllm-reference.md) and [`llmfit-reference.md`](./llmfit-reference.md) for full CLI docs.

**On unified memory (Mac / UMA laptops):** whichllm and llmfit may treat system RAM as full “VRAM” and rank models that will freeze or reboot the machine. Keep their output as a **candidate list**, then **drop anything that would consume most of the unified RAM** (leave clear headroom for OS/IDE — e.g. reject ~12 GB GGUF on 16 GB Mac even though 12 < 16).

Key flags for this workflow:

| Flag / Command | Tool | Use |
|---|---|---|
| `--gpu-only` / `--fit gpu` | `whichllm` | Only models that fit FULL GPU (faster, no offload penalty) |
| `--speed usable` | `whichllm` | Hide models too slow to be practical |
| `--gpu "RTX 4090"` | `whichllm` | Simulate different hardware before buying |
| `--profile coding` | `whichllm` | Rank by coding-agent quality |
| `whichllm plan "qwen 3.6 35b"` | `whichllm` | VRAM/quant options + estimated tok/s for one model |
| `llmfit plan "qwen 3.6 35b"` | `llmfit` | Interactive memory fit & footprint planning per quant |

**Caveat**: discovery tool "scores" (e.g. intelligence index) are broad quality blends. They are **NOT** coding-agent benchmarks. Gemma 4 26B A4B ranks top on general intelligence lists but scores only 17.4% on SWE-bench Verified — bad for coding agents despite the high score.

Always cross-check coding quality on real benchmarks before committing.

## Step 2 — Cross-check coding benchmarks

Source priority for coding-agent decision-making:

1. **SWE-bench Verified** — multi-step agentic coding, closest to Claude Code / Pi Agent workload.
2. **Aider polyglot** — single-turn cross-language code editing. YAML at [aider/.../polyglot_leaderboard.yml](https://raw.githubusercontent.com/Aider-AI/aider/main/aider/website/_data/polyglot_leaderboard.yml).
3. **LiveCodeBench** — single-turn coding, contamination-free.
4. **Artificial Analysis Intelligence Index** — broad 10-benchmark intelligence (includes code but also math/science/agentic).
5. **Chatbot Arena ELO** — frozen 2025-07, useful as long-tail coverage of older models.

Cross-reference these directly. Treat whichllm output as a **candidate list**, not a score.

## Step 3 — Pareto frontier

Plot **tok/s (measured)** vs **coding quality (SWE-bench Verified or equivalent)** for each candidate that fits your hardware. Pareto-optimal models are NOT dominated — better in at least one axis without being worse in the other.

```
coding quality (SWE-bench %)
   80 ┤
77.2 ●──── Qwen3.6-27B (12-15 tok/s, dense)
      ╲
73.4 ●─────────● Qwen3.6-35B-A3B (22.5 tok/s, MoE 3B-active)
      │                ← Pareto-optimal sweet spot
   65 ●─────────────● Qwen3.5-9B-MTP (134+ tok/s, dense)
      │
   17 ●─────── Gemma-4-26B-A4B (36 tok/s whichllm est.)
   ──┴──────────────────────────── tok/s →
```

Dominance rule: model X dominates Y if X.tps ≥ Y.tps AND X.quality ≥ Y.quality (at least one strict). Drop dominated models from your shortlist.

### Why not just pick the highest score?

For Claude Code / Pi Agent loops, error cost (debugging a wrong code change) is high, latency cost is moderate, abstention is rare. Per [Zellinger & Thomson (arXiv 2507.03834, 2025)](https://arxiv.org/abs/2507.03834), expected cost = `C_a · P(abstain) + C_m · P(wrong) + C_l · latency`. For `C_m > $0.01`, use the most powerful model that runs at acceptable speed. For coding agents, **Pareto-optimal beats highest-score** because quality is roughly sigmoid in score — the gap from 65% to 73% SWE-bench matters more than 73% to 77%.

## Step 4 — Pick ONE Pareto-optimal model to autotune

Don't try to autotune every candidate. Pick the Pareto-optimal point that matches your tolerance:

| Preference | Pick | Rationale |
|---|---|---|
| Coding quality matters most | Best-quality Pareto point | Higher SWE-bench = fewer wrong code edits |
| Speed matters most | Fastest Pareto point | More iterations per minute, snappier UX |
| Balanced (default) | Sweet spot (middle of frontier) | Best quality-per-tok/s ratio |

Once picked, download the GGUF and place it where `local-model-autotuning` expects.

## Step 5 — Hand off: Pareto Set → Baseline via Profile

Do **not** burn overnight Claw full while hunting flags. Default path — the picked point joins the measured **Pareto Set** ([ADR 0006](../adr/0006-pareto-frontier-search.md)):

1. Seed the FULL Baseline (ENGINE + SAMPLER) in `autoresearch/core/config.py` from the model card's Recommended settings
2. Follow [`good-enough-tuning.md`](./good-enough-tuning.md): `--validation` → `autoloop.py --mode tps` → **complete the Objective Vector** (`--agentic-full` + coding-10 on the same Fingerprint)
3. Read status (`on_front` / `dominated` / `incomplete` / `rejected`) via `scripts/recompute_status.py` — `on_front` requires a complete, non-dominated vector; partial vectors merge by Fingerprint
4. **Baseline via Profile**: `autoloop.py --profile day|night` starts from the Day/Night pick off the `results.tsv` front, loading that row's `config_json` as the Baseline (issue #8)

```bash
# Edit autoresearch/core/config.py
MODEL = '<your-chosen-model-filename>.gguf'
# Seed SAMPLER_DEFAULTS + ENGINE_DEFAULTS from docs/models/<card>.md Recommended settings

# Optional: point at a non-default llama.cpp tree (upstream submodule is default)
export AUTORESEARCH_LLAMA_CPP_ROOT=/path/to/your/llama.cpp

cd local-model-autotuning
.\venv\Scripts\python.exe autoloop.py --mode tps --vram-limit-mb=<your-VRAM-budget-in-MB>
```

The TPS autoloop hill-climbs engine knobs, rewrites `config.py` on acceptance (engine-only vectors use the legacy scalar keep; complete vectors use `improves_set`), and appends rows to `results.tsv` (gitignored, stays local).

**Only after TPS is acceptable**, complete the Objective Vector (Claw full + coding-10) on the same Fingerprint (good-enough-tuning.md §4). Overnight `--mode both` is for quality search *after* speed, not the default first pass.

**Expected behavior (TPS mode)**:
- Cheap Trials (bench + PPL ceiling) — minutes, not Claw-full hours
- Each Trial writes 1 row to `results.tsv` with TPS / VRAM / status
- On acceptance, `config.py` rewrites with the better config
- SIGINT handler saves state — kill any time, resume later
- TPS Floor (`TPS_FLOOR` in Baseline `config.py`, default 20): configs below the floor are auto-`rejected`; lower it for large MoE on constrained VRAM

## Quick checklist

- [ ] `scripts/check_hardware.py` — memory class + pool; explain/confirm with user
- [ ] `uvx whichllm@latest` or `llmfit` — shortlist of candidates (filter by Step 0 pool)
- [ ] Cross-reference SWE-bench Verified / Aider for each
- [ ] Plot Pareto frontier on tok/s vs coding-quality axes
- [ ] Pick the Pareto-optimal point matching your preference **and** local pool
- [ ] Download GGUF, place in models/, seed FULL Baseline in config.py from the card
- [ ] Set `AUTORESEARCH_LLAMA_CPP_ROOT` if using a non-upstream llama.cpp fork
- [ ] Smoke: `benchmark_search.py --validation`
- [ ] Speed search: `autoloop.py --mode tps` ([good-enough-tuning.md](./good-enough-tuning.md))
- [ ] Complete Objective Vector: `--agentic-full` + coding-10 (same Fingerprint)
- [ ] Read status: `scripts/recompute_status.py` → `on_front` needs a complete vector ([ADR 0006](../adr/0006-pareto-frontier-search.md))
- [ ] Baseline via Profile: `autoloop.py --profile day|night` (issue #8)

## Common pitfalls

0. **Trusting whichllm/llmfit fit on unified memory** — On Mac / UMA they may recommend ~12 GB GGUFs for 16 GB unified RAM. Always run `check_hardware` first and discard oversized picks.
1. **Picking by whichllm score alone** — Gemma-4-26B-A4B ranks top but is bad at coding agents.
2. **Picking the densest model that fits** — Qwen3.6-27B fits partial but is slower than the MoE alternative.
3. **Trying to autotune every candidate** — 24h × 1 model beats 8h × 3 models (each gets a full search).
4. **Wrong `AUTORESEARCH_LLAMA_CPP_ROOT`** — autoloop silently fails to find llama-server.
5. **Not watching VRAM at startup** — use Baseline `VRAM_LIMIT_MB` (or whatever your budget) to skip configs that would OOM.
6. **Downloading before hardware is known** — agents must not `hf download` or run validation until pool is detected and explained.
7. **Treating keep/Val Score as Search truth** — membership is four-axis non-domination on a complete Objective Vector ([ADR 0006](../adr/0006-pareto-frontier-search.md)); Val Score is legacy display and `keep` a deprecated alias of `on_front`. Read `scripts/recompute_status.py` statuses instead of a scalar.

## Related docs

- [`pareto-leaderboard.md`](./pareto-leaderboard.md) — measured global Pareto Set + Day/Night on the operator host (ADR 0006/0008)
- [`claw-eval-leaderboard.md`](./claw-eval-leaderboard.md) / [`coding-leaderboard.md`](./coding-leaderboard.md) — per-axis ranks
- [`good-enough-tuning.md`](./good-enough-tuning.md) — default speed path after you pick a GGUF
- `docs/models/` — per-model GGUF specs and architecture notes
- `docs/sessions/` — empirical session logs (yours and others)
- `docs/adr/` — architecture decisions (why certain conventions exist)
- `docs/AGENTS.md` — top-level documentation index
