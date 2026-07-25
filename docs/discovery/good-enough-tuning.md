# Good-Enough Tuning (Default Speed Path)

**Goal:** find a runtime config that is *fast enough* on your rig, quickly — not the global optimum.

This is the **default path** for new agents and humans when the question is “what flags should I use for this GGUF?”. Use Claw-Eval full / coding-10 only to **validate a champion**, not to search.

## Cost–benefit rule

| Approach | When | Cost |
|---|---|---|
| **Autoloop `--mode tps`** (default here) | Few speed knobs; want “good enough” fast | Low — bench + PPL, no Claw full per Trial |
| Manual A/B via `config.py` | Closing 2–3 rivals after a plateau | Medium — one explicit Baseline at a time |
| Autoloop `--mode both` / quality overnight | After speed is acceptable | High — Claw full / agentic Val Score |

**Do not** run Claw full or coding-10 on every neighbor. That is champion validation, not search.

## Knobs that usually matter for speed

Touch these first (edit `autoresearch/core/config.py` only — no CLI Baseline soup):

- `KV_CACHE` / `KV_CACHE_K` / `KV_CACHE_V`
- `BATCH_SIZE` / `UBATCH_SIZE`
- `THREADS` / `THREADS_BATCH`
- `SPEC_DRAFT_N_MAX` (and draft model) when MTP/draft exists
- MoE: `N_CPU_MOE` (`None` = auto full expert CPU offload; `0` = full GPU only if it fits **physical** VRAM)

Leave sampler at the card recommendation through the TPS speed path. Mutate sampler only in a later quality pass.

## Default recipe

### 1. Seed Baseline

```bash
cp autoresearch/core/config.py.example autoresearch/core/config.py
# Set MODEL=…gguf basename; adjust TPS_FLOOR / CTX_SIZE / N_CPU_MOE for the rig
# Seed SAMPLER_DEFAULTS from docs/models/<card>.md Recommended settings
# (agentic/general vs coding profile) BEFORE any Trial — not template TEMP=0.4
```

### 2. Smoke that it loads

```bash
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate <model>"
```

Gates: load + TPS floor + Claw quick smoke. One model at a time. Read the latest `results.tsv` row.

### 3. Hill-climb for speed

```bash
.\venv\Scripts\python.exe autoloop.py --mode tps
```

Set `VRAM_LIMIT_MB` in `config.py`; the same Baseline value governs preflight and runtime monitoring.

- Mutates **one** engine knob per Trial.
- Optimizes TPS; perplexity acts as quality ceiling (do not keep big PPL regressions).
- Stop on local maxima / Ctrl+C. Baseline stays in `config.py`; history in `results.tsv`.

Optional: small manual A/B of 2–3 configs via `config.py` + `--validation` if you already know the candidates.

### 4. Validate the champion (quality)

Only after TPS looks acceptable:

```bash
# Edit config.py: enable INCLUDE_AGENTIC_FULL (and optional INCLUDE_CODING=10 tasks)
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "champion claw-full"
```

Compare Val Score / coding-10 in `results.tsv`. Keep or revert `config.py`.

## Anti-patterns

1. **Overnight `autoloop` default/`both` before a TPS pass** — burns hours on slow configs.
2. **CLI sweep flags** — every Trial must edit `config.py` first.
3. **Coding-10 / Claw full inside the search loop** — use for the winner only.
4. **Dense models spilling to shared GPU memory** — cut `CTX_SIZE` / KV / draft, or reject; never “spill and hope”.
5. **Parallel validations** — one Trial at a time on the shared GPU/port.

## Related

- [`agent-onboarding.md`](./agent-onboarding.md) — bootstrap for agents
- [`discover-models.md`](./discover-models.md) — pick which GGUF first (Pareto), then this guide
- [`agent-shell-hard-gates.md`](./agent-shell-hard-gates.md) — config.py-only Baseline
- `GOLDEN-RULES.md` — validation gates and harness rules
- `program.md` — Search terminology and Trial logging
