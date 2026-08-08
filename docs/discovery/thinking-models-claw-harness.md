# Thinking models × Claw / agentic harness

**Contract:** Hybrid-thinking GGUFs (`reasoning_content`, `/think`, `enable_thinking`, card `REASONING: on`) must be measured with an agentic loop that surfaces reasoning text and allocates enough `max_tokens`. Otherwise Claw-Eval **under-scores** them and rankings lie.

Ground truth for scores remains `results.tsv`. This note is the **harness failure mode** checklist so future thinking/reasoning regressions are not mistaken for “dumb models.”

## Symptoms (false weak Claw)

| Signal | What you see |
| :--- | :--- |
| Empty / near-empty graders | Tasks score 0 with little or no assistant text in mock logs |
| Mid-loop **HTTP 400** | Continuations fail after a “think-only” turn |
| Low Claw, decent coding-10 | Coding path already used `reasoning_content`; Claw did not |
| Score cliff after a harness/runtime change | Same GGUF + fingerprint; agentic drops without Baseline change |
| Classic false floor | Ornith UD @ 65k claw **0.3333** → after fix **0.9333** (14/15) |

Non-thinking models with low Claw and ok coding are usually **real agentic weakness** — do not blanket remasure them for this bug.

## Root cause (fixed on `main`, 2026-08-08)

1. **Agentic runner ignored `reasoning_content`.** llama-server (jinja / thinking templates) puts thinking (and sometimes tool-bound text) in `reasoning_content` while `content` is empty. The coding-10 path already fell back to reasoning when `content` was empty; the Claw / agentic loop did not.
2. **History dropped reasoning.** Assistant turns were stored from `content` only → next requests looked like blank turns → schema / HTTP 400.
3. **Default `max_tokens=512` on agentic.** Thinking consumes the budget; visible answer/tool JSON never lands.

**Fix (owned harness):**

- `autoresearch/benchmarks/agentic_runner.py` — surface reasoning when `content` empty; round-trip reasoning in chat history.
- `autoresearch/runners/evaluation.py` — agentic tiers use `max_tokens ≥ 2048`.
- Related (same ship): Pareto merge keeps **best** remasured maximize-axes; budget bucket prefers `VRAM_LIMIT_MB` over peak; optional `AUTORESEARCH_SKIP_FREE_CLAMP=1` for WDDM free-clamp false-rejects.

Code pointer: `autoresearch/AGENTS.md` (Claw/agentic loop bullet).

## Who to remasure (Claw-full only)

**Remasure** when any of:

- Card / template enables thinking or reasoning by default (Qwen3.x thinking, Gemma-4 `enable_thinking`, Ornith, Qwythos/Mythos, KAT-Coder thinking, Nanbeige `REASONING`, Pocket think template, …).
- Pre-fix Claw looks suspiciously low for the family (especially ≤ ~0.40, or mid scores that match historical “capped” Ornith **0.60**).
- Mock/server logs show empty `content` with non-empty `reasoning_content`.

**Skip for this bug:** clearly non-thinking chat models with completed Claw and no empty-content pattern (e.g. granite / grug / Laguna / LFM families — treat low Claw as skill unless logs contradict).

**Do not** remasure coding-10 for this bug alone — that path already handled `reasoning_content`.

Evidence session: [2026-08-08-thinking-claw-harness-fix.md](../sessions/2026-08-08-thinking-claw-harness-fix.md). Leaderboard context: [claw-eval-leaderboard.md](claw-eval-leaderboard.md).

## Operator checklist (before Claw-full on a thinking GGUF)

1. Confirm harness on `main` (or later) includes the agentic `reasoning_content` + `max_tokens≥2048` fix.
2. Seed Baseline from the model card (thinking / agentic sampler profile), not leftover `config.py`.
3. Same Fingerprint / `VRAM_LIMIT_MB` / ctx you care about for Pareto merge.
4. Run claw-full via harness only (`benchmark_search.py --agentic-full …`). One Trial at a time.
5. If dense false-rejects on free VRAM while measured peaks fit budget: `AUTORESEARCH_SKIP_FREE_CLAMP=1` (runtime monitor still kills true OOM).
6. After write: status recompute merges **max** agentic/coding on the Fingerprint — stale lower claw must not win.

## Regression watch (future thinking bugs)

When adding a new agentic path, client, or mock:

- [ ] Assistant text for graders = `content` or fallback `reasoning_content` (never ignore reasoning-only turns).
- [ ] Multi-turn history preserves whatever the server needs to continue (reasoning and/or content).
- [ ] Agentic `max_tokens` large enough for think + tools + answer (floor **2048**; raise if card recommends huge think budgets).
- [ ] Tool-call / JSON extraction still strips think wrappers without poisoning tool args.
- [ ] Smoke one known thinking GGUF (e.g. Ornith UD or Qwen3.x) before trusting a new loop against the whole library.
- [ ] Treat score cliffs on thinking families as **harness suspects** first, model IQ second.

## See also

- [claw-eval-leaderboard.md](claw-eval-leaderboard.md) — ranked Claw; pre-fix thinking rows may be invalid
- [good-enough-tuning.md](good-enough-tuning.md) — Trial order
- [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI
- Model card example: [ornith-1.0-9b.md](../models/ornith-1.0-9b.md)
