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
| Also hit “non-thinking” small chat | LFM2.5-2.6B **0.3333→0.8667**; Nemotron3-Nano-4B **0.3333→0.7333** @ 65k (same Fingerprints post-fix) |

**Do not assume** “not a thinking card” ⇒ immune. `max_tokens=512` and empty-`content` / `reasoning_content` handling can false-floor agentic runs on any model that thinks long, emits reasoning fields, or needs longer tool turns. Prefer remasuring **low Claw + decent coding** suspects, not only labeled thinking families.

## Root cause (fixed on `main`, 2026-08-08)

1. **Agentic runner ignored `reasoning_content`.** llama-server (jinja / thinking templates) puts thinking (and sometimes tool-bound text) in `reasoning_content` while `content` is empty. The coding-10 path already fell back to reasoning when `content` was empty; the Claw / agentic loop did not.
2. **History dropped reasoning.** Assistant turns were stored from `content` only → next requests looked like blank turns → schema / HTTP 400.
3. **Default `max_tokens=512` on agentic.** Thinking consumes the budget; visible answer/tool JSON never lands.

**Fix (owned harness):**

- `autoresearch/benchmarks/agentic_runner.py` — surface reasoning when `content` empty; round-trip reasoning in chat history.
- `autoresearch/runners/evaluation.py` — agentic tiers use `max_tokens ≥ 2048`; floor raised to **≥ 4096** on 2026-08-19 (1.5-class CoT exhausted 2048 mid-`<think>`; server log `n_decoded = 2048` exactly; Ornith-1.5-9B agentic 0.8000 → 0.9333 after the raise, Claw turn timeout 240 → 420 s).
- Related (same ship): Pareto merge keeps **best** remasured maximize-axes; budget bucket prefers `VRAM_LIMIT_MB` over peak; optional `AUTORESEARCH_SKIP_FREE_CLAMP=1` for WDDM free-clamp false-rejects.

Code pointer: `autoresearch/AGENTS.md` (Claw/agentic loop bullet).

## Effort vs budget vs preserve (llama.cpp levers, verified 2026-08-19)

Reasoning control on llama.cpp is **not** a single knob; "effort" and "budget" are different levers and only some are expressible:

- **Effort tiers (off / minimal / low / medium / high / xhigh / max) are NOT a llama.cpp CLI flag.** They are `chat-template-kwargs` consumed only by templates that implement them (DeepSeek-V4, gpt-oss class). The qwen35 template (Ornith, Qwen3.5/3.6/3.8 families) does **not** read `reasoning_effort`.
- **llama-server OpenAI API**: request body `reasoning_effort` — only the value `"none"` is handled (disables thinking); other values are explicitly *"model-specific and not yet handled"* (server-common.cpp). So effort tiers are **not expressible for qwen35-family models** on this stack.
- **Hard levers that DO work** (all plumbed in Baseline):
  | Lever | Baseline key | Effect |
  |---|---|---|
  | `--reasoning on/off` | `REASONING` | allow / kill thinking entirely |
  | `--reasoning-budget N` | `REASONING_BUDGET` | hard think-token cap (-1 unrestricted, 0 immediate end) |
  | `max_tokens` | harness per-turn | think + answer **pool** — the lever that truncates |
  | `--reasoning-preserve` | `REASONING_PRESERVE` | re-render older turns' think traces into the prompt; only meaningful when `GET /props` reports `chat_template_caps.supports_preserve_reasoning` (verified **true** for Ornith-1.5 GGUFs, 2026-08-19) |

**Measurement policy (capability first):** capability evals run at default effort (thinking unrestricted) with a generous `max_tokens` (floor **4096** since 2026-08-19) — measuring a model under an artificial think cap under-scores it. An **efficiency profile** (bounded thinking) is a separate choice: `REASONING_BUDGET` N + modest `max_tokens` → separate Fingerprint, documented in the card. Never set `REASONING_BUDGET` to "fix" a low capability score — raise `max_tokens` first; the 65k ctx ceiling (not a token cap) is the next real limiter on long research tasks.

**`finish_reason == "length"` caveat (verified 2026-08-19):** llama.cpp reports `length` both when `max_tokens` is exhausted AND when a tool-call turn stops at a template boundary (no EOS). The agentic loop counts only tool-call-free length stops as truncation suspects; to confirm an actual cap hit, check `n_decoded == max_tokens` in the per-run server log (`autoresearch/runners/logs/`).

## Who to remasure (Claw-full only)

**Remasure** when any of:

- Card / template enables thinking or reasoning by default (Qwen3.x thinking, Gemma-4 `enable_thinking`, Ornith, Qwythos/Mythos, KAT-Coder thinking, Nanbeige `REASONING`, Pocket think template, …).
- Pre-fix Claw looks suspiciously low for the family (especially ≤ ~0.40, or mid scores that match historical “capped” Ornith **0.60**).
- Low Claw with decent coding-10 on the same Fingerprint (e.g. historical LFM / Nemotron floors at **0.3333**) — verify with one Claw-full remasure; do not assume skill.
- Mock/server logs show empty `content` with non-empty `reasoning_content`.

**Lower priority to remasure for this bug alone:** models with already-strong Claw (granite / grug / Laguna-class ≥ ~0.65) and no empty-content pattern.

**Do not** remasure coding-10 for this bug alone — that path already handled `reasoning_content`.

Evidence: [2026-08-08 session](../sessions/2026-08-08-thinking-claw-harness-fix.md) (Ornith + LFM + Nemotron remasures). Leaderboard: [claw-eval-leaderboard.md](claw-eval-leaderboard.md).

## Operator checklist (before Claw-full on a thinking GGUF)

1. Confirm harness on `main` (or later) includes the agentic `reasoning_content` + `max_tokens≥4096` fix (floor since 2026-08-19).
2. Seed Baseline from the model card (thinking / agentic sampler profile), not leftover `config.py`. Optional engine seed: `REASONING_PRESERVE=True` only when llama-server `GET /props` reports `chat_template_caps.supports_preserve_reasoning` and the publisher wants preserved thinking for agentic (verified **true** for Ornith-1.5 GGUFs, 2026-08-19 → seeded). Default `None` (omit flag). Not an IQ Search neighbor; coding-10 does not use it.
3. Same Fingerprint / `VRAM_LIMIT_MB` / ctx you care about for Pareto merge.
4. Run claw-full via harness only (`benchmark_search.py --agentic-full …`). One Trial at a time.
5. If dense false-rejects on free VRAM while measured peaks fit budget: `AUTORESEARCH_SKIP_FREE_CLAMP=1` (runtime monitor still kills true OOM).
6. After write: status recompute merges **max** agentic/coding on the Fingerprint — stale lower claw must not win.

## Regression watch (future thinking bugs)

When adding a new agentic path, client, or mock:

- [ ] Assistant text for graders = `content` or fallback `reasoning_content` (never ignore reasoning-only turns).
- [ ] Multi-turn history preserves whatever the server needs to continue (reasoning and/or content).
- [ ] Agentic `max_tokens` large enough for think + tools + answer (floor **4096** since 2026-08-19; raise if card recommends huge think budgets).
- [ ] Tool-call / JSON extraction still strips think wrappers without poisoning tool args.
- [ ] Smoke one known thinking GGUF (e.g. Ornith UD or Qwen3.x) before trusting a new loop against the whole library.
- [ ] Treat score cliffs on thinking families as **harness suspects** first, model IQ second.

## See also

- [claw-eval-leaderboard.md](claw-eval-leaderboard.md) — ranked Claw; pre-fix thinking rows may be invalid
- [good-enough-tuning.md](good-enough-tuning.md) — Trial order
- [agentic-coding-benchmarks.md](agentic-coding-benchmarks.md) — tiers / CLI
- Model card example: [ornith-1.0-9b.md](../models/ornith-1.0-9b.md)
