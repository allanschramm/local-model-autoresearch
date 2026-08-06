# Pareto Selection: Picking Day/Night Points From the Front

**Scope:** how to pick 1–2 points from a non-dominated Pareto Set (axes: ctx, TPS, agentic, coding) for two usage modes — **Day** (supervised, wants speed, must not collapse intelligence) and **Night** (unsupervised long loops, wants balanced IQ with enough ctx). This is a selection-lens note, not a new membership rule.

## Membership (out of scope, one sentence)

Front membership is plain Pareto non-domination on the four axes ([ADR 0006](../adr/0006-pareto-frontier-search.md)); this note only covers which point(s) to pick *off* an already-computed front.

## Status recompute (issue #5)

Stored statuses (`results.tsv`) are refreshed after every Trial write and via `scripts/recompute_status.py`: a new `on_front` point demotes rows it dominates to `dominated`. Each row's status derives from its (hardware+budget bucket, fingerprint) merged vector ([`autoresearch/core/recompute.py`](../../autoresearch/core/recompute.py)); incomplete and rejected rows never compete; rows without a `config_json` fingerprint (legacy keep/discard) are left untouched. The recompute is pure and idempotent. The canonical stored status is the global-by-bucket front across models; a per-model lens is available read-only (`scripts/recompute_status.py --scope model`, no rewrite).

## Agent contract: Trial-a-Trial workflow (issue #9)

Agent-facing step list for driving a model toward the front one Trial at a time, no autoloop required. Same rules that autoloop follows, spelled out for a manual loop.

> **Autoloop shortcut (issue #8):** `autoloop.py --profile day|night` replaces steps 1–3's Baseline start — it picks the Day/Night point off the `results.tsv` front (`pick_day`/`pick_night`), loads that row's `config_json` as the Baseline, and runs rounds from there. Neighbor acceptance inside the loop is the same Pareto rule as step 4 (`improves_set`); the legacy scalar keep only applies to incomplete vectors (engine-only / quality-only modes). `--dry-run` prints the plan (pick, baseline, neighbors) without running benchmarks.

1. **Profile pick** — choose the job profile the Trial must serve (agentic/general vs coding); before the first Trial on a model, seed `SAMPLER_DEFAULTS` from the model card's Recommended settings for that profile.
2. **Edit Baseline** — set the knobs in `autoresearch/core/config.py` (`ENGINE_DEFAULTS` / `SAMPLER_DEFAULTS`), never as CLI flags. `config.py` is the only mutable Baseline; harnesses and `program.md` stay fixed.
3. **Run the Trial** — invoke a harness (validation smoke → TPS exploration → complete the Objective Vector: Claw full + coding-10 on the same Fingerprint). A Trial is one `results.tsv` row keyed by (hardware+budget bucket, Fingerprint).
4. **Read the status** — `scripts/recompute_status.py` refreshes statuses after every Trial write; a point with a complete, non-dominated vector is what can become `on_front` ([ADR 0006](../adr/0006-pareto-frontier-search.md), [CONTEXT.md](../../CONTEXT.md)).
5. **Merge by Fingerprint** — rows sharing a Fingerprint are the same point; later Trials on it complete the vector, they do not spawn new points. Different GGUF basenames = different points.

### States for agents

- **incomplete** — vector missing axes (e.g. no Claw full or no coding-10 yet). Normal mid-work state, not a failure.
- **on_front** — complete vector and non-dominated on ctx × TPS × agentic × coding; a candidate for the Day/Night pick below.
- **dominated** — complete vector but another front point beats it on every axis; recompute demotes rows a new `on_front` dominates.
- **rejected** — a preflight or harness gate refused the Trial (e.g. host-memory preflight). Does not compete. Low eval scores are never a rejection reason.

Keep/discard is retired as Search truth: membership comes only from the four-axis non-domination test. A weak but measured point stays as `on_front` or `dominated` — data stays valuable.

## Why a scalar pick, not full multi-criteria decision-making (MCDM)

Picking a single point from a Pareto Set is the classic **a posteriori** MCDM problem: generate the whole front first, then apply a decision-maker preference to choose one point ("generate-first, choose-later"). The literature offers a spectrum from one-line scalarizations to full interactive optimization. This repo's front is small (roughly one row per candidate model/config, teachable as a leaderboard table), so the right tool is the cheapest scalarization that is still theoretically grounded — not a library.

**Rejected as overkill for this repo:**
- Full **interactive EMO** (decision-maker-in-the-loop weight elicitation, e.g. interactive Tchebycheff procedures) — needs a live human steering weights per query; this repo wants a fixed, teachable default.
- **Surrogate-assisted knee search** (training a model to predict knee regions, Bayesian optimization over front shape) — solves a problem (huge many-objective fronts) this repo doesn't have.
- **Heavy MCDM libraries** (`pymoo` decision-making module, full TOPSIS/AHP pipelines) — adds a dependency and a black box for a front anyone can read off `results.tsv` by eye.

Two closed-form rules — **maximin/Chebyshev** for Night, **ε-constraint** for Day — cover both usage modes with nothing beyond `min()`, a ratio, and a threshold.

## Night: maximin on intelligence with a ctx floor

**Rule (ADR 0006 Night; restated in [0008](../adr/0008-day-iq-epsilon-then-tps.md)):** among front points with `CTX_SIZE ≥ NIGHT_CTX_FLOOR`, maximize `min(agentic, coding)`. Fallback: max ctx with a complete vector.

**Why this is theoretically sound, not ad hoc:** maximizing the minimum of two objectives is the **maximin** (Rawlsian) criterion, which is the special case of a **weighted Tchebycheff / compromise-programming** scalarization with equal weights and an ideal point at each axis's best value. Minimizing the worst-case shortfall from an ideal point (Tchebycheff metric) with equal weights is algebraically the same as maximizing the smaller of the two normalized objectives. Weighted Tchebycheff scalarizations are guaranteed to land on the Pareto-optimal (non-dominated) set, including non-convex regions of the front where a linear weighted sum cannot reach — a property that motivated their use for sampling efficient frontiers:

- Steuer, R. E., & Choo, E. U. (1983). *An interactive weighted Tchebycheff procedure for multiple objective programming.* Mathematical Programming, 26(3), 326–344. <https://doi.org/10.1007/BF02591870>

Night's ctx floor plus maximin is exactly this: floor filters the feasible region (a hard constraint, same shape as ε-constraint below), then maximin/Chebyshev balances the two remaining quality axes without needing arbitrary weights. No relative-IQ-band or knee search is needed here because Night's stated preference is already "balanced," which maximin directly encodes.

## Day: TPS Floor (>= 50.0 TPS) first, then max IQ

**Rule ([ADR 0009](../adr/0009-day-profile-tps-floor.md), supersedes ADR 0008 Day):**

1. Filter Pareto Set points clearing `TPS >= DAY_TPS_FLOOR` (default `DAY_TPS_FLOOR = 50.0`).
2. Among points clearing the TPS floor, maximize `min(agentic, coding)`; ties → higher TPS, then higher ctx.
3. **Fallback:** If no point clears `DAY_TPS_FLOOR`, sort by TPS descending.

This guarantees daytime interactive users get snappy throughput (>= 50.0 TPS) while selecting the smartest available model among those fast enough.

## ADR status

- [ADR 0006](../adr/0006-pareto-frontier-search.md) — Pareto Set membership (four axes, non-domination). Unaffected by this note.
- [ADR 0007](../adr/0007-day-profile-speed-band.md) — superseded Day (speed band → IQ); Night unchanged.
- [ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md) — superseded Day (IQ ratio → max TPS); Night maximin unchanged.
- [ADR 0009](../adr/0009-day-profile-tps-floor.md) — **current Day**: TPS floor (`TPS ≥ 50.0`) → max IQ; Night maximin unchanged.

## See also

- [`claw-eval-leaderboard.md`](./claw-eval-leaderboard.md) — agentic axis scores.
- [`coding-leaderboard.md`](./coding-leaderboard.md) — coding axis scores.
- [`pareto-leaderboard.md`](./pareto-leaderboard.md) — global front + current Day/Night picks.
- [`../adr/`](../adr/) — architecture decision records.
