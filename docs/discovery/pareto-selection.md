# Pareto Selection: Picking Day/Night Points From the Front

**Scope:** how to pick 1–2 points from a non-dominated Pareto Set (axes: ctx, TPS, agentic, coding) for two usage modes — **Day** (supervised, wants speed, must not collapse intelligence) and **Night** (unsupervised long loops, wants balanced IQ with enough ctx). This is a selection-lens note, not a new membership rule.

## Membership (out of scope, one sentence)

Front membership is plain Pareto non-domination on the four axes ([ADR 0006](../adr/0006-pareto-frontier-search.md)); this note only covers which point(s) to pick *off* an already-computed front.

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

## Day: ε-constraint (IQ band) first, then max TPS

**Rule ([ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md), supersedes ADR 0007 Day):**

1. Compute `IQ_best = max(min(agentic, coding))` over the front.
2. Keep points with `min(agentic, coding) ≥ DAY_IQ_RATIO × IQ_best` (default `DAY_IQ_RATIO = 0.75`).
3. Among that IQ band, maximize TPS; ties → higher `min(agentic, coding)`, then higher ctx.

This is the **ε-constraint method**: optimize the objective you actually want to push (TPS) while turning the objective you cannot afford to sacrifice (intelligence) into a hard constraint, `min(agentic, coding) ≥ ε`. Sweeping ε traces the Pareto front, including non-convex regions a weighted sum would miss:

- Haimes, Y. Y., Lasdon, L. S., & Wismer, D. A. (1971). *On a bicriterion formulation of the problems of integrated system identification and system optimization.* IEEE Transactions on Systems, Man, and Cybernetics, SMC-1(3), 296–297. <https://doi.org/10.1109/TSMC.1971.4308298>

Relative IQ bands vs. a front's best point (rather than absolute score thresholds) are also the standard way to keep a scalar cutoff portable across fronts of different size/hardware, echoing the **knee-region** literature's finding that decision-makers reliably prefer solutions within a bounded trade-off region near the best-in-class point, not just the extreme:

- Deb, K., & Gupta, S. (2011). *Understanding knee points in bicriteria problems and their implications as preferred solution principles.* Engineering Optimization, 43(11), 1175–1204. <https://doi.org/10.1080/0305215X.2010.548863>
- Branke, J., Deb, K., Dierolf, H., & Osswald, M. (2004). *Finding knees in multi-objective optimization.* In Parallel Problem Solving from Nature (PPSN VIII), LNCS 3242, 722–731. <https://doi.org/10.1007/978-3-540-30217-9_73>

**Why pure max TPS fails ([ADR 0006](../adr/0006-pareto-frontier-search.md)'s original Day rule):** it has no quality floor at all, so it happily hands you the single fastest front point even when another point 5% slower is dramatically smarter. On this repo's front that meant picking a TPS-only survivor (`LFM2.5-8B-A1B`) over a smarter, still-fast peer (`LFM2.5-1.2B`) — exactly the "quality-sensitive task" failure mode this note is guarding against.

**Why speed-band-then-IQ ([ADR 0007](../adr/0007-day-profile-speed-band.md)'s superseded Day rule) also fails:** gating on `TPS ≥ ratio × max(TPS)` first, then maximizing quality, optimizes the *wrong* objective under constraint. It encodes "prefer quality among the models that are fast enough" — but Day's actual preference is "prefer speed, but never sacrifice much intelligence to get it." Those are not the same ordering when TPS spreads are uneven (ADR 0007 already flags this as a known weakness): a narrow, top-heavy TPS distribution can produce a speed band that still contains only mediocre-quality points, while a much smarter model sitting just below the speed cutoff — one a supervised user would happily accept as "fast enough" — gets excluded before quality is ever considered. Constraining on the axis you cannot afford to lose (intelligence) and optimizing the axis you're willing to trade (speed) is the correct ε-constraint direction for "must not collapse intelligence."

### Tuning `DAY_IQ_RATIO`

Default **0.75**: Day tolerates giving up at most a quarter of the front's best `min(agentic, coding)` in exchange for maximum available TPS. Raise to **0.8** when daytime use is more quality-sensitive (e.g., interactive coding review where small intelligence drops are costlier than a modest TPS loss) and a tighter floor is worth narrowing the speed band.

## ADR status

- [ADR 0006](../adr/0006-pareto-frontier-search.md) — Pareto Set membership (four axes, non-domination). Unaffected by this note.
- [ADR 0007](../adr/0007-day-profile-speed-band.md) — superseded Day (speed band → IQ); Night unchanged.
- [ADR 0008](../adr/0008-day-iq-epsilon-then-tps.md) — **current Day**: IQ ε-band → max TPS; Night maximin unchanged.

## See also

- [`claw-eval-leaderboard.md`](./claw-eval-leaderboard.md) — agentic axis scores.
- [`coding-leaderboard.md`](./coding-leaderboard.md) — coding axis scores.
- [`pareto-leaderboard.md`](./pareto-leaderboard.md) — global front + current Day/Night picks.
- [`../adr/`](../adr/) — architecture decision records.
