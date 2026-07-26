# ADR 0006: Multi-Objective Pareto Frontier Search

**Date:** 2026-07-25
**Status:** Accepted. **Day Usage Profile pick superseded by [0007](0007-day-profile-speed-band.md)** (2026-07-26). Night + Pareto Set membership unchanged.
**Supersedes in part:** [0004](0004-agentic-first-search.md) (canonical scalar Val Score + keep/discard as Search truth). Agentic + coding benchmarks remain the intelligence measurements; Baseline location remains [0005](0005-config-py-mutable-baseline.md).

## Context & Problem Statement

The repo teaches people to find the best local model for *their* rig. Real use is multi-objective: large configured context (long agent loops), throughput (daytime supervised chat), agentic tool-use (Claw), and coding skill (coding-10). A single Val Score and one Baseline champion collapse those tradeoffs. Historical “Pareto Tie-Breaker” only broke exact Val Score ties — it was not a frontier.

Day vs night use differs: supervised daytime wants speed; unsupervised night `/loop` wants balanced intelligence and enough context. One TPS Floor on keep cannot serve both.

## Decision

1. **Pareto Set** is the keep surface: non-dominated Trials under four maximize axes — configured `CTX_SIZE`, TPS, agentic (Claw-Eval full), coding (coding-10).
2. **Search / Neighbors stay per model.** The **global** frontier (union across models) is ranked for a hardware+budget identity so users pick a model for their rig.
3. **Baseline** is the Neighbor origin (active point), not the sole champion. Profile **Day** / **Night** selects that origin (manual override allowed): Day → max TPS on the model front *(superseded by [0007](0007-day-profile-speed-band.md): speed band then max `min(agentic, coding)`)*; Night → among points with `CTX_SIZE ≥ NIGHT_CTX_FLOOR` (default 65536), max `min(agentic, coding)`; if none qualify, fallback to max ctx with a complete vector.
4. **Complete vector required for `on_front`.** Partial Trials (coding-only, claw-only, …) stay `incomplete` and **merge** into the same **Fingerprint** (full `ENGINE_DEFAULTS` + `SAMPLER_DEFAULTS`) when axes arrive later.
5. **Status vocabulary:** `on_front` | `dominated` | `incomplete` | `rejected`. Do not overload `keep`/`discard` with new meaning (`keep` may remain a deprecated alias of `on_front` during migration).
6. **No TPS Floor on frontier membership.** TPS remains an axis. Day/Night apply throughput (and Night ctx floor) only when *selecting* a point. Legacy `TPS_FLOOR` may remain for smoke/tooling until removed.
7. **Ship cut:** Phase 0 (done 2026-07-25) = domain docs (`CONTEXT.md` + this ADR + AGENTS preferences). Phase 1 = frontier nucleus (domination, fingerprint merge, status, leaderboard/TSV). Phase 2 = Search/autoloop profile pick + honest peak preflight / dynamic headroom. No big-bang eval rewrite.

## Consequences

### Positive
- Teaching and leaderboards show tradeoffs instead of a fake single winner.
- Day/Night map to real workflows without splitting measurement into two frontiers.
- Partial cheap Trials still contribute via fingerprint merge.

### Negative
- Breaking change for `keep`/`discard` consumers and for “beat Baseline Val Score” Search logic.
- Four-axis completeness makes a point slower/more expensive to land on the front than a scalar keep.

### Neutral
- Claw full and coding-10 remain the intelligence proxies; they are axes, not a blended “intelligence” score.
- `Val Score` becomes legacy display/compat, not Search truth (see glossary).

## Considered Options (rejected)

- Single Baseline + scalar Val Score with richer tie-breakers — hides Day/Night and ctx vs smart tradeoffs.
- Blending agentic+coding into one “intelligence” axis — recreates Gemma-vs-SWE style lies.
- Separate day/night Pareto Sets — duplicates measurement; teaching worse.
- Hard TPS Floor on keep — fights night-loop models that are slow but capable.
- Night pick = max ctx first — can prefer huge-context weak models over balanced ones that still clear a ctx floor.
- Day pick = pure max TPS — see [0007](0007-day-profile-speed-band.md) (superseding correction).
