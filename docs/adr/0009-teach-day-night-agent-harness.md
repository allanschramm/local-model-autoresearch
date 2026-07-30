# ADR 0009: Teach path centers Day/Night usage and Agent Harness

**Date:** 2026-07-30
**Status:** Accepted

## Context & Problem Statement

The repository already ships a `teach/` workshop (local AI from zero) and a separate Eval / Pareto Search stack with Day/Night **selection lenses** (ADR 0006 / 0008). The teach path must produce learners who use local models in a real developer workflow: Day for supervised work, Night for overnight issue loops, with skills / MCP / guardrails configured in an agent client.

Without an explicit teach architecture, lesson order drifted (skills inside Semana 1; Day/Night absent from the student arc), and “harness” collided with the Eval harness.

## Decision

1. **`teach/SPEC.md` is the teach architecture contract** for curriculum target state, slot map, and student-facing Day/Night / Agent Harness rules.
2. **Semana 1** ends on local AI only: motors, TPS, MoE, samplers, then Day/Night as **usage** (S1D4). Skills move to **S2D1**.
3. **Semana 2** builds Agent Harness: skills → MCP → guardrails → S2D4 full application (Day + Night overnight + repo checklist + transpose).
4. **Slot ids and lesson filenames stay**; content is reassigned per `teach/SPEC.md`. Progress / quiz ids follow slots.
5. **Student HTML** uses Day/Night usage language only; Pareto selection math remains in `CONTEXT.md` / discovery docs.
6. **Agent Harness** (skills / MCP / guardrails) is in scope; **Eval Harness** is not a student objective.
7. Learners **without** capable local hardware complete the conceptual path; real local practice is optional and never gates completion.

## Consequences

### Positive

- Autodidact path matches a concrete Day/Night work workflow.
- Clear vocabulary split between Agent Harness and Eval Harness.
- Stable URLs / progress slots while themes move.

### Negative

- Existing lesson HTML and quiz maps must be rewritten to match the slot map before the guide matches the spec.
- Filenames (e.g. `s1d4-usecase-fluxo-zero.html`) may not match the new theme.

### Neutral

- ADR 0008 Day/Night selection lenses unchanged; teach adds a parallel usage framing for students.
- Workshop video order remains independent of repo lesson order.

## References

- [teach/SPEC.md](../../teach/SPEC.md)
- [teach/MISSION.md](../../teach/MISSION.md)
- [0008-day-iq-epsilon-then-tps.md](0008-day-iq-epsilon-then-tps.md)
