# Teach — Course Materials

## Purpose
Autodidact course materials: local AI from zero through Day/Night Agent Harness workflow. Target architecture: [SPEC.md](SPEC.md). Mission: [MISSION.md](MISSION.md).

## Ownership
Course operator / instructors. Not part of the autotuning runtime loop.

## Local Contracts
- **SPEC is law for arc/slots:** [SPEC.md](SPEC.md). Slot ids + filenames stable; content follows the SPEC slot map.
- Módulo 0: conceptual foundations for absolute beginners.
- Semana 1: local AI only (motors, TPS, MoE, samplers, Day/Night **usage**). Skills are **not** Semana 1.
- Semana 2: Agent Harness (skills → MCP → guardrails → S2D4 full application). Each lesson ends with transpose-to-your-client framing.
- **Agent Harness** = skills / MCP / guardrails in the agent client. **Eval Harness** (Trials / `benchmark_search`) is out of scope as a student objective.
- Student workflow: 100% local checkout. Entrypoints: `/teach` tutor and `teach/index.html`.
- Lesson HTML nav: Guia = `../index.html`; prev/next lesson HTML only. Never link `MISSION.md` / `CURRICULUM.md` / `SPEC.md` from student-facing HTML.
- Student-facing HTML surface: `index.html` + `reference/glossario.html` + lessons under `lessons/`. Prefer glossary over `docs/*.md` in lesson bodies.
- Day/Night in student HTML = usage language only. Pareto selection math stays in `CONTEXT.md` / `docs/discovery/` / ADR 0008.
- Fixed Day/Night example numbers: `results.tsv` via `scripts/rank_results.py` (basenames + scores). No aliases / GGUF inventory in tracked docs.
- Learners without capable hardware: conceptual path complete; real local practice optional; never gate “Concluir”.
- **Zero meta no HTML do aluno (hard gate):** never leak planning, curriculum decisions, or agent↔instructor scaffolding into `lessons/*.html`, `index.html`, `reference/*.html`. Full rule list unchanged — teachable concept contrast OK; draft banner on unpublished lessons OK.
- **Leigo voice:** Portuguese name first; English flag in parentheses; effects not formulas.
- **Hardware before download (hard gate):** after `check_hardware`, explain `discrete_gpu` vs `unified_memory`; confirm; never blind download.
- Dense GGUF: physical VRAM or unified pool with headroom; MoE-only expert offload.
- Quizzes: hashed answers (`assets/QUIZ-HASH.md`). Completion = quiz + practice (`assets/progress.js`).
- No GGUFs, results, or run logs in this tree.

## Work Guidance
- Prefer editing lesson HTML + `CURRICULUM.md` / `MISSION.md` / `SPEC.md` together.
- After SPEC-affecting changes: learning record under `learning-records/`.
- Keep interactive HTML offline-capable (`file://`); classic scripts (no ES modules) for quiz/progress.

## Verification
- Open `index.html` or lesson HTML; quizzes and Concluir gate behave correctly.
- `node --test teach/progress.test.js` after `LESSON_ORDER` / quiz map changes.
- Student nav has no `.md` links.

## Child DOX Index
- [SPEC.md](SPEC.md) — target arc, slot map, Day/Night + Agent Harness contracts.
- [MISSION.md](MISSION.md) — why / success / constraints.
- [CURRICULUM.md](CURRICULUM.md) — human slot index.
- [GLOSSARY.md](GLOSSARY.md) — canonical terms (HTML: `reference/glossario.html`).
- [NOTES.md](NOTES.md) — instructor scratch / preferences.
- [learning-records/](learning-records/) — decision-grade teaching insights.
- [assets/sampler-sim.js](assets/sampler-sim.js) — S2D1 restaurant sampler (until content swap completes, sim follows sampler lesson slot).
- [assets/mcp-sim.js](assets/mcp-sim.js) — S2D2 kitchen MCP sim.
- [assets/guardrails-sim.js](assets/guardrails-sim.js) — S2D3 nightclub door sim.
- [lessons/s2d4-usecase-completo.html](lessons/s2d4-usecase-completo.html) — S2D4 Day/Night full application (inline flow sim).
- [assets/diagrams/](assets/diagrams/) — instructor theory boards (metaphor only).
- [ADR 0009](../docs/adr/0009-teach-day-night-agent-harness.md) — architecture decision.
