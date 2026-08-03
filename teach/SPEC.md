# Spec: Learning codebase Day / Night

Contract for the `teach/` surface. Implements the workshop mission as an autodidact path inside this repository. Implementation status of each HTML lesson may lag; this file is the target architecture.

## Outcome

A learner who completes the published path can:

1. Understand local AI fundamentals (weights, hardware, quant, troubleshooting).
2. Run and tune local inference for speed (motors, llama.cpp flags, MoE offload, samplers).
3. Distinguish **Day** vs **Night** usage of local models.
4. Configure an **Agent Harness** (skills, MCP, guardrails) using this repo’s Claude Code + Cursor examples.
5. Apply Day (supervised day work, less reliance on subscription tokens) and Night (overnight issue loop → morning review, test, open PR).
6. Transpose the same Agent Harness ideas to whatever agent client they already use.

The course configures and explains the repo’s Agent Harness. It does not require the learner to build a harness from scratch. It does not teach the Python Eval / Trial harness (`benchmark_search`, Pareto Search) as a learning objective.

## Arc

| Block | Role |
|---|---|
| Módulo 0 | Conceptual foundation (leigo). |
| Semana 1 | Local AI only: motors, TPS, MoE, samplers, then Day/Night as **usage**. |
| Semana 2 | Agent Harness layers: skills → MCP → guardrails → full application (S2D4). |

Guide narrative centers on Day/Night. Workshop video order does not constrain repo lesson order.

## Slot map (target)

Slot ids and current filenames stay. Content moves to match the table.

| Slot | Filename (stable) | Target content |
|---|---|---|
| S1D4 | `lessons/s1d4-usecase-fluxo-zero.html` | Samplers + Day/Night usage (picks from the measured Pareto front) |
| S2D1 | `lessons/s2d1-parametros-qualidade.html` | Skills (`skills.sh`, Matt Pocock chain) |
| S2D2 | `lessons/s2d2-mcp-ferramentas.html` | MCP + Context7; light arc + transpose framing |
| S2D3 | `lessons/s2d3-sandbox-hooks-gates.html` | Guardrails; light arc + transpose framing |
| S2D4 | `lessons/s2d4-usecase-completo.html` | Full application: Day → Night overnight → review/test/PR + operational checklist of this repo’s Agent Harness examples + transpose |

Quizzes and `progress.js` lesson ids follow the **slot** (`s1d4`, `s2d1`, …), not the historical theme of the filename.

## Day / Night (student-facing)

In lesson HTML and the student glossary, Day/Night means **usage**:

- **Day** — supervised local model for daytime work; good enough and fast enough; reduces subscription-token spend.
- **Night** — stronger / longer-context local model for unsupervised overnight loops (e.g. open issues); morning = review, test, open PR.

The **default recipe taught is Pareto Set + Day/Night pick**: the repo measures candidates into a Pareto front — no single “best model”, each candidate is a point on four axes (ctx × TPS × agentic × coding) — and each usage mode picks its point off that front ([ADR 0006](../docs/adr/0006-pareto-frontier-search.md)). Lessons teach that concept in usage language; never a scalar “Val Score champion” as truth.

Do not put Pareto selection math (`DAY_IQ_RATIO`, maximin, TSV schema) in lesson bodies. Selection-lens detail stays in `CONTEXT.md`, [ADR 0008](../docs/adr/0008-day-iq-epsilon-then-tps.md), and `docs/discovery/`. Optional “go deeper” links from the guide are allowed.

S1D4 includes a **fixed** Day vs Night example table (basenames + scores from `results.tsv` via `scripts/rank_results.py`). No committed aliases; no on-disk GGUF inventory in tracked docs.

## Agent Harness vs Eval Harness

| Term | Meaning in `teach/` |
|---|---|
| **Agent Harness** | Skills, MCP, guardrails in the agent client (Claude Code, Cursor, Codex, …). |
| **Eval Harness** | This repo’s Trial / benchmark Python path. Out of scope as a student objective. |

Student glossary may keep short **Harness** = API client, and sharpen **Agent Harness** where Semana 2 needs it.

## Accessibility without local hardware

Learners who cannot run a local model still complete the conceptual path. Simulated practice counts toward progress. Real local practice is optional and must not gate “Concluir”.

## Semana 2 transpose thread

Each Semana 2 lesson ends with how to apply the same idea in the learner’s own Agent Harness. Repo examples = Claude Code + Cursor.

## S2D4 shape

- One full workflow: configure Agent Harness pieces already taught → Day daytime use → Night overnight on issues → morning review / test / PR.
- Operational checklist pointing at this repo’s example config (skills, MCP, guardrails / hooks).
- Simulated practice required for progress; real overnight run optional.
- Optional further-study trailhead at end (“Já sei o essencial — e agora?”): speed, engines, harnesses, meta-harnesses — not part of the required curriculum.

## Out of scope

- Curriculum beyond Módulo 0 + Semanas 1–2.
- Remote progress platform or certification product.
- Eval Harness / Pareto Search / Trials as student goals.
- Committing `model-up` alias registry or machine GGUF inventory into tracked docs.

## Related contracts

- [MISSION.md](MISSION.md) — why / success / constraints
- [CURRICULUM.md](CURRICULUM.md) — slot index for humans
- [AGENTS.md](AGENTS.md) — agent rules for editing `teach/`
- [GLOSSARY.md](GLOSSARY.md) — canonical terms
- [ADR 0009](../docs/adr/0009-teach-day-night-agent-harness.md) — architecture decision record
- Map issue: [#28](https://github.com/allanschramm/local-model-autotuning/issues/28)
