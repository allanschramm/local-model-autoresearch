# ADR 0011: Dashboard Adopts the AILOCAL Design Language

**Date:** 2026-08-07
**Status:** Accepted

## Context & Problem Statement

Issue #42 restyles the read-only operator dashboard (`ui/`, localhost 18765). The current shell is a minimal stdlib HTML dump (system font, bordered table, plain `<dl>`, black `<pre>` log) that is hard to scan during a live Search. The panels and data contracts from #23–#27 stay. The operator wants the visual identity of [AILOCAL](https://ailocal.com.br/) (dark developer UI: `#171717` base, `#339dff` accent-only blue, Inter + JetBrains Mono, subtle precision-grid) — without marketing animation.

The previous `ui/AGENTS.md` contract restricted `ui/` to stdlib-only with inline HTML. A faithful design language is not maintainable as one Python string.

## Decision

1. **Adopt the AILOCAL design tokens** (from `ailocal-website/docs/style.md`): dark neutral base `#171717`, card `#1f1f1f`, border `#2e2f2f`, muted `#262626`, muted-fg `#a3a3a3`, accent `#339dff` used for highlights/accents only (never dominant surface), status colors green `#00c758` / amber / red `#fb2c36`. Fonts Inter + JetBrains Mono bundled locally under `ui/static/fonts/` (OFL).
2. **Restraint over spectacle**: static precision-grid (128px, ~3% white lines, vignette) — no animated electric lines, no mouse-following card glow, no section icons. Only the run-state badge dot keeps its subtle pulse.
3. **Layout** (#3): fixed frosted header (wordmark `AUTOTUNING`, active model, run-state badge, "atualizado há Xs" freshness) + Baseline left rail + Trials main right + Log full-width; sticky Trials header; stat tiles for critical Baseline keys, SAMPLER as chips, remainder collapsed in `<details>`.
4. **Trials table**: 9 columns (status pill localized pt-BR, outcome with `diagnostic` tooltip, ctx, TPS, agentic, coding, memory, elapsed, truncated description + tooltip); sticky header, subtle zebra striping, mono right-aligned `tabular-nums`.
5. **Widen the `ui/` asset contract**: stdlib-only → "no external runtime deps, vanilla JS, static assets allowed under `ui/static/`". CSS as pure `ui/static/style.css` served by the existing `http.server`. No CSS/JS frameworks, no CDN, no build step.
6. **Run State / Status de exibição**: add Run State and the pt-BR Trial Status display mapping to `CONTEXT.md`.

## Consequences

### Positive
- Operator-facing hierarchy (badge primary, distinct readable sections) matches the acceptance criteria of #42.
- Coherent brand identity with the operator's AILOCAL ecosystem; zero runtime deps preserved.
- Motion stays minimal, `prefers-reduced-motion`-friendly; read-only panels stay visually read-only.

### Negative
- Bundled font files add ~200KB of static assets under `ui/static/fonts/`.
- The `ui/AGENTS.md` "stdlib-only" phrasing must be reworded (assets allowed); JS remains framework-free.

### Neutral
- `/api/status` payload shape unchanged (freshness is computed client-side); Trials/data contracts from #23–#27 untouched.
- Wireframe/design capture lives in the Traycer ticket artifact for issue #42.

## References
- [AILOCAL style guide](https://ailocal.com.br/) / `ailocal-website/docs/style.md` (design tokens source)
- [ui/AGENTS.md](../../ui/AGENTS.md)
- [CONTEXT.md](../../CONTEXT.md)
