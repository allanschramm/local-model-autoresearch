# Teach — Course Materials

## Purpose
Course materials to teach anyone to run local AI from scratch. Módulo 0, Semana 1 (incl. Dia 4 skills), and Semana 2 · Dias 1–3 (samplers + MCP/Context7 + guardrails) are published; Semana 2 Dia 4 remains a curriculum draft.

## Ownership
Course operator / instructors. Not part of the autotuning runtime loop.

## Local Contracts
- Purpose of repo course: Workshop AILOCAL Essentials (teach anyone to run local AI from scratch).
- Módulo 0: Conceptual foundations for absolute beginners (AI basics, Cloud vs Local, CPU/GPU/VRAM, Quantization, interactive VRAM calculator).
- Semana 1 Dias 1–3: TPS / performance — no model-quality scoring.
- Semana 1 Dia 4: agent skills (`skills.sh`, project vs global, Matt Pocock chain). `implement` uses local model; `code-review` may use a normal (cloud) model — intentional hybrid.
- Semana 2 · Dias 1–3 published (sampling / `SAMPLER_DEFAULTS`; MCP + Context7; guardrails). Dia 4 stays “Em construção”, outside progress until published. Skills stay in S1D4 — S2D2 is MCP-only.
- Student workflow: 100% local inside git checkout. Entrypoints: (1) Agent mode via `/teach` skill in CLI/IDE, (2) Browser mode via local `teach/index.html`.
- Lesson HTML nav: always link `../index.html` as **Guia**; prev/next lesson HTML only. Never link `MISSION.md` / `CURRICULUM.md` from student-facing HTML (`file://` shows raw markdown).
- Student-facing HTML surface: `index.html` + `reference/glossario.html` + the 10 lessons under `lessons/` (S0D1–S0D2, S1D1–S1D4, S2D1–S2D4). No extra cheat-sheet pages. Prefer glossary over `docs/*.md` links in lesson bodies.
- Guia CTA: `index.html` “Próximo passo” uses `TeachProgress.getNextLesson` (first incomplete in curriculum order).
- Onboarding step 5 (`verify_setup.py`) is optional until a local server listens on the selected port; the repository server helper defaults to 18080.
- Student CLI tools: `scripts/check_hardware.py` (cross-OS hardware + model-pool recommender) and `scripts/verify_setup.py` (server health & TPS benchmark).
- Agent guidance: During `/teach` sessions, agent acts as interactive tutor. Follow 5-step onboarding: (1) Check/install Python, (2) Create `venv`, (3) Install `requirements.txt`, (4) Run `check_hardware.py`, (5) Run `verify_setup.py`. Then proceed to Module 0.
- **Hardware before download (hard gate):** After step 4, read `check_hardware` output (`discrete_gpu` vs `unified_memory`, RAM/VRAM). Explain in pt-BR (unified = one pool shared with OS/IDE/browser). Confirm numbers. **Forbidden** until that is done: `hf download`, whichllm/llmfit “just download #1”, `benchmark_search` / validation with a large GGUF. whichllm/llmfit = candidates only; on unified RAM reject picks that would consume most of the pool (e.g. ~12 GB on 16 GB). If detection incomplete, guide manual checks — never download blind.
- **Leigo voice (esp. Semana 2):** Portuguese name first; English flag in parentheses. Explain effect (“maior = mais criativo”), not formulas. Chain related knobs. No logit/softmax/“guloso”.
- **Zero meta no HTML do aluno (hard gate):** NUNCA vazar para arquivos que o aluno lê
  (`lessons/*.html`, `index.html`, `reference/*.html`) pensamentos, scaffolding agente↔instrutor,
  decisões de currículo, ou rótulos de planejamento. Proibido em título, parágrafo, aside ou nav:
  “nota rápida”, “simplificado”, “não é o foco”, “desta aula”, “já vimos X”, “hoje o foco”,
  “o que cortamos”, “(rascunho)” em link de próximo, “para o leigo”, justificativa de design,
  comentários do tipo “meta”. Contraste de conceitos que ensina (ex. skill vs MCP, Claude vs Cursor
  no Windows) OK. Banner `draft-banner` / pill “Em construção” no guia = exceção de status de publicação.
  Só conteúdo que o aluno precisa aprender. Violação = editar de novo até zerar meta.
- Quizzes: hashed answers only (`assets/QUIZ-HASH.md`); options simplified in pt-BR for beginners (no LM Studio references in quizzes).
- **Completion gate:** each published lesson requires its quiz plus a practice check (`assets/progress.js`; localStorage keys `teach_quiz_pass_v1` and `teach_practice_pass_v1`). Simulated practice counts as completion but remains labeled until replaced by real practice. Preserve draft Semana 2 Dia 4 state but ignore it in published progress.
- Dense GGUF guidance: fit **physical VRAM** on discrete NVIDIA, or the **unified RAM pool** (with OS headroom) on Mac/UMA; never recommend partial dense offload/shared-memory spill. Expert offload is MoE-only.
- No GGUFs, results, or run logs in this tree.

## Work Guidance
- Prefer editing lesson HTML + `CURRICULUM.md` / `MISSION.md` together.
- Keep glossary/definitions accurate (dense fits physical VRAM; expert offload/MoE = Dia 3).
- Ensure interactive HTML elements (quizzes, calculators, troubleshooting wizards) work 100% offline in static browser view (`file://`). Quiz/progress scripts are classic (no ES modules) for that reason.

## Verification
- Open `index.html` or lesson HTML in a browser; click quizzes (client-side hash check).
- Confirm “Concluir” stays locked until quiz and practice pass; simulated practice shows its pending-real-practice label.
- Confirm lesson headers link to `index.html` (Guia) and prev/next HTML lessons — no `.md` in student nav.
- Run `node --test teach/progress.test.js` after changing `LESSON_ORDER` / quiz maps.

## Child DOX Index
- [GLOSSARY.md](GLOSSARY.md) — Canonical terms (HTML: `reference/glossario.html` only).
- [assets/sampler-sim.js](assets/sampler-sim.js) — Interactive restaurant sampler embedded in S2D1 (not a standalone page).
- [assets/mcp-sim.js](assets/mcp-sim.js) — Kitchen-bench MCP utensil toggle embedded in S2D2 (not a standalone page).
- [assets/guardrails-sim.js](assets/guardrails-sim.js) — Nightclub door sim (Deny/Allow/Pre/Post) embedded in S2D3 (not a standalone page).
- [assets/diagrams/s2d1-amostragem-restaurante.excalidraw](assets/diagrams/s2d1-amostragem-restaurante.excalidraw) — Instructor theory board for S2D1: vertical scroll, restaurant metaphor only (no config/code).
- [assets/diagrams/s2d2-mcp-cozinha.excalidraw](assets/diagrams/s2d2-mcp-cozinha.excalidraw) — Instructor theory board for S2D2: vertical scroll, kitchen metaphor only (skill=receita, MCP=utensílio; no config/code).
- [assets/diagrams/s2d3-guardrails-boate.excalidraw](assets/diagrams/s2d3-guardrails-boate.excalidraw) — Instructor theory board for S2D3: vertical scroll, nightclub metaphor only (fila=tools, lista=Deny/Allow/Ask, porta=Pre-hook, comanda=Post-hook; sandbox aside; no config/code).
- (otherwise flat under `teach/`)
