# Teaching notes

- Curso publicado = Módulo 0 + Semana 1 + Semana 2 · Dias 1–3 (9 aulas). S2D4 ainda em desenho.
- Dia 1 = **LM Studio** (visual; sem workflow neste repo). Dia 2+ = **este repo + llama.cpp**. Ponte UI↔CLI em s1d1/s1d2.
- pt-BR em prosa e opções de quiz; flags/ferramentas verbatim.
- Gabarito hasheado (`assets/QUIZ-HASH.md`). Sem chave plaintext.
- Gate: Concluir no `index.html` exige quiz + prática (`assets/progress.js`). Simulação conta, com prática real pendente.
- Guia: CTA próxima lição + glossário em `reference/glossario.html`. Sem páginas-cola soltas (flags/samplers/inferência fundidos no glossário).
- `verify_setup` = passo 5 opcional; `serve-config.py` usa porta 18080 por padrão.
- Não misturar qualidade de modelo nos Dias 1–3 da Semana 1; samplers = S2D1; MCP = S2D2.
- S1D4 = skills (`skills.sh`, local vs global, Matt Pocock:
  grill-with-docs → to-tickets → implement local → code-review modelo normal).
  Exceção didática: review pode sair do local.
- Preferência do aluno (2026-07-27): quer domínio completo dos samplers via S2D1 + tutor `/teach`.
- Linguagem S2: leigo. Nome em pt-BR primeiro; flag/inglês entre parênteses.
  Explicar efeito (“maior = mais criativo”), não fórmula. Encadear knobs
  (temp baixa → preciso mas pode repetir → pen. de repetição).
  Evitar jargão: logit, softmax, guloso, nucleus, massa de probabilidade.
  Nunca vazar meta no texto do aluno. Proibido: pensamentos/decisões da conversa
  de preparação, “hoje o foco…”, “já vimos X então…”, “skills ficam pro outro dia”,
  justificativa de design, aside “para o leigo”, títulos tipo “(nota rápida)”,
  “simplificado”, “desta aula”, “(rascunho)” em link de próximo. Só o conteúdo da aula.
- Analogia canônica S2D1: restaurante (cardápio = candidatas; temperatura = ousadia;
  K/P/Min-P = como o garçom encurta a mesa; penalidades = freios contra repetir prato).
  Manter essa metáfora em todos os knobs da aula e na cola.
- Analogia canônica S2D2: cozinha — agente = cozinheiro; skill = receita; harness = utensílios já na bancada;
  MCP = utensílios extras plug-and-play (Context7 = manual atualizado). Simulador = bancada liga/desliga.
  Hands-on Cursor: Settings → Plugins → Browse Marketplace → Context7 → Add (MCP vem junto).
  Alternativa Cursor: JSON manual em Tools & MCP.
  Hands-on Claude Code: `/plugin marketplace add upstash/context7` depois
  `/plugin install context7@context7-marketplace` (docs oficiais). Alternativa: `claude mcp add`…
- Analogia canônica S2D3: **boate** — fila = tools; lista da casa = Deny/Allow; segurança
  na porta = Pre-hook; comanda na saída = Post-hook; sem segurança = fadiga de Enter.
  Sandbox = **caixa de areia** (borda = SO; fora não brinca) — não misturar com a boate.
  Harness = termo do aluno (manter). “allowlist” = jargão → lista de permitidos / lista da casa;
  flag CLI `--allowedTools` = comando (manter). Sandbox: Claude Windows só WSL2;
  Cursor sem WSL2 quase não protege arquivos. CLI: `--tools` / `--allowedTools` /
  `--disallowedTools`. Ordem Claude: deny → ask → allow.
  Simulador: `assets/guardrails-sim.js`. **Publicado** (no progresso).
