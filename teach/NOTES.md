# Teaching notes

- Curso publicado = Módulo 0 + Semana 1 + Semana 2 · Dia 1 (7 aulas). S2D2–S2D4 ainda em desenho.
- Dia 1 = **LM Studio** (visual; sem workflow neste repo). Dia 2+ = **este repo + llama.cpp**. Ponte UI↔CLI em s1d1/s1d2.
- pt-BR em prosa e opções de quiz; flags/ferramentas verbatim.
- Gabarito hasheado (`assets/QUIZ-HASH.md`). Sem chave plaintext.
- Gate: Concluir no `index.html` exige quiz + prática (`assets/progress.js`). Simulação conta, com prática real pendente.
- Guia: CTA próxima lição + glossário em `reference/glossario.html`. Sem páginas-cola soltas (flags/samplers/inferência fundidos no glossário).
- `verify_setup` = passo 5 opcional; `serve-config.py` usa porta 18080 por padrão.
- Não misturar qualidade de modelo nos Dias 1–3 da Semana 1; samplers = S2D1.
- S1D4 = skills (`skills.sh`, local vs global, Matt Pocock:
  grill-with-docs → to-tickets → implement local → code-review modelo normal).
  Exceção didática: review pode sair do local.
- Preferência do aluno (2026-07-27): quer domínio completo dos samplers via S2D1 + tutor `/teach`.
- Linguagem S2: leigo. Nome em pt-BR primeiro; flag/inglês entre parênteses.
  Explicar efeito (“maior = mais criativo”), não fórmula. Encadear knobs
  (temp baixa → preciso mas pode repetir → pen. de repetição).
  Evitar jargão: logit, softmax, guloso, nucleus, massa de probabilidade.
  Nunca vazar meta no texto do aluno (“porque a ferramenta…”, “para o leigo…”) — só o conteúdo.
- Analogia canônica S2D1: restaurante (cardápio = candidatas; temperatura = ousadia;
  K/P/Min-P = como o garçom encurta a mesa; penalidades = freios contra repetir prato).
  Manter essa metáfora em todos os knobs da aula e na cola.

