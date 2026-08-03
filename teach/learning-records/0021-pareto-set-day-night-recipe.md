# Learning record 0021 — Receita padrão: Pareto Set + Day/Night pick

`teach/` passa a ensinar a receita padrão do repositório (issue #11): **Pareto Set + escolha Day/Night** — medição em fronteira Pareto (nenhum vencedor único; cada candidato é um ponto em ctx × TPS × agente × código; Day e Night escolhem pontos diferentes), nunca campeão por nota scalar / Val Score como verdade. Conceito em linguagem de uso, sem matemática de seleção.

Mudanças: S1D4 (tabela fixa + texto "fronteira Pareto" + go-deeper da receita), S2D4 (nota de uso → pick por modo), glossário HTML + GLOSSARY.md (termo "Fronteira Pareto"), SPEC.md (default recipe + conceito em escopo; matemática continua fora), AGENTS.md (linha Day/Night), index.html/CURRICULUM.md (descrições), RESOURCES.md (links receita). Docs correlatos já atualizados: [good-enough-tuning.md](../../docs/discovery/good-enough-tuning.md) e [discover-models.md](../../docs/discovery/discover-models.md).

Contratos: [SPEC.md](../SPEC.md) · [AGENTS.md](../AGENTS.md) · [ADR 0006](../../docs/adr/0006-pareto-frontier-search.md) · [ADR 0008](../../docs/adr/0008-day-iq-epsilon-then-tps.md).

**Implications:** lições ensinam o conceito (não a matemática); glossário ganha termo; Eval Harness / Trials continuam fora do objetivo do aluno.
