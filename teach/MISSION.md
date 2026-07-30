# Missão: Workshop AILOCAL Essentials

## Por que existe
Capacitar qualquer pessoa a **rodar IA local** e **configurar o Agent Harness** (skills, MCP, guardrails) para um fluxo de trabalho Day/Night: economizar tokens de assinatura de dia e adiantar issues de noite — depois review, teste e PR. Material autodidata neste repositório (`teach/index.html` + tutor `/teach`).

## Como é o sucesso
### Módulo 0 — Fundação Conceitual do Zero (para leigos)
- O que é um LLM de forma intuitiva (pesos no disco, contexto na memória, tokens)
- Nuvem vs Local (privacidade, custo zero por token, controle, offline)
- Hardware 101 (CPU vs GPU, RAM vs VRAM, o papel crítico da memória de vídeo)
- O milagre da Quantização (GGUF Q4 vs Q8 - analogia com compressão de imagem)
- Calculadora interativa de compatibilidade de VRAM

### Semana 1 — IA local (desempenho + amostragem + Day/Night)
- Distinguir **motor de inferência** vs **modelo** vs **quant**
- Baixar e escolher modelos GGUF; API local compatível com OpenAI
- Ajustar llama.cpp para TPS; MoE maior que a VRAM (offload)
- Samplers (temperatura, top-k/p, min-p, penalidades)
- Conceito **Day** vs **Night** como uso (não como curso de Pareto)

### Semana 2 — Agent Harness + aplicação
- Skills (`skills.sh`, fluxo Matt Pocock; `implement` no modelo local)
- MCP + Context7; guardrails (Allow/Deny/hooks; sandbox)
- Cada aula: transpor a ideia ao harness que o aluno já usa
- Dia 4: fluxo completo Day + Night overnight + checklist deste repo

Detalhe do arco e dos slots: [SPEC.md](SPEC.md).

## Regras e Restrições
- Material em **pt-BR** simples e didático para leigos
- Experiência 100% local no checkout (sem portal remoto de progresso)
- Sem rig capaz: caminho conceitual completo; prática local opcional (não trava “Concluir”)
- Quizzes com hash; progresso exige quiz + prática (simulação conta)
- Exemplo de Agent Harness no repo: Claude Code + Cursor
- Aluno configura e transpõe; não precisa criar harness do zero no curso

## Fora de Escopo
- Currículo além de Módulo 0 + Semanas 1–2
- Eval Harness / Pareto Search / Trials como objetivo do aluno
- Treinamento ou fine-tuning de modelos
- Plataforma remota de certificação / progresso
- Commitar aliases `model-up` ou inventário local de GGUFs em docs trackeados
