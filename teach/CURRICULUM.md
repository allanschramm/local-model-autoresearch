# Currículo: Workshop AILOCAL Essentials

> **Guia:** [teach/index.html](index.html) · **Contrato de arco:** [SPEC.md](SPEC.md) · **Glossário:** [reference/glossario.html](reference/glossario.html)
>
> **Gate:** “Concluir” exige quiz + prática. Simulação conta; prática real local é opcional.
>
> **Nota de implementação:** slots/filenames estáveis; conteúdo segue o SPEC. Swap S1D4/S2D1 concluído (samplers → S1D4; skills → S2D1).

## Checklist de preparação (multiplataforma)

### Windows (PowerShell)
1. `winget install Python.Python.3.12` (se necessário)
2. `python -m venv venv`
3. `.\venv\Scripts\pip install -r requirements.txt`
4. `.\venv\Scripts\python.exe scripts/check_hardware.py`
5. `.\venv\Scripts\python.exe scripts/verify_setup.py --port 18080` — pule se ainda não subiu o servidor

### macOS / Linux
1. Instalar Python 3 (`brew` / `apt` + venv)
2. `python3 -m venv venv`
3. `./venv/bin/pip install -r requirements.txt`
4. `./venv/bin/python scripts/check_hardware.py`
5. `./venv/bin/python scripts/verify_setup.py --port 18080` — pule se ainda não subiu o servidor

---

## Módulo 0 — Fundação Conceitual do Zero

| Slot | Foco | Lição HTML |
|---|---|---|
| **Dia 1** | Como funcionam as IAs, Nuvem vs Local, Hardware, Tokenização | [s0d1](lessons/s0d1-fundamentos-ia-hardware.html) |
| **Dia 2** | Troubleshooting (OOM, portas, CUDA, prompt templates) | [s0d2](lessons/s0d2-troubleshooting-erros-comuns.html) |

---

## Semana 1 — IA local (desempenho + amostragem + Day/Night)

| Slot | Foco (alvo SPEC) | Lição HTML |
|---|---|---|
| **Dia 1** | Motores, baixar/escolher modelos, API local | [s1d1](lessons/s1d1-lmstudio-avisos-motores-api.html) |
| **Dia 2** | llama.cpp + flags TPS | [s1d2](lessons/s1d2-llamacpp-flags-tps.html) |
| **Dia 3** | MoE maior que a GPU (offload) | [s1d3](lessons/s1d3-moe-maior-que-a-vram.html) |
| **Dia 4** | Samplers + Day/Night (uso): picks da fronteira Pareto, tabela fixa | [s1d4](lessons/s1d4-usecase-fluxo-zero.html) | Publicado (swap + Day/Night uso) |

---

## Semana 2 — Agent Harness + aplicação

| Slot | Foco (alvo SPEC) | Lição HTML | Status |
|---|---|---|---|
| **Dia 1** | Skills (`skills.sh`, Matt Pocock) + transpor | [s2d1](lessons/s2d1-parametros-qualidade.html) | Publicado (swap) |
| **Dia 2** | MCP + Context7; arco + transpor | [s2d2](lessons/s2d2-mcp-ferramentas.html) | Publicado |
| **Dia 3** | Guardrails + hooks; arco + transpor | [s2d3](lessons/s2d3-sandbox-hooks-gates.html) | Publicado |
| **Dia 4** | Aplicação Day/Night + checklist do repo | [s2d4](lessons/s2d4-usecase-completo.html) | Publicado |
