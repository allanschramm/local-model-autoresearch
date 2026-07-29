# Currículo: Workshop AILOCAL Essentials

> 📌 **Guia Interativo do Aluno:** Abra o [teach/index.html](index.html) no navegador para ver o mapa visual de lições e salvar seu progresso.
>
> **Gate:** “Concluir” só libera depois de acertar o quiz e registrar a prática da lição. A simulação conta, mas fica marcada como prática real pendente.
>
> **Referências HTML:** [Glossário](reference/glossario.html) · 10 aulas em `lessons/` (S0D1–S0D2, S1D1–S1D4, S2D1–S2D4)

## 🚀 Checklist de Preparação Inicial do Aluno (Multiplataforma)

Siga os 5 passos abaixo no terminal conforme o seu sistema operacional para preparar seu ambiente local:

### 🪟 Windows (PowerShell)
1. **Instalar Python (se necessário):** `winget install Python.Python.3.12`
2. **Criar Ambiente Virtual:** `python -m venv venv`
3. **Instalar Dependências:** `.\venv\Scripts\pip install -r requirements.txt`
4. **Diagnóstico de Hardware:** `.\venv\Scripts\python.exe scripts/check_hardware.py`
5. **Validação de TPS e Servidor (depois da Semana 1):** `.\venv\Scripts\python.exe scripts/verify_setup.py --port 18080` — **pule agora** se ainda não subiu o servidor.

### 🍎 macOS / 🐧 Linux (Terminal)
1. **Instalar Python (se necessário):** `brew install python` (Mac) / `sudo apt install python3 python3-venv` (Linux)
2. **Criar Ambiente Virtual:** `python3 -m venv venv`
3. **Instalar Dependências:** `./venv/bin/pip install -r requirements.txt`
4. **Diagnóstico de Hardware:** `./venv/bin/python scripts/check_hardware.py`
5. **Validação de TPS e Servidor (depois da Semana 1):** `./venv/bin/python scripts/verify_setup.py --port 18080` — **pule agora** se ainda não subiu o servidor.

---

## Módulo 0 — Fundação Conceitual do Zero (para leigos)

| Slot | Foco | Lição HTML |
|---|---|---|
| **Dia 1** | Como funcionam as IAs, Nuvem vs Local, Hardware (CPU/GPU/VRAM), Tokenização | [s0d1](lessons/s0d1-fundamentos-ia-hardware.html) |
| **Dia 2** | Troubleshooting e Solução de Erros Comuns (OOM, Portas, CUDA, Prompt Templates) | [s0d2](lessons/s0d2-troubleshooting-erros-comuns.html) |

---

## Semana 1 — Desempenho + skills (fechada)

| Slot | Foco | Lição HTML |
|---|---|---|
| **Dia 1** | Mini-glossário + motores de inferência, baixar/escolher modelos, API local | [s1d1](lessons/s1d1-lmstudio-avisos-motores-api.html) |
| **Dia 2** | Mesmo fluxo com **llama.cpp** + ajustar parâmetros de velocidade (TPS) | [s1d2](lessons/s1d2-llamacpp-flags-tps.html) |
| **Dia 3** | **MoE** maior que a GPU (divisão de carga / offload no llama.cpp) | [s1d3](lessons/s1d3-moe-maior-que-a-vram.html) |
| **Dia 4** | Skills: `skills.sh`, local vs global, fluxo Matt Pocock (`grill-with-docs` → `to-tickets` → `implement` local → `code-review` nuvem) | [s1d4](lessons/s1d4-usecase-fluxo-zero.html) |

---

## Semana 2 — Qualidade dos LLMs e Ferramentas

| Slot | Foco | Lição HTML | Status |
|---|---|---|---|
| **Dia 1** | Samplers: temp, top-k/p, min-p, penalidades + simulador | [s2d1](lessons/s2d1-parametros-qualidade.html) | **Publicado** |
| **Dia 2** | MCP (utensílios extras / plug-and-play) + Context7; skills = receitas no S1D4 | [s2d2](lessons/s2d2-mcp-ferramentas.html) | **Publicado** |
| **Dia 3** | Guardrails: Allow/Deny + hooks (Cursor + Claude); sandbox Claude vs Cursor no Windows | [s2d3](lessons/s2d3-sandbox-hooks-gates.html) | **Publicado** |
| **Dia 4** | Caso de uso final integrado | [s2d4](lessons/s2d4-usecase-completo.html) | Em construção |
