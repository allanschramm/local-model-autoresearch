# Dashboard (operador)

Painel **somente leitura** em `http://127.0.0.1:18765` (pt-BR), com a linguagem visual AILOCAL (tema escuro, azul de acento, Inter + JetBrains Mono; ADR 0011).

## Instalar / iniciar

Sem deps de runtime (`ui/requirements.txt` é vazio; assets estáticos ficam em `ui/static/`). Com o venv do repo:

```powershell
.\venv\Scripts\python.exe -m ui
```

```bash
./venv/bin/python -m ui
```

Abra `http://127.0.0.1:18765` no navegador. Poll a cada ~2.5s.

## O que mostra

| Painel | Fonte |
|--------|--------|
| Baseline | `autoresearch/core/config.py` (ENGINE/SAMPLER), não state JSON |
| Últimos 50 Trials | `results.tsv` via harness (`read_rows` / `is_on_front`) |
| Idle / Em execução | crescimento recente (~10s) de `autoresearch/runners/llama_server.log` |
| Tail do log | o mesmo `llama_server.log` |

## Não-objetivos

- Sem start/stop de `llama-server`, autoloop ou Search
- Sem stdout de agentes / Process Guard como sinal de “Em execução”
- Sem mutação de Baseline ou escrita em `results.tsv`
