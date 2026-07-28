# IA local — Recursos

## Conhecimento

### Semana 1 — Performance
- [LM Studio](https://lmstudio.ai/)
  App local com UI + servidor API. Use for: Dia 1 (ver o fluxo antes do CLI).
- [LM Studio — Local Server / OpenAI compatible](https://lmstudio.ai/docs)
  Docs oficiais do server. Use for: expor modelo via API para qualquer harness.
- [Hugging Face Hub — Download](https://huggingface.co/docs/huggingface_hub/guides/download)
  `hf download`. Use for: onde baixar GGUF.
- [llama.cpp (ggml-org)](https://github.com/ggml-org/llama.cpp)
  Motor CLI/server. Use for: Dia 2+.
- [docs/llamacpp-toolset.md](../docs/llamacpp-toolset.md)
  Flags neste checkout. Use for: tunar TPS no Dia 2.
- [docs/discovery/quantization-cascade.md](../docs/discovery/quantization-cascade.md)
  Escolha de quant vs VRAM. Use for: escolher modelo (performance).
- [docs/models/vitriol-technique.md](../docs/models/vitriol-technique.md)
  MoE / offload. Use for: Dia 3.
- [skills.sh](https://skills.sh/)
  Catálogo de skills para agentes. Use for: S1D4 achar pacotes.
- [mattpocock/skills](https://github.com/mattpocock/skills)
  Pacote foco: grill-with-docs, to-tickets, implement, code-review. Use for: S1D4 fluxo.

### Semana 2 — Qualidade
- [llama.cpp CLI — Sampling params](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
  Flags oficiais (`--temp`, `--top-k/p`, `--min-p`, penalties) e ordem dos samplers. Use for: S2D1 fonte primária.
- [Nguyen et al. — Min-p sampling (arXiv:2407.01082)](https://arxiv.org/abs/2407.01082)
  Corte relativo à chance da líder. Use for: explicar Min-P vs Top-P na S2D1.
- [LLM Sampling Parameters Guide — Sam McLeod](https://smcleod.net/2025/04/llm-sampling-parameters-guide/)
  Guia prático alinhado a llama.cpp (defaults, presence vs frequency, troubleshooting). Use for: intução e perfis.
- Cards/sampling em `docs/models/` — Use for: semear `SAMPLER_DEFAULTS` do publisher (prática real S2D1).
- [What is MCP?](https://modelcontextprotocol.io/docs/getting-started/intro)
  Definição oficial (USB-C para apps de IA). Use for: S2D2 conceito.
- [Context7 (Upstash)](https://github.com/upstash/context7)
  Servidor MCP de documentação por versão. Use for: S2D2 prática.
- [Context7 — all clients](https://context7.com/docs/resources/all-clients)
  JSON Cursor + `claude mcp add`. Use for: instalação S2D2.
- [context7.com/dashboard](https://context7.com/dashboard)
  API key. Use for: rate limit na aula ao vivo.

## Sabedoria (comunidades)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — relatos de VRAM/TPS (filtrar marketing).
- [llama.cpp Discussions](https://github.com/ggml-org/llama.cpp/discussions) — flags e builds.
- [Upstash Discord](https://upstash.com/discord) — Context7 / MCP clients.

## Lacunas
- Link canónico LM Studio “Local Server” pode mudar de URL — validar na véspera do Dia 1.
- Motor “surpresa” do Dia 3: não documentar até o operador revelar.
- Semana 2 live: Dias 3–4 em aberto; Dias 1–2 (samplers + MCP) publicados.
