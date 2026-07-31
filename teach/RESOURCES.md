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
- [Anthropic — Claude Code sandboxing](https://www.anthropic.com/engineering/claude-code-sandboxing)
  Fadiga de aprovação + isolamento/network (nota S2D3). Use for: contexto Mac/Linux.
- [Claude Code docs — permissions](https://code.claude.com/docs/en/permissions)
  Allow / Deny / Ask, ordem de avaliação. Use for: S2D3 núcleo.
- [Claude Code docs — hooks](https://code.claude.com/docs/en/hooks)
  PreToolUse / PostToolUse. Use for: S2D3 hooks.
- [Claude Code docs — CLI reference](https://code.claude.com/docs/en/cli-reference)
  `--tools`, `--allowedTools`, `--disallowedTools`. Use for: S2D3 CLI.
- [Claude Code docs — sandboxed Bash](https://code.claude.com/docs/en/sandboxing)
  Native Windows não; WSL2 sim. Use for: S2D3 nota sandbox Claude.
- [Cursor — agent sandboxing (blog)](https://cursor.com/blog/agent-sandboxing)
  Mac/Linux/Windows (Win via WSL2). Use for: S2D3 nota sandbox Cursor.
- [Cursor — Run modes](https://cursor.com/docs/agent/security/run-modes)
  Seatbelt / Landlock; sandbox + Auto-review. Use for: S2D3 nota Cursor.
- [Cursor — Hooks](https://cursor.com/docs/hooks)
  preToolUse / postToolUse. Use for: S2D3 Cursor.
- [Cursor — permissions.json](https://cursor.com/docs/reference/permissions)
  Listas de permitidos IDE / Run Mode. Use for: S2D3 Cursor IDE.
- [Cursor CLI — permissions](https://cursor.com/docs/cli/reference/permissions)
  allow/deny tokens. Use for: S2D3 Cursor CLI.

## Sabedoria (comunidades)
- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) — relatos de VRAM/TPS (filtrar marketing).
- [llama.cpp Discussions](https://github.com/ggml-org/llama.cpp/discussions) — flags e builds.
- [Upstash Discord](https://upstash.com/discord) — Context7 / MCP clients.

## Lacunas
- Link canónico LM Studio “Local Server” pode mudar de URL — validar na véspera do Dia 1.
- Motor “surpresa” do Dia 3: não documentar até o operador revelar.
- Semana 2 live: Dias 1–4 publicados (S2D4 = Day/Night + checklist + “e agora?” opcional).

## Depois do essencial (opcional)

Mapa de curiosidade no fim da S2D4. Estudar por conta — não é currículo obrigatório.

### Mais velocidade (llama.cpp)
- Speculative decoding: draft rápido propõe tokens; modelo alvo verifica em lote. MTP = uma forma (`draft-mtp`: heads embutidas ou GGUF assistant).
- [speculative-decoding-formats.md](../docs/discovery/speculative-decoding-formats.md)
- [mtp-baseline-guide.md](../docs/discovery/mtp-baseline-guide.md)
- [advanced-inference-optimizations.md](../docs/discovery/advanced-inference-optimizations.md) — CUDA Graphs, allocators, KV
- [cpu-inference-guide.md](../docs/discovery/cpu-inference-guide.md)

### Motores
- [vLLM](https://docs.vllm.ai/) · [SGLang](https://github.com/sgl-project/sglang) · [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) · [LMDeploy](https://github.com/InternLM/lmdeploy)
- [OpenVINO GenAI](../docs/discovery/openvino-genai-cpu-igpu-guide.md) · [Ollama](https://ollama.com/) · [MLX](https://github.com/ml-explore/mlx) · [Colibrì](../docs/discovery/colibri-inference-engine.md)
- [inference-engines-landscape.md](../docs/discovery/inference-engines-landscape.md)

### Agent Harnesses
- [Pi](https://pi.dev/) · [Hermes Agent](https://github.com/NousResearch/hermes-agent) · [OpenCode](https://github.com/anomalyco/opencode)
- [Aider](https://aider.chat/) · [Continue](https://continue.dev/) · [Cline](https://cline.bot/) · [Kilo Code](https://kilocode.ai/)
- [goose](https://github.com/block/goose) · [OpenHands](https://github.com/OpenHands/OpenHands)

### Meta-harnesses
- [Archon](https://archon.diy/) · [Case (workos/case)](https://github.com/workos/case) · [Sandcastle](https://github.com/mattpocock/sandcastle)
- [OpenHands Agent Canvas](https://docs.openhands.dev/) · [Claude Code dynamic workflows](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)