# Glossário — Workshop AILOCAL Essentials

Canonical terms for this teaching workspace. Student HTML: [reference/glossario.html](reference/glossario.html).
Only student HTML reference page besides the lesson set + `index.html`.

## Terms

**Inferência**:
Gerar texto com pesos já treinados. Não atualiza o modelo. Este curso = só inferência.

**Motor**:
Programa que carrega pesos e gera tokens (ex.: llama-server). Não é o arquivo do modelo.

**Modelo**:
Pesos no disco, tipicamente um `.gguf`.

**Quant**:
Compressão dos pesos (Q4/Q5/Q8…). Menos bits → menos VRAM; TPS costuma subir.

**VRAM**:
Memória da GPU onde pesos + KV competem. Dense deve caber na VRAM física neste curso.

**KV / contexto**:
Memória do histórico; controlada em grande parte por `-c`.

**TPS**:
Tokens por segundo — métrica de velocidade.

**API local**:
HTTP na máquina (ex.: `127.0.0.1:18080`), em geral formato OpenAI-compatível.

**Harness**:
Cliente da API (script, app, agente, IDE).

**Skill**:
Pacote de instruções (`SKILL.md`) que ensina o agente a seguir um fluxo. Catálogo: [skills.sh](https://skills.sh/). Semana 1 · Dia 4.

**Skill local (projeto)**:
Instalação padrão do `npx skills add` (sem `-g`). Vive no repo.

**Skill global (usuário)**:
`npx skills add … -g`. Vive na conta do usuário; vale em qualquer projeto.

**Offload**:
Parte do modelo fora da VRAM. Dense: evitar. MoE: ferramenta do Dia 3.

**OOM**:
Memória esgotada. Em denso, cortar contexto/KV, remover draft ou escolher GGUF menor — nunca “spill and hope”.

**Amostragem (sampling)**:
Escolher a próxima palavra (restaurante). Ordem no motor: penalties → top_k → top_p → min_p → temperature → sorteio. Semana 2 · Dia 1.

**MCP**:
Encaixe plug-and-play de utensílios extras pro agente (cozinheiro). O harness já tem faca/panela; MCP adiciona ferramentas de fora (docs, APIs…). Semana 2 · Dia 2. Skill = receita (S1D4).

**Context7**:
Utensílio MCP: manual atualizado (docs por versão) das bibliotecas. Hands-on S2D2 (Cursor / Claude Code).

**Sandbox (caixa de areia)**:
Brinca à vontade dentro da borda; fora, o SO impede (arquivos + rede). Nome vem da caixa de areia de criança. Claude: Mac/Linux; no Windows só dentro do WSL2. Cursor: Mac/Linux; no Windows melhor com WSL2 — sem WSL2 quase não protege os arquivos do PC. Semana 2 · Dia 3.

**Guardrail**:
Lista da casa + segurança na porta das tools (Allow / Deny / Ask + hooks). Metáfora: boate. Cursor e Claude Code. Semana 2 · Dia 3.

**Fadiga de aprovação**:
Aprovar no reflexo (Enter) sem ler. Motivo dos guardrails. Semana 2 · Dia 3.

**Allow / Deny / Ask**:
Permitir sem prompt / bloquear / sempre perguntar. Claude: `permissions` em settings; ordem deny→ask→allow. Cursor IDE: lista de permitidos; Cursor CLI: `permissions.allow`/`deny`.

**Hook (pré / pós)**:
Script antes (pode bloquear) ou depois (registra / avisa) da tool. Claude: `PreToolUse` / `PostToolUse`. Cursor: `preToolUse` / `postToolUse`. Ex. repo: `.claude/hooks/`.

**`--tools` / `--allowedTools` / `--disallowedTools`**:
Claude CLI: lista da sessão / rodar sem perguntar / deny. Não confundir `--tools` com `--allowedTools`.

**Temperatura / Top-K / Top-P / Min-P / penalidades**:
Ver tabela e defs no HTML do glossário (valores de “desligar” inclusos).

**Flags TPS (`-m`, `-ngl`, `-c`, `-fa`, `-ctk`/`-ctv`, `--n-cpu-moe`, …)**:
Cola no glossário HTML; detalhe nas aulas S1D2 / S1D3.
