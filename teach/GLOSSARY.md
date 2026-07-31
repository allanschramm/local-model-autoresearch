# Glossário — Workshop AILOCAL Essentials

Canonical terms for this teaching workspace. Student HTML: [reference/glossario.html](reference/glossario.html).
Only student HTML reference page besides the lesson set + `index.html`.

Order follows first appearance along the published lesson path (`LESSON_ORDER`). No “Semana N / Dia N” pointers — the sequence is the map.

Keep defs rich enough to teach the idea (what it is + what it does + why it matters). Keep MD and HTML in sync.

## Terms (lesson order)

### Fundamentos

**LLM**:
Modelo de linguagem grande. Não “pensa” como gente: prevê o próximo pedaço de texto a partir do que já viu. O curso ensina a rodar isso na sua máquina.

**Inferência**:
Usar o modelo para gerar texto com pesos já treinados. Os pesos não aprendem nada novo nessa hora — só respondem. Este curso = só inferência (não treino).

**Token**:
Unidade mínima que o modelo “mastiga”. Não é letra nem palavra inteira: o texto é fatiado em pedaços (em média ~4 caracteres / ~0,75 palavra). Contexto e velocidade se contam em tokens.

**Pesos**:
Os números do modelo no disco — o “conhecimento congelado”. Sem pesos não há modelo; sem motor, pesos são só arquivo.

**Motor**:
Programa que carrega os pesos na memória e gera tokens um a um (ex.: `llama-server`, LM Studio). É o motor; o GGUF é a peça que ele gira.

**Modelo**:
O arquivo de pesos que você escolhe (quase sempre um `.gguf`). Trocar de modelo = trocar esse arquivo (e, em geral, as recomendações de amostragem).

**Quant (quantização)**:
Comprimir os pesos com menos bits (Q4, Q5, Q8…). Cabe em menos VRAM e costuma gerar mais rápido; qualidade pode cair se o quant for agressivo demais pro seu uso.

**GGUF**:
Formato de arquivo feito para motores tipo llama.cpp: pesos + metadados num pacote que o motor sabe carregar.

**CPU / GPU**:
CPU = processador geral (bom em sequência, ruim em bilhões de contas em paralelo). GPU = milhares de núcleos pequenos — onde a IA local costuma rodar de verdade. Sem GPU capaz, dá para tentar na CPU, mas o TPS cai.

**VRAM**:
Memória dedicada da GPU. Pesos do modelo e memória de conversa (KV) competem por ela. Modelo denso neste curso deve caber na VRAM física — “derramar” para RAM compartilhada congela o PC.

**RAM**:
Memória do sistema, usada pela CPU e pelo SO. No MoE, experts em offload podem viver na RAM; isso não é velocidade de graça — é troca de VRAM por lentidão.

**KV / contexto**:
Memória do histórico da conversa (o que já foi dito). Mais contexto (`-c`) → mais KV → mais memória. Estourar contexto ou KV é caminho clássico de OOM.

**TPS**:
Tokens por segundo — quão rápido o motor “fala”. Mede velocidade, não inteligência.

**Dense (denso)**:
Arquitetura em que (quase) todo o conhecimento é consultado a cada token. Potente, mas pesado: tem que caber na VRAM. Sem offload parcial “pra ver se cola”.

**Hugging Face / `hf`**:
Hub de modelos na internet. A CLI `hf` baixa GGUFs de forma previsível para `models/`. Prefira isso a download “no browser e torcer”.

**OOM**:
Out of Memory — VRAM ou RAM acabou. Sintoma: crash, freeze, processo morto. Em denso: corte contexto/KV, tire draft, escolha GGUF menor. Nunca “spill and hope”.

**Draft / MTP**:
Arquivo ou trilha auxiliar de especulação para tentar acelerar a geração. Fora do escopo básico de TPS — saiba que existe; não comece por aí.

**API local**:
Porta HTTP na sua máquina (ex.: `127.0.0.1:18080`) onde o motor entrega completions, em geral no formato OpenAI. O harness aponta para cá — a nuvem não precisa existir.

### Motor na prática e Baseline

**LM Studio**:
App com tela para baixar/carregar GGUF e subir um servidor local. Bom para ver o fluxo motor → modelo → API antes do terminal.

**llama.cpp**:
Motor open-source (CLI + `llama-server`) usado no trabalho prático de flags, Baseline e medição de TPS.

**Harness**:
O “corpo” do modelo no mundo. Sozinho, o LLM só gera texto numa API. O harness é o que dá mãos e olhos: conversa, tools, arquivos, terminal, browser, regras. Pode ser um script, um app, uma IDE ou um agente completo. Sem harness, o modelo fala no vazio; com harness, ele age.

**Baseline**:
A configuração estável do motor no projeto (`autoresearch/core/config.py`): modelo, contexto, flags, amostragem. Você muda o Baseline e sobe o server — sem colar sopa de flags na mão a cada teste.

### MoE e offload

**MoE (Mixture of Experts)**:
Arquitetura com vários experts; a cada token só alguns acordam. O arquivo pode ser enorme, mas a VRAM “ativa” é menor — daí o offload de experts fazer sentido.

**Expert / Router**:
No MoE, expert = fatia de conhecimento que pode dormir ou trabalhar. Router = quem escolhe quais experts acordam naquele token. Sem router, MoE vira “tudo ligado”.

**Offload**:
Mandar parte do modelo para fora da VRAM (CPU/RAM). Em denso: evitar. Em MoE: ferramenta legítima (`--n-cpu-moe` / `N_CPU_MOE`) para caber o que não precisa estar na GPU o tempo todo.

### Skills

**Skill**:
Pacote de procedimento (`SKILL.md`) que ensina o agente *como* fazer um tipo de trabalho (não só *o quê*). Catálogo público: [skills.sh](https://skills.sh/). Skill = receita; tool = utensílio.

**Skill local (projeto)**:
Skill instalada no repositório (padrão do `npx skills add`). Viaja com o projeto — o time vê a mesma receita.

**Skill global (usuário)**:
Skill na sua conta (`npx skills add … -g`). Vale em qualquer projeto da máquina; não fica no git do repo.

**Cadeia de skills (wayfinder → …)**:
Sequência de exemplo para ir de mapa a PR:
`wayfinder` → `grill-with-docs` → `to-tickets` → `implement` → `code-review`.
Cada skill é um passo; a cadeia é o método.

### Amostragem

**Amostragem (sampling)**:
Como o motor escolhe a próxima palavra (metáfora do restaurante): freios → cardápio curto → ousadia → sorteio. Ordem típica: penalties → top_k → top_p → min_p → temperature → sorteio.

**Temperatura / Top-K / Top-P / Min-P / penalidades**:
Botões da amostragem. Temperatura = ousadia; Top-K / Top-P / Min-P = o que entra no cardápio; penalidades = freio de repetição/presença/frequência. Valores de “desligar” e detalhe no HTML do glossário.

### MCP

**MCP**:
Padrão de tomada para plugar utensílios extras no harness (docs, grafos, APIs…). O harness já traz faca e panela; MCP acrescenta ferramentas de fora sem reescrever o agente. Skill = receita; MCP = utensílio novo na cozinha.

**Context7**:
Servidor MCP de documentação: busca o manual *atual* da biblioteca/versão, em vez do modelo chutar de memória.

### Guardrails

**Tool**:
Uma ação concreta que o harness deixa o modelo pedir: ler arquivo, editar, rodar comando, chamar API… Sem tool, só chat; com tool, o texto vira efeito no mundo — por isso existem guardrails.

**Sandbox (caixa de areia)**:
Cerca do sistema operacional: o agente brinca à vontade *dentro*; fora (arquivos/rede sensíveis) o SO impede. Claude: Mac/Linux nativo; no Windows, de verdade só via WSL2. Cursor: Mac/Linux; no Windows sem WSL2 a proteção de arquivo quase some.

**WSL2**:
Linux rodando dentro do Windows. No Windows, sandbox sério de arquivo costuma depender dele — sem WSL2, “caixa de areia” vira ilusão.

**Guardrail**:
Regras e portas que limitam o que as tools podem fazer *antes* do estrago: Allow / Deny / Ask + hooks. Metáfora da boate: lista na porta, segurança, câmera. Existirem porque tools mexem no mundo real.

**Fadiga de aprovação**:
Clicar “Permitir” no reflexo (Enter) sem ler. É quando Ask vira teatro — e o motivo de Deny/hooks bem feitos.

**Allow / Deny / Ask**:
Três portas: Allow = passa sem perguntar; Deny = nem tenta; Ask = sempre confirma com você. Claude ordena deny → ask → allow. Cursor IDE usa lista de permitidos / Run Mode.

**Lista de permitidos / Run Mode**:
No Cursor IDE: o conjunto do que pode rodar sem popup. É o Allow do dia a dia — estreito demais trava; largo demais vira fadiga ou acidente.

**Hook (pré / pós)**:
Script encaixado na porta da tool. Pré (`PreToolUse` / `preToolUse`): pode inspecionar e *bloquear*. Pós (`PostToolUse` / `postToolUse`): só registra ou avisa — o que rolou, rolou.

**`--tools` / `--allowedTools` / `--disallowedTools`**:
Flags da Claude CLI. `--tools` = quais tools existem na sessão; `--allowedTools` = pode rodar sem perguntar; `--disallowedTools` = deny. Não misturar `--tools` com `--allowedTools`.

### Day / Night e entrega

**Agent Harness**:
Harness feito para agente: skills (como trabalhar), MCP (utensílios plugáveis), guardrails (o que pode / não pode) e tools (ações). Exemplos: Claude Code, Cursor, Codex. É o que transforma “modelo que completa texto” em “assistente que executa trabalho”.

**Day**:
Uso do modelo local de dia, com você no volante: rápido o bastante para o trabalho, menos gasto de tokens de assinatura. Supervisionado — você revisa o que importa.

**Night**:
Uso do modelo local sozinho (ex.: fila de issues enquanto você dorme). De manhã vem o humano: review, teste, PR. Night sem manhã vira bagunça.

**Issue**:
Ticket de trabalho (ex.: GitHub Issues): problema, tarefa ou pedido registrado. No Night, a fila de issues é o combustível do loop.

**PR (Pull Request)**:
Pedido para incorporar um diff no repositório. É o “entregável” da manhã depois do Night: código revisado, testado, pronto para outro olhar.

### Flags TPS

**Flags TPS (`-m`, `-ngl`, `-c`, `-fa`, `-ctk`/`-ctv`, `--n-cpu-moe`, …)**:
Botões do llama.cpp para caber na VRAM e subir velocidade. Cola e dicas no HTML do glossário.
