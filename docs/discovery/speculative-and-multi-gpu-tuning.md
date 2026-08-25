# Universal Speculative Decoding & Multi-GPU Tuning Playbook

This playbook documents universal inference mechanics, empirical benchmarks, and runtime tuning strategies for **Speculative Decoding (MTP, DFlash, Eagle, N-gram)**, **Reasoning/Thinking Models**, and **Multi-GPU Topologies** in `llama.cpp`.

While benchmarked initially using 27B dense setups ([`llm-qwen-fast`](https://github.com/RAFAEL-SILVASOUZA/llm-qwen-fast) / [Referência Técnica](https://youtu.be/W2r6GczmP_o)), the laws, trade-offs, and failure modes documented here apply broadly across **all local LLM architectures**.

---

## 1. Cross-Model Applicability Matrix

| Model Family | Speculative Mechanism | Optimal Sampling (`--temp`) | Reasoning Control (`reasoning_effort`) | Multi-GPU Recommendation |
|---|---|---|---|---|
| **Qwen 2.5 / 3.x** (Dense & MoE) | Embedded `draft-mtp` (`nextn`) | `0.2 – 0.3` (+50% TPS) | `low` (curbs runaway `<think>`) | Prefer 1 GPU (via `q8_0` KV) over 2-GPU layer-split. |
| **Gemma 4** (E4B, 26B, 31B) | External assistant draft (`--spec-draft-model`) | `0.2 – 0.3` (+80%–95% TPS) | N/A | Dedicated GPU pinning per instance. |
| **DeepSeek V3 / R1** | Multi-token prediction / DFlash / DSpark | `0.2 – 0.4` (+40%–60% TPS) | `low` for coding; `medium` for math | Split across PCIe only when weights exceed single card. |
| **Nemotron 3.5 / 4** | Embedded MTP (`nextn_predict_layers`) | `0.2 – 0.3` (+45%–55% TPS) | `low` / `medium` | Single GPU whenever possible. |
| **Llama 3.1 / 3.3 / Mistral** | `draft-eagle3` / Assistant draft / `ngram-mod` | `0.2 – 0.4` (+30%–45% TPS) | N/A | Dual independent servers (`1234`/`1235`) for multi-agent. |

---

## 2. Universal Performance Laws

Empirical analysis across `llama.cpp` builds (`b10356+`) isolates the impact of fundamental inference levers:

| Parameter / Technique | Typical Impact | Universal Mechanism | Applicable Models |
|---|---|---|---|
| **Low Temperature (`--temp 0.3` vs `1.0`)** | **+50% to +60%** generation TPS | Collapses output token distribution; raises draft acceptance rate and cuts verification time per step (~72 ms $\to$ ~49 ms). | **All speculative models** (MTP, Draft models, Eagle, DFlash, N-gram). |
| **Single GPU vs Multi-GPU (`-sm layer`)** | **+20% to +37%** TPS on 1 GPU | Eliminates inter-GPU PCIe sync latency per layer. Multi-GPU adds VRAM capacity, never linear generation speed. | **All dense & MoE models** where weights + KV fit on 1 card. |
| **Speculative Depth (`--spec-draft-n-max 4–5`)** | **+10% to +20%** TPS in agent tasks | Structured coding / JSON agent tasks have high predictability (draft acceptance 75%–98%), saturating small `n_max`. | All speculative models in deterministic/coding tasks. |
| **Native Host vs Virtualized (Win/Linux vs WSL2 drvfs)** | **+6% to +8%** TPS | Lower memory mapping latency and reduced I/O overhead compared to VM filesystem translation layers. | All models. |
| **KV Cache Quantization (`q8_0` / `q4_0`)** | −5% to −8% compute, **50%–75% VRAM savings** | Frees VRAM to keep the model on a single GPU or extend context 2x–4x without falling back to slow multi-GPU layer splitting. | All models with FlashAttention enabled (`-fa on`). |
| **Reasoning Effort Throttling (`low` / `medium`)** | **2x to 8x** faster task completion | Prevents thinking loops on routine tasks (e.g. 1.5k–3k thinking tokens instead of 16k–25k tokens on open tasks). | **All reasoning/thinking models** (DeepSeek-R1, Qwen 3.x, Nemotron). |

---

## 3. The Temperature–Speculative Symbiosis

A common failure mode in local LLM deployments is evaluating speculative decoding (MTP/DFlash/Eagle) with default high sampling temperatures (`temp >= 0.8`).

```
High Temperature (1.0):
  Target distribution is wide & stochastic
    -> Draft divergence is frequent
    -> Draft Acceptance plunges (30%–50%)
    -> Verification pass has high rejection cost (~70–80 ms/step)
    -> Result: Speculative decoding yields minimal gain or slows down generation.

Low Temperature (0.2 – 0.4):
  Target distribution collapses around top probabilities
    -> Draft tokens match target model intent
    -> Draft Acceptance surges (75%–98%)
    -> Verification pass is fast & batch-verified (~45–50 ms/step)
    -> Result: 50% to 100%+ net generation speedup across all architectures.
```

### Tuning Rules:
1. **Deterministic / Agentic Workloads (Coding, JSON, Tool Calling):** Always configure `--temp 0.2` to `--temp 0.3`, `--top-p 0.95`, `--top-k 20`, `--min-p 0`.
2. **Draft Length Sizing Rule:** Monitor `draft acceptance` and `mean len` in `llama-server` / `llama-cli` logs:
   - Theoretical ceiling = `n_max + 1`.
   - If `mean len` approaches the ceiling (e.g., `mean len = 3.9` with `n_max = 3`), increment `n_max` to `4` or `5`.
   - If `mean len` is significantly below `n_max`, the model is entropy-limited, not draft-capacity limited.

---

## 4. GPU Topologies: Layer Splitting vs Multi-Instance Pinning

When multiple GPUs are available (e.g. 2x RTX 3090, 2x RTX 4090, or heterogeneous pairs):

### The Layer-Split Fallacy (`-sm layer`)
Splitting a model across two GPUs across PCIe introduces synchronization barriers at every layer boundary:
- **Short prompts:** ~20% slower than running on a single GPU.
- **Long context (50k+ tokens):** Up to **37% slower** than running on a single GPU.
- **Conclusion:** Only use `-sm layer` when the model + desired context physically cannot fit inside the VRAM of a single GPU.

### Cross-axis rule: MTP depth optimum **shifts with the tensor-split ratio**

When the target forces a layer split across two GPUs (e.g. `--tensor-split 70,30`
on 12 GB + 8 GB), the **optimum `--spec-draft-n-max` changes with the split**,
not just with the target. Mechanism: a larger headroom on the fast device lets
draft-tokens land there, shifting where the verify cost is spent; deeper drafts
diverge more on the overflow device.

Empirical (heterogeneous dual-GPU, Qwen3.8-27B-UD-IQ3_S @ 131k, MTP embedded):

| `--tensor-split` | best `--spec-draft-n-max` | peak t/s |
|---|---|---:|
| `75,25` | **1** | 18.68 |
| `70,30` | **2** | 20.03 |

→ Sweep both axes together (`tensor-split` ∈ {`75,25`, `70,30`, `65,35`} ×
`spec-draft-n-max` ∈ {`1`, `2`, `3`}) on any target that needs a layer split.
Do not copy the dense single-GPU `n_max=4` canon onto a split config — the
optimum is one or two lower.

Hard ceiling caveats for this case (still apply): at 131k ctx the **native**
context can OOM mid-session at 45-61k tokens on dual-GPU; drop
`--ctx 49152` as a stable fallback if the 131k run dies.

### Token-budget hazard on 131k ctx (dual-GPU layer split)

Layer-split targets that pass preflight at startup can **still OOM
mid-session** once the KV cache fills past ~45-61k tokens — the failure
mode is not at load time, it is during prefill of a long Turn-N prompt.

- **Symptom:** server exits with `failed to allocate KV cache` at request
  time, after the model has been running fine for several shorter turns.
- **Mitigation:** ship a `ctx 49152` companion profile for any dual-GPU
  split target; the agent / Trial can fall back to it on OOM without
  restarting the workflow.


### The Multi-Instance Architecture (Best Practice for Multiple Agents)
Instead of serving multiple agent slots on a single multi-GPU instance (`-np 2`), run **isolated instances pinned to specific GPUs**:

```
[Agent 1 / IDE]  --->  llama-server (Port 1234, CUDA_VISIBLE_DEVICES=0)  --->  GPU 0 (~50 tok/s)
[Agent 2 / Background Task]  --->  llama-server (Port 1235, CUDA_VISIBLE_DEVICES=1)  --->  GPU 1 (~50 tok/s)
```
- **Single `-np 2` server across 2 GPUs:** Drops to **~25 tok/s per slot** due to fixed forward-pass overhead.
- **Dual pinned servers:** Delivers **~45–55 tok/s per agent**, with zero slot contention.

---

## 5. Universal KV Cache & VRAM Budgeting Traps

### 1. Speculative Decoding Doubles KV Context Allocation
In `llama.cpp` (`common/speculative.cpp`), speculative decoders (MTP, DFlash, Draft assistants) instantiate a **second context instance** (`llama_init_from_model`) with identical `n_ctx`:
$$\text{Total KV Allocation} \approx 2 \times \text{KV}(\text{n\_ctx})$$

*Operational Rule:* When pushing context limits on a single GPU, calculate VRAM requirements assuming $2\times$ KV memory. If VRAM is exhausted:
1. Switch KV cache from `f16` to `q8_0` or `q4_0` (`--cache-type-k q8_0 --cache-type-v q8_0`).
2. Disable vision projectors (`mmproj`) if multimodal input is not required (~1.5 GB VRAM freed).
3. If necessary, disable MTP to reclaim the secondary KV context.

### 2. Context Length Degradation Curve
Memory bandwidth pressure scales with active context depth:
$$\text{Step Time} \approx T_{\text{fixed}} + \alpha \cdot \text{Context Tokens}$$
- On 27B dense models: $\approx 53\text{ ms fixed} + 0.29\text{ ms per 1,000 context tokens}$.
- Generation throughput naturally drops from ~69 tok/s at 10k context to ~40 tok/s at 118k context.
- For long-running agent loops, resetting or compacting session context periodically provides greater throughput gains than any runtime flag.

### 3. MoE server prefill is micro-batch-bound (measured 2026-08-25)
35B-A3B @ `--n-cpu-moe 41` (full expert offload, codacus fork): server-side prefill was **~89 t/s at `--ubatch-size 128`** vs **704+ t/s at llama-bench `-ub 2048`** (fork features-off baseline; 820 t/s stock b10549; see [2026-08-24-codacus-fork-validation](../sessions/2026-08-24-codacus-fork-validation.md)). The bench gap is **ubatch**, NOT the host-register env levers (`GGML_CUDA_REGISTER_HOST` + `GGML_SCHED_PREFETCH_EXPERTS`, +78 % at full offload) — register-host cannot close an 8× gap. Raising the alias to `--batch-size 512 --ubatch-size 512` → **~292 t/s (3.3×)**, VRAM +0.1 GB (7.64 GB at 131k + 48-slot expert cache on 8 GB-class). ub 2048 server-side is VRAM-gated at 131k + expert cache; freeing VRAM (ctx 65k or fewer cache slots) is the prerequisite, with decode-speed tradeoffs. Falsifiable: sweep `--ubatch-size` ∈ {128, 512, 1024, 2048} at fixed ctx, record prefill t/s + peak VRAM.


---
## 6. Controlling Overthinking in Reasoning Models

Modern reasoning LLMs (DeepSeek-R1, Qwen 3.x, Nemotron) generate internal thinking chains (`<think>...</think>`). Without explicit constraint, open-ended tasks can cause runaway reasoning:

### Modulating Thinking Depth via Chat Template:
```json
{
  "chat_template_kwargs": {
    "preserve_thinking": true,
    "reasoning_effort": "low"
  }
}
```

| Level | Behavior | Recommended Use Case |
|---|---|---|
| **`low`** | Injects prompt constraints for concise, direct reasoning (1.4k–3k tokens). | Coding edits, tool calling, classification, agent execution loops. |
| **`medium`** | Default model behavior (no prompt injection). | Balanced exploration and architectural planning. |
| **`xhigh` / unconstrained** | Allows unlimited chain-of-thought (often exceeding 16k–25k tokens). | Complex mathematical proofs, deep algorithm synthesis. |

> **Note:** In `llama-server`, pass `reasoning_effort` inside `chat_template_kwargs` or via environment variable `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`. A top-level OpenAI-style `reasoning_effort` key is ignored by GGUF Jinja templates.

> **Qwen 3.8+ caveat (verified 2026-08-23):** The open-source Qwen3.8 chat template (e.g. `Qwen/Qwen3.8-27B-FP8` [chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/raw/main/chat_template.jinja), accessed 2026-08-23) **does** read `chat_template_kwargs.reasoning_effort` and **raises** on anything outside the ladder `xhigh / medium / low` (default `xhigh`). The `unconstrained` cell above is shorthand for "no `reasoning_instructions` injected", which is exactly what `medium` does in the Qwen3.8 template. Unsloth's stock template aliases `high → xhigh` so passing `reasoning_effort: "high"` reaches the same prompt as `xhigh` ([gist walk-through](https://gist.github.com/mdierolf/cb86ee2925e48777143346bc78933d59), accessed 2026-08-23). `chat_template_kwargs.enable_thinking: false` is **accepted** by the official template (it renders an empty `<think>` block) but causes downstream "empty-think poisoning" — the community froggeric patch restores clean fast mode for it ([froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates), accessed 2026-08-23). On the engine side, llama.cpp upstream exposes `--reasoning on|off|auto` (full on/off kill), `--reasoning-budget N` (-1 unrestricted / 0 immediate end / N>0 cap), and `--reasoning-effort LEVEL` (`minimal/low/medium/high/xhigh/max/default`) which is forwarded into `chat_template_kwargs` ([tools/server/README.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md), accessed 2026-08-23; `common_chat_templates_inputs.enable_thinking = true` and the `chat_template_kwargs` map live in [common/chat.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/chat.h), accessed 2026-08-23). The hosted QwenCloud API for `qwen3.8-max` documents a different surface (`reasoning_effort ∈ {low, medium, xhigh}`, default `xhigh`, and `enable_thinking`/`thinking_budget` are separate toggles; `reasoning_effort` and `thinking_budget` cannot be set simultaneously) ([QwenCloud Thinking guide](https://docs.qwencloud.com/developer-guides/text-generation/thinking), accessed 2026-08-23). External / unverified on this rig — confirm via `GET /props` `chat_template_caps` and a one-shot `llama-cli -p "..."` smoke before relying on `enable_thinking=false` for a Trial.
>
> **Server-side vs template-side reasoning (verified 2026-08-23):** `common/arg.cpp` is the **engine-side** surface; it writes into `params.default_template_kwargs` — the same map Jinja reads as `chat_template_kwargs`. The two layers **compose**, they don't compete:
>
> - **`--reasoning-format FORMAT`** ([`common/arg.cpp` L3638-3649](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)) — `none | deepseek | deepseek-legacy | auto`. Controls `<think>` → `message.reasoning_content`; default `auto`.
> - **`--reasoning [on|off|auto]`** ([`common/arg.cpp` L3651-3667](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)) — default `auto`. `on`→`enable_thinking:"true"`; `off`→`"false"`; `auto` leaves the template default. Only engine knob mapped onto the Qwen 3.8+ `enable_thinking` flag.
> - **`--reasoning-effort LEVEL`** ([`common/arg.cpp` L3669-3679](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)) — `default | minimal | low | medium | high | xhigh | max`. `default` **erases** the kwarg; else dumps the string. C++ doesn't validate against the template's ladder — Qwen 3.8 above only accepts `xhigh | medium | low`, so `--reasoning-effort minimal` raises in Jinja.
> - **`--reasoning-budget N`** ([`common/arg.cpp` L3681-3688](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)) — `-1` unrestricted (default), `0` immediate end, `N>0` cap; `<-1` throws.
> - **`--reasoning-budget-message MESSAGE`** ([`common/arg.cpp` L3690-3696](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)) — prepended to end-of-thinking tag on exhaustion; default `none`.
> - **`--reasoning-preserve` / `--no-reasoning-preserve`** ([`common/arg.cpp` L3698-3713](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp)) — sets `"preserve_reasoning":"true"|"false"`; only when template declares `supports_preserve_reasoning`.
>
> **OpenAI-compat wire surface** (request body — same machinery, no restart):
>
> - **`chat_template_kwargs`** ([`tools/server/README.md` L1312](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)) — JSON merged into `default_template_kwargs` per request. Flag equivalent `--chat-template-kwargs STRING` ([README L212](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)).
> - **`reasoning_effort`** ([`tools/server/README.md` L1314](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)) — **`"none"` disables** reasoning/thinking (the wire knob the Qwen 3.8+ empty-think-poisoning caveat wants); other values pass to the template.
> - **`reasoning_budget_tokens` / `thinking_budget_tokens`** ([`tools/server/server-common.cpp` L1353-1367](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/server-common.cpp)) — `-1` unrestricted, else cap; `thinking_budget_tokens` is the legacy alias.
> - **`reasoning_budget_message`** — same L1353-1367 block; injected on exhaustion; default `opt.reasoning_budget_message` from server start.
> - **`reasoning_control: true`** ([`tools/server/README.md` L1316](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)) — arms **early-end-of-reasoning** via `POST /v1/chat/completions/control` body `{"id":"...","action":"reasoning_end"}` ([README L1439-1447](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md)). Default `false`; no-op on finished completions.
>
> **Falsifiable experiment** — confirm `reasoning_effort: "none"` kills thinking on a Qwen 3.8 GGUF without restart:
>
> ```bash
> curl -s http://127.0.0.1:1234/v1/chat/completions \
>   -H 'Content-Type: application/json' \
>   -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"2+2=?"}],
>        "reasoning_effort":"none","max_tokens":64}' \
> | jq '.choices[0].message | {reasoning_content, content}'
> # Expect: reasoning_content="" (or null), content="4".
> # Non-empty → template ignored the flag.
> ```

---

## 7. Universal Production Startup Configuration Template

```bash
llama-server \
  -m "models/<MODEL_NAME>.gguf" \
  --alias "<MODEL_NAME>" \
  -c 98304 \
  -ngl 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -ub 1024 \
  -np 1 \
  -n -1 \
  -t 8 -tb 8 \
  --temp 0.3 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0 \
  --repeat-penalty 1 \
  --no-context-shift \
  --reasoning on \
  --cache-ram 16384 \
  --jinja \
  --metrics \
  --host 0.0.0.0 \
  --port 1234
```

### Inactive / Incompatible Flag Checklist:
- **`--backend-sampling`:** Disabled automatically by `llama.cpp` whenever grammar / JSON schema / tool calling is present (harmless but effectively a no-op for agents).
- **`-sm row`:** Incompatible with modern CUDA split buffers (`b9890+`).
- **`--spec-draft-n-max > 5`:** Excessive draft lengths suffer from diminishing returns and can trigger engine crashes or hangs. Keep between 3 and 5.
