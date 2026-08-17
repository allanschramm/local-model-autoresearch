# Capability Extraction Harness — Raising IQ per Token Without Engine Changes

## Purpose

Engine TPS has a hard ceiling per model+quant Fingerprint. The remaining local lever is **how many useful tokens the harness spends per completed task** — the ratio between a model's measured skill and its realized task completion is called *capability-realization loss* in the 2026 literature. This guide catalogs harness-side capability-extraction techniques (led state, verification loops, test-time compute) and maps each to a falsifiable experiment on this repo's agentic / coding axes.

Reference: [thinking-models-claw-harness.md](./thinking-models-claw-harness.md) — the in-repo precedent: a harness bug (ignored `reasoning_content`, `max_tokens=512`) produced false "dumb model" floors (Ornith 0.333 → 0.933). That fix was pure IQ recovery at identical engine TPS. This guide generalizes that lesson.

## 1. Core insight

- Wall-clock per task = `tokens_spent / TPS`. At fixed engine TPS, raising agentic/coding = raising **IQ per token**.
- Capability-realization loss sources: interface mismatch (persona/schema drift between training and harness), goal-fade on long loops, blank retries without failure diagnosis, runaway reasoning chains, under-budgeted `max_tokens`.
- **score/token** (and score/time) is the metric that separates "harness extracted IQ" from "engine got faster". It is observational — derived from `results.tsv` + mock logs, **not** a Pareto axis. Pareto axes stay ctx × TPS × agentic × coding (ADR 0006).

## 2. Inference-engine / ctx survey (August 2026) — what's real

- **Pinned engine is already current.** Prebuilt `llama.cpp-releases/upstream/b10375` (build dated 2026-08-13) postdates the 2026 Gemma-4 KV-cache fix (~40% memory cut in context-heavy scenarios, Apr 2026), the upstream MTP merge (May 2026), and the MMVQ kernel rewrites. No upgrade action.
- **N-gram speculation on coding text.** `ngram-cache` / `ngram-simple` are universal (any model), 0 MB VRAM, and 2026 reports show n-gram drafting is surprisingly effective on repetitive/coding text. The 2026-07-20 small-model TPS matrix has **no ngram row** — candidate Search neighbor on coding-10. **TBD:** measured gain.
- **`--model-draft` external drafts** merged upstream May 2026: equivalent to MTP for dense targets; known-fail on MoE + CPU-expert offload (see [speculative-decoding-formats.md](./speculative-decoding-formats.md) §4b). Skip.
- **DFlash KV ideas are 3× this rig's VRAM scale.** The DFlash 256K-ctx-on-24-GB-class demo (TQ3_0 3.5 bpv KV, 4096-slot target-feature ring buffer, 2048-token sliding-window flash attention) is a concept reference only — no transfer to a discrete 8 GB-class rig.

## 3. dflash / dspark — vendor claims vs in-repo measurement

| Technique | Vendor claim (2026) | In-repo measurement (8 GB-class) | Verdict |
|---|---|---|---|
| DFlash (block-diffusion drafter, NVIDIA/DeepSeek DeepSpec) | 3–6× lossless; up to 15× Blackwell | 12.5 vs 24.6 t/s (−49%) on Qwen3.6-35B-A3B with `n-cpu-moe`; needs 35B target fully on GPU | Dead end on MoE + CPU offload |
| DSpark (semi-autoregressive drafter + confidence head + hardware-aware scheduler, DeepSeek) | 60–85% faster per-user, up to 6.6× throughput | 19.2 vs 37.3 t/s (−48.5%) on Bonsai-27B Q1_0 (quantization-speed inversion: Q4 draft slower than Q1 target) | Dead end under extreme low-bit targets |

**Durable rule:** neural speculation wins only when target **and** draft fit on GPU and the draft is faster per token than the target. On this rig's MoE/CPU-offload fingerprints it is a proven loss.

**New (untested) path:** official DeepSeek DSpark is integrated into SGLang with Gemma-4 drafts (`deepseek-ai/dspark_gemma4_12b_block7`; community GGUF `ankk98/dspark-gemma4-12b-block7-Q4_0-GGUF`). Only reachable via the SGLang backend with a dense on-GPU target. **TBD:** one SGLang Trial only if that path is ever exercised.

## 4. J-Space report (Tiger3807861189) — assessment

Facts: report-only repo (README + LICENSE + CITATION, no code — the plugin lives in a separate repo); CC BY-ND license (do not copy text into tracked docs); **single-run results, no confidence intervals** (the report itself states single runs do not represent a stable distribution); target models are API-only (DeepSeek V4-Flash 284B / V4-Pro 1.6T — nothing local-runnable); 354 stars / 19 forks on a 3-file repo.

Verdict: **technique reference, not a dependency.** The diagnosis is sound; the numbers are unverified marketing-grade evidence.

Transferable mechanisms — each falsifiable via a harness flag switched on/off on one Fingerprint:

- **Led state**: persistent `Goal / Core / Verified / Open / Next` block the model maintains across tool turns → attacks goal-fade on long (65k) loops.
- **Checkpoint with verifier + coverage**: resume from the last *verified* state, not the last action; wire hidden-test results into retry decisions. This repo's `agentic_coding` pass already has hidden tests — the missing piece is feeding them back into continuation.
- **Fail-diagnosis retry**: retry with the failure reason in context, never a blank retry.
- **fast/full/loop gating**: spend think budget proportional to task depth — maps to the Day (fast) vs Night (deep) profile split.

Caveats (from the report itself): first-person word frequency (`we`/`let me`) is a behavioral probe, not a quality mechanism; gains must be reproduced on the same model + harness (switch on/off) — exactly this repo's Trial protocol.

## 5. Test-time compute catalog (harness-side, evidence-backed)

- **Self-consistency / best-of-N + self-verification**: sample N diverse continuations for grader-critical turns, verify, keep best. No second model, no extra VRAM — small models verify their own outputs. Cost: N× tokens at same TPS → trades wall-clock for IQ. Strongest evidence base in the catalog.
- **Small verifiers**: verification is structured reasoning, not scale; small efficient models rank outputs adequately. Enables verifier-in-the-loop without a second on-GPU model.
- **SELFCOMPACT adaptive compaction**: the model decides when to summarize its own context → lower KV pressure for the same effective working memory. The only item in this catalog that touches **both** the IQ and the CTX axis.
- **Constrained/structured decoding for tool calls** (XGrammar on the SGLang path; llama.cpp grammars): malformed tool JSON is simultaneously a TPS waste and an IQ waste — one broken tool call costs a full retry round-trip.
- **MCTS / LATS**: real but token-hungry; only worth it on `agentic_coding`, and only after the cheaper items above are exhausted.

## 6. What NOT to chase (dead ends with evidence)

- **Engine bump**: pinned release already current (see §2).
- **dflash / dspark re-runs** on 8 GB-class MoE: measured losses (§3); vendor claims are datacenter/Blackwell conditions.
- **J-Space code integration**: CC BY-ND, API-only models, no code in the report repo (§4).
- **Consumer GUI engines** (Atomic Chat, LM Studio etc.): no harness path, no measured edge over llama.cpp CUDA on this rig — see [fastest-tps-inference-engine.md](./fastest-tps-inference-engine.md).
- **Colibrì / TRT-LLM / OpenVINO / vLLM / LMDeploy**: already documented and classified in [inference-engines-landscape.md](./inference-engines-landscape.md).

## 7. Candidate experiments (falsifiable, one Fingerprint each)

1. **ngram-cache as Search neighbor** — engine-only, 0 MB, universal. Measure coding-10 + Claw vs `--spec-type none` on the same Fingerprint. Zero cost, zero risk.
2. **Led-state + fail-diagnosis retry harness flag** — `agentic_coding` switch on/off on one Fingerprint. Harness change → operator decision (ADR if adopted).
3. **Best-of-N self-consistency** on grader-critical turns — measure agentic/coding gain per token budget.
4. **score/token observational metric** in Night loops — derived from existing results + mock logs; not a Pareto axis.
5. **SELFCOMPACT adaptive compaction** in the agentic runner — operator decision; only item affecting the CTX axis.

Protocol rule: harness is fixed unless the operator approves a change; every candidate must be measured same Fingerprint, same tasks, switch on/off.

## 8. Open questions

- **TBD:** ngram-cache gain on coding-10 — unmeasured in the 2026-07-20 matrix.
- **TBD:** DSpark-in-SGLang — untested on this rig; only relevant with a dense on-GPU target via the SGLang backend.
- **TBD:** best-of-N token cost vs agentic gain — no local measurement yet.
- **TBD:** whether newer upstream builds than `b10375` contain further CUDA kernel rewrites (Ada/Blackwell-class gains) — verify at the next engine bump; keep `nextn_predict_layers` (Nemotron-3.5 embedded-MTP) load as the gating criterion.

## Sources

- [nvidia.com — DFlash up to 15× on Blackwell](https://developer.nvidia.com/blog/boost-inference-performance-up-to-15x-on-nvidia-blackwell-using-dflash-speculative-decoding/)
- [deepseek.ai — Inside DeepSeek DSpark (lossless inference)](https://deepseek.ai/blog/inside-deepseek-dspark-lossless-inference)
- [baseten.co — DFlash faster LLM inference](https://www.baseten.co/blog/dflash-faster-llm-inference/)
- [note.com — DFlash 256K ctx on 24 GB, TQ3_0 KV, ring buffer](https://note.com/samehadaonsen/n/nde52cc429dfd)
- [dev.to — why MTP doesn't speed up and how to fix it (SWA caveat, n-gram)](https://dev.to/alanwest/why-mtp-doesnt-speed-up-your-llamacpp-inference-and-how-to-actually-fix-it-2m2m)
- [n1n.ai — Gemma-4 local inference KV-cache fix (Apr 2026)](https://explore.n1n.ai/blog/gemma-4-local-inference-llama-cpp-kv-cache-fix-npu-benchmarks-2026-04-05)
- [arxiv 2606.23525 — adaptive context compaction (SELFCOMPACT)](https://arxiv.org/pdf/2606.23525)
- [arxiv 2501.14304 — LATS (Language Agent Tree Search)](https://arxiv.org/html/2501.14304v1)
- [arxiv 2607.05391 — small models as verifiers](https://arxiv.org/html/2607.05391v1)
- [Tiger3807861189/DeepSeek-V4-J-Space-Capability-Realization-Report](https://github.com/Tiger3807861189/DeepSeek-V4-J-Space-Capability-Realization-Report)
- [xiaobright/dsh-anchored-standard](https://github.com/xiaobright/dsh-anchored-standard)
- [yjh051108/dsh-routing-suite](https://github.com/yjh051108/dsh-routing-suite)
