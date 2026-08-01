# 2026-07-31 — DAY model candidates with a 100k context floor

## Goal

Find new GGUF candidates for this rig that can plausibly run at least 100k effective context and compete as DAY models. No model was downloaded or benchmarked in this session.

## Hardware and selection target

- RTX 4060 8 GB, about 32 GB host RAM, `VRAM_LIMIT_MB=7900`.
- Dense models may not spill layers into shared memory. MoE experts may use `N_CPU_MOE`.
- User floor: `CTX_SIZE >= 100000`; target the repo default `131072` where the model supports it.
- DAY admission today: `min(agentic, coding) >= 0.4612`; then maximize TPS.

Current measured 131k references from `results.tsv` via `scripts/rank_results.py --mode all`:

| GGUF basename | TPS | agentic | coding | DAY admission |
| :--- | ---: | ---: | ---: | :--- |
| `Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf` | 43.6 | 0.3333 | 0.6400 | no |
| `ornith-1.0-9b-Q4_K_M.gguf` | 42.5 | 0.4000 | 0.5800 | no |
| `Qwen3.5-4B-MTP-Q4_K_M.gguf` | 87.0 | 0.2667 | 0.3850 | no |

The practical search target is therefore not merely “faster than Qwythos”; it must retain `min(agentic, coding) >= 0.4612` at 100k+.

## Ranked shortlist

### 1. NVIDIA Nemotron 3 Nano 4B — first trial

- Official sources: [model/GGUF card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF), [official files](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF/tree/main).
- Official GGUF: `NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf`, 2.84 GB.
- Architecture: 3.97B-parameter Mamba2/Transformer hybrid, primarily Mamba-2 and MLP with only four attention layers; compressed from Nemotron Nano 9B v2.
- Context: up to 262k; NVIDIA reports RULER at 128k for the Q4_K_M.
- License: NVIDIA Nemotron Open Model License; official card says commercial use is allowed.
- Recommended sampling: reasoning `TEMP=1.0`, `TOP_P=0.95`; tool calling `TEMP=0.6`, `TOP_P=0.95`. `TOP_K`, `MIN_P`, and penalties are not published: leave them explicitly TBD rather than inventing values.
- DAY rationale: strongest 100k fit. The 2.84 GB weights and only four attention layers leave substantially more KV headroom than a full-attention 4B/8B model. Official positioning covers agentic edge use and coding languages. Quality and llama.cpp throughput remain unmeasured locally.

### 2. IBM Granite 4.0 H Tiny — speed wildcard

- Official sources: [model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny), [official GGUF](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF).
- Official GGUF choices: Q4_K_S 4.00 GB, Q4_0 3.96 GB, Q4_K_M 4.23 GB, Q5_K_M 4.95 GB, Q6_K 5.71 GB, Q8_0 7.39 GB. Start candidate: Q4_K_M; Q4_K_S is the fallback if preflight lacks headroom.
- Architecture: decoder-only hybrid MoE, 7B total / 1B active, four attention plus 36 Mamba-2 layers, 64 experts / six active, GQA.
- Context: 128k.
- License: Apache-2.0.
- Recommended sampling: not published on the official card; all sampler fields are TBD and must be resolved before a Trial.
- DAY rationale: 1B active parameters suggest high generation speed, while four attention layers make 100k KV plausible. Under repo rules the first Baseline uses `N_CPU_MOE=None` (full expert CPU offload resolved by GGUF metadata); actual CPU-offload TPS may erase the advantage.

### 3. IBM Granite 4.1 3B — quality/control candidate

- Official sources: [model card](https://huggingface.co/ibm-granite/granite-4.1-3b), [official GGUF](https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF).
- Official GGUF choices: Q4_K_S 2.00 GB, Q4_K_M 2.10 GB, Q5_K_M 2.44 GB, Q6_K 2.80 GB, Q8_0 3.62 GB. Start candidate: Q4_K_M.
- Architecture: dense decoder-only Transformer, 3B parameters, 40 layers, hidden size 2560, 40 attention heads / eight KV heads, head size 64, GQA/RoPE/SwiGLU.
- Context: 131072.
- License: Apache-2.0.
- Recommended sampling: not published on the official card; sampler fields remain TBD.
- DAY rationale: official card reports HumanEval 81.71, HumanEval+ 76.83, MBPP 71.16, BigCodeBench 32.19, BFCL v3 60.80 and explicit tool-use support. Its small weights and half-size attention heads make 131k with quantized KV plausible on 8 GB, but unlike the two hybrids it pays KV for all 40 layers.

## Rejected or deferred at the 100k floor

| Model | Decision | Reason |
| :--- | :--- | :--- |
| Granite 4.1 8B Q4_K_M (5.35 GB) | reject | Forty full-attention layers, eight KV heads and 128-wide heads leave no realistic 100k KV + runtime headroom on 8 GB under the dense no-spill rule. |
| Qwen3 4B Q4_K_M (2.50 GB) | defer | Official 131k support, but 36 full-attention layers with eight 128-wide KV heads make 100k very tight before runtime overhead; Granite 3B offers a safer full-attention control. |
| Qwen3.5 4B | not new / reject for DAY | It already has a complete local 131k vector: 87.0 TPS, agentic 0.2667, coding 0.3850; below DAY admission. |
| Gemma 3 4B QAT Q4_0 | defer | 128k-capable and 3.16 GB, but official evidence is weaker for tool use and agentic coding than the three shortlisted candidates. |

## Commands

```powershell
.\venv\Scripts\python.exe scripts\rank_results.py --mode day
.\venv\Scripts\python.exe scripts\rank_results.py --mode all
```

Primary-source research used official publisher Hugging Face cards/repositories only. No `hf` download, hardware probe, server, validation, or benchmark command ran.

## Findings

1. Test Nemotron 3 Nano 4B Q4_K_M first.
2. Test Granite 4.0 H Tiny Q4_K_M second; its DAY case depends on measured CPU-expert-offload TPS.
3. Use Granite 4.1 3B Q4_K_M as the dense quality/control trial.
4. A candidate is not a DAY model until the same 100k+ Fingerprint completes Claw full and coding-10 and clears the current 0.4612 IQ band.

## Errors and unknowns

- Fit claims above are architecture/file-size screening, not local measurements.
- Official cards for both Granite candidates do not publish a complete recommended sampler. The root sampler-seeding gate blocks blind Trials until this is resolved from an official source or explicitly approved as a quality experiment.
- GGUF metadata (`block_count`, expert count, KV heads, MTP fields) cannot be verified until a user-authorized `hf` download exists locally.

## Decisions

- Minimum context is 100k, so the current 32k DAY pick is not a valid comparison target for this search.
- Prefer hybrid layouts with few attention layers; they buy KV headroom without dense shared-memory spill.
- No downloads or Trials were authorized by this research request.
