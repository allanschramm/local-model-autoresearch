# Ornith-1.0-9B — Model Card (Local)

**Source (Unsloth UD):** https://huggingface.co/unsloth/Ornith-1.0-9B-GGUF  
**Source (deepreinforce official):** https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF  
**Unsloth docs:** https://unsloth.ai/docs/models/qwen35 (model uses Qwen 3.5 architecture)  
**License:** Apache-2.0 / upstream as published  
**GGUF basenames (separate Trials):**  
- `Ornith-1.0-9B-UD-Q4_K_XL.gguf` — Unsloth Dynamic  
- `Ornith-1.0-9B-MTP-Q4_K_M.gguf` — MTP pack  
- `ornith-1.0-9b-Q4_K_M.gguf` — deepreinforce official Q4_K_M (coding-10 **0.5800**; claw-full **0.4000** @ 65k)  
**Family:** Ornith (deepreinforce-ai; Qwen 3.5 architecture)  
**Quantizations:** Unsloth `UD-Q4_K_XL` ≠ deepreinforce `Q4_K_M` ≠ MTP pack — each is its own Objective Vector.

## Architecture (from GGUF metadata, verified via gguf lib)
- Causal LM (hybrid Attention + SSM)
- **`block_count` = 32 layers**
- Hidden **4096**, vocab 248320, ctx **262144**
- **Hybrid Attention + SSM (Mamba-2 style) layers**:
  - `full_attention_interval = 4` — every 4th layer is full attention
  - Contains `ssm.conv_kernel=4`, `ssm.state_size=128`, `ssm.group_count=16`, `ssm.time_step_rank=32`, `ssm.inner_size=4096`
  - 8 layers of full attention (head count: 16 Q, 4 KV, key/value length 256)
  - 24 layers of SSM / linear path
- `rope.freq_base = 10,000,000`
- **`general.name` = `Ornith 1.0 9B`**, file_type=15 (Q4_K_M), quantization_version=2
- 427 tensors total

## Hardware requirements (per community and size)
| Quant | Total RAM / VRAM |
|---|---|
| **Q4_K_M (our pick)** | **~5.6 GB VRAM** (VRAM target is ~5.6 GB + KV cache overhead) |
| Q8_0 | ~9.5 GB VRAM |

**Our target:** 8 GB VRAM (RTX 4060). The 4-bit model size is ~5.6 GB, meaning it fits entirely in GPU VRAM (NGL = 999). However, active KV cache overhead for large contexts can push VRAM usage above 8 GB. Setting safe context size limits is important.

## Recommended Settings (based on Qwen 3.5)
- **Temperature:** 0.4
- **Top P:** 0.95
- **Top K:** 20
- **Min P:** 0.0
- **Repeat Penalty:** 1.0 (disabled)

## MTP (Multi-Token Prediction)
- **Base UD GGUF (`Ornith-1.0-9B-UD-Q4_K_XL.gguf`): NO `nextn`.**
- **MTP GGUF (downloaded 2026-07-20):** `models/Ornith-1.0-9B-MTP-Q4_K_M.gguf` from `protoLabsAI/Ornith-1.0-9B-MTP-GGUF` — embedded `nextn`.
- Enable on MTP file: `SPEC_TYPE=draft-mtp`, `SPEC_DRAFT_N_MAX=4`, no draft model path.
- Fair matrix: base UD **38.7 t/s** → MTP GGUF **56.3 t/s** (**+46%**). [session](../sessions/2026-07-20-small-model-tps-matrix.md) · [guide](../discovery/small-model-mtp-tps.md).

## VITRIOL / Split strategy
Since the model is ~5.6 GB and we have 8 GB of VRAM, we can run with maximum GPU offload (`--n-gpu-layers 999`), loading the model completely into GPU VRAM.

## Our config baseline (speed path 2026-07-20)
- Prefer MTP file when throughput matters: `MODEL = 'Ornith-1.0-9B-MTP-Q4_K_M.gguf'`
- `SPEC_TYPE = 'draft-mtp'`, `SPEC_DRAFT_N_MAX = 4`
- Shared matrix knobs: `KV q4_0`, batch 256/128, threads 6/8, `NO_MMAP=True`
- Quality/coding baselines historically used non-MTP UD — keep separate if comparing scores to older runs.

## Older verified baseline (2026-06-26, non-MTP)
- `MODEL = 'ornith-1.0-9b-Q4_K_M.gguf'` — deepreinforce official Q4_K_M (separate Trial from UD/MTP)
- Day/Night picks on this front use Unsloth UD / MTP basenames when those vectors are complete — see Pareto leaderboard
- `CTX_SIZE = 131072`
- `KV_CACHE = 'q4_0'`
- `NGL = 99`
- `THREADS = 8`
- `THREADS_BATCH = 8`
- `FLASH_ATTN = 'on'`

### Benchmark Scores (10 tasks baseline)
- **Coding Score:** `0.4800`
  - **HumanEval+:** `0.4000`
  - **MBPP+:** `0.9000`
  - **LiveCodeBench:** `0.4000`
  - **BigCodeBench Hard:** `0.1000`
- **Peak VRAM:** `7.9 GB`
- **TPS:** `49.4`

## Sources / Verification
- HuggingFace Model Card (`deepreinforce-ai/Ornith-1.0-9B-GGUF`)
- Checked with local scratch tool parsing `models/ornith-1.0-9b-Q4_K_M.gguf` metadata via `GGUFReader` on 2026-06-26.
- Verification baseline run completed successfully on 2026-06-26.

## Tuning History (2026-06-29)

### Hyperparam sweep
| Param | Score | TPS | Verdict |
|-------|-------|-----|---------|
| Baseline (0.4/0.95/20) | **0.580** | **52.2** | ✅ Best |
| TEMP=0.7 | 0.325 | 51.9 | ❌ |
| TOP_P=0.9 | 0.425 | 51.3 | ❌ |
| REPEAT_PENALTY=1.05 | 0.490 | 48.2 | ❌ |
| TOP_K=40 + REPEAT_PENALTY=1.05 | 0.625 (val) | 51.2 | ⚠️ Full 10-task: 0.49 |

### BeeLlama tested (no gains)
- BeeLlama baseline: 41.7 TPS (20% slower than stock fork)
- BeeLlama + CopySpec: 45.1 TPS (marginal, server crashes)
- BeeLlama + TCQ turbo3_tcq: HTTP 500 at 131k ctx
- BeeLlama + turbo3: 31.4 TPS (saves VRAM but kills TPS)

### 2026-07-01 Validation (2-task, b1024/ub256, upstream build-cuda)
| Metric | Value |
|--------|-------|
| Score | 0.5500 |
| HE+ | 1.0000 |
| MBPP+ | 0.5000 |
| LCB | 0.5000 |
| BigCode | 0.0000 |
| TPS | 50.2 |
| VRAM | 7.1 GB |
| Bench tg | 43.7 t/s |

### Verdict
- **0.580 / 52.2 TPS is the ceiling** for RTX 4060 8 GB
- No hyperparam, fork, or quant improves either score or speed
- 9B is the optimal model for this hardware

### 2026-07-19 Update (Unsloth Dynamic 4-bit XL Quant)
- Upgraded local model to the newly released `Ornith-1.0-9B-UD-Q4_K_XL.gguf` (5.98 GB) from Unsloth.
- Alias: `ornith-9b` (INDEX name; old `o9` retired 2026-07-23).

### Claw-Eval full (2026-07-24)
- **Val Score 0.6000** (9/15) @ ctx **65536**, bench_tg **42.1**, peak **7.5 GB**. Prior claw-quick 0.80.

### Objective Vector @ 65k (2026-08-08, `VRAM_LIMIT_MB=8100`, `AUTORESEARCH_SKIP_FREE_CLAMP=1`)
- Fingerprint shared: coding-10 **0.5400** (HE 0.90 / MBPP 0.70 / LCB 0.40 / BC 0.00; peak **7.8 GB**; TPS **48.6**) + claw-full.
- First claw re-run after Jul scored **0.3333** — harness bug (ignored `reasoning_content`, `max_tokens=512`). After agentic fix: claw-full **0.9333** (14/15), peak **7.7 GB**, TPS **42.1**.
- Merged vector **complete** (`agentic=0.9333`, `coding=0.5400`); status **on_front** / Night #3. Historical Jul coding **0.5700** @ 32k remains a different Fingerprint.

### Coding-10 (2026-07-24, UD-Q4_K_XL)

| Metric | Value |
|---|---|
| ctx | **32768** (65k VRAM-kill mid-HumanEval) |
| coding | **0.5700** (LCB patched†) |
| LCB / HE / MBPP / BC | 0.40 / 0.90 / 0.70 / 0.20 |
| bench_tg | 38.4 t/s |
| peak VRAM | 7.4 GB |

† HE/MBPP/BC from Trial `15f6bed0-…`; LCB via `scripts/lcb_only.py`. Alias `ornith-9b` stays **65k for agentic**; use **≤32k for coding** on 8 GB. Evidence: [coding-10](../sessions/2026-07-24-coding-10-claw-leaders.md), [coding leaderboard](../discovery/coding-leaderboard.md).

### Coding-10 MTP variant (2026-07-27, `Ornith-1.0-9B-MTP-Q4_K_M.gguf`)
- ctx **32768**, draft-mtp n=4, TEMP 0.4.
- **coding 0.5800** (LCB 0.40 / HE 0.80 / MBPP 0.90 / BC 0.10); Combined TPS **86.7**; bench_tg **64.0**; peak **7.4 GB**.
- Ties UD coding (0.57); agentic claw-full **0.4667** (weaker than UD 0.60). Speed pick only.

### deepreinforce `ornith-1.0-9b-Q4_K_M` claw-full (2026-07-28)
- First try @ 65k / `VRAM_LIMIT_MB=7900`: **VRAM kill** mid T054 (used 7930 MB) after 6/15 @ 0.4000 — row `MODEL_REJECTED`.
- Retry @ 32k: early VRAM kill on T002 (7910 MB) — not usable.
- Success @ 65k / `VRAM_LIMIT_MB=8000`: **agentic_full 0.4000** (6/15), bench_tg **42.5**, peak **7.8 GB**. Weaker than UD claw **0.6000**. Vector **complete** with coding **0.5800** (`iq_min=0.4000`).

## Open questions
- None (baseline verified). Coding vs historical Mythos 0.64 still open if Mythos GGUF present.
