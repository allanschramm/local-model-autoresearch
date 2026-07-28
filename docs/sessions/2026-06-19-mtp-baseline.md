# Session 2026-06-19 — MTP Baseline Validation

## Goal
Confirm empirically that swapping the base GGUF for the MTP-GGUF (same flags, MTP inactive) moves TPS. Investigation started from a then-best measured baseline of 11.1 tok/s (`r_q4`) vs an unmet ~16 tok/s target.

## Hardware & Build
- CPU: R7 5800X (8C/16T)
- RAM: 32 GB DDR4-3200
- GPU: RTX 4060 8 GB
- OS: WSL2 Ubuntu-24.04
- llama-server: turboquant build (MTP, TurboQuant, QAT, diffusion)
- llama-server path: `./llama.cpp-turboquant/build-cuda/bin/llama-server`

## Models tested this session
- **MTP-GGUF**: `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.11 GB)
  - Source: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` (HF)
  - **Validated**: GGUF contains MTP tensors — `qwen35moe.nextn_predict_layers u32 = 1` in llama-server log
  - **Filename does NOT contain "MTP"** → auto-detect in `llama_runner.py` does not fire alone. Pass `spec_type` explicitly or rename.

## Decision sequence

### Step 0 — Initial investigation
- Operator reported 5 manual runs (qwen_baseline 11.5, r_q4 11.1, r_turbo 11.3, qwen_turbo 4.4, qwen_final 6.4) and unmet 16 tok/s target.
- Agent hypothesis: "16 tok/s = mix-up with Codacus 18 (YouTube) and Gemma 4 ~13-18 (prior smoke)". Operator confirmed.
- Agent hypothesis #2: MTP-GGUF would be the lever (1.4-2.2×). Operator pointed to HF MTP-GGUF + Unsloth MTP guide.

### Step 1 — YouTube as additional data
- Operator watched a video claiming 17 tok/s on Qwen3.6-35B-A3B on weaker hardware (GTX 1060 6GB + old CPU + DDR4) with flags: MoE offload (`--n-cpu-moe 35-36`), `--no-mmap`, layer count, TurboQuant 4-bit K + 3-bit V, `--mlock` + Docker IPC lock.
- Video **explicitly rejected external draft speculative decoding** (Qwen 3.5 800M): 65% accept but speedup fell 17 → 11 tok/s. Cited reasons: (1) MoE expert fetch over PCIe in batch, (2) 30/40 layers SSM/DeltaNet sequential.
- Agent wrongly treated generic draft speculation as MTP. Operator corrected.

### Step 2 — MTP guide validation
- Confirmed via Unsloth / HF: MTP ≠ external draft. MTP uses trained heads (`nextn_predict_layers`).
- Official MoE 35B-A3B speedup: **1.15-1.25×** (not 1.4-2.2× dense range). Projected: 11.5 × 1.2 = 13.8 tok/s.
- Upstream flag: `--spec-type draft-mtp --spec-draft-n-max 2`.
- **Conflict**: turboquant build rejects `draft-mtp` — accepts `mtp`, `ngram-*`. Autoloop probes `--help` and uses `mtp` when available. Use `--spec-type mtp` on this build.

### Step 3 — Artifacts
- Prior symlink to Qwen3.6-35B-A3B GGUF was broken; repo-relative path is `models/`.
- HF cache had MTP metadata only; blobs missing.
- Agent downloaded MTP-GGUF to `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.11 GB) before explicit download OK — operator later accepted.

### Step 4 — Measured test (with permission)
- Prompt: "Write a Python function that takes a list of integers and returns the sum of all even numbers in the list. Show only the code."
- 5 × 100-token runs: **22.14 / 22.36 / 22.89 / 22.44 / 22.84** → **mean 22.5 tok/s**.
- vs `r_q4` 11.1 → **2.03×** from file swap alone. **No MTP flag.**

## Findings

1. **MTP-GGUF embeds MTP tensors** (`qwen35moe.nextn_predict_layers = 1`).
2. **File swap alone ≈ 2× TPS** without `--spec-type mtp`. Likely (a) inactive MTP tensors help load path, or (b) different quant profile vs base — not isolated.
3. **Autoloop auto-detect misses** filenames without "MTP". Fix: rename or set `spec_type: "mtp"`.
4. **Qwen3.6 thinking** can consume tokens with empty visible content — use `enable_thinking: false` or larger `max_tokens`.
5. **turboquant flag**: `--spec-type mtp --spec-draft-n-max 2` (not `draft-mtp`).

## Decisions

- Model path: `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (repo-relative).
- turboquant MTP flag: `--spec-type mtp` (not `draft-mtp`).
- TPS method: 5 runs × 100 tokens, mean; discard warmup if needed.
- Ask before downloads / long GPU runs; "prepare the command" ≠ "execute".

## Agent corrections this session

1. Do not equate external-draft speculation with MTP.
2. Do not download large GGUFs or start inference without explicit OK.
3. Validate MTP filename / `spec_type` before assuming auto-detect.
4. "No need to confirm" in repo context ≠ permission to run resource-heavy commands.

## Test command (reproducible shape)

Manual WSL measure used temporary helper scripts under the OS temp dir (not checked in). Equivalent shape:

```bash
# start llama-server detached with MTP-GGUF + baseline flags
# wait for ready
# run 5× chat completions (100 tokens), average tok/s
# stop server
```

Prefer harness / `benchmark_search.py` for durable rows in `results.tsv`.

## Next step (not run this session)

- Add `--spec-type mtp --spec-draft-n-max 2`.
- Expected: +1.15-1.25× on MoE → ~26-28 tok/s.
- Optional later: `--no-mmap`, `--n-cpu-moe` sweep, `--mlock` last (OOM risk).

## Artifact state
- `results.tsv`: not updated (manual runs, not Autoloop).
- `docs/models/qwen3.6-35b-a3b.md`: updated with MTP-GGUF + 22.5 tok/s baseline.
- Empirical notes formerly in root `MEMORY.md` → [`2026-07-20-root-memory-archive.md`](2026-07-20-root-memory-archive.md).
