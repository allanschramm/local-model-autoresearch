# MTP Speculative Decoding & llama-bench Guide

This guide explains how to verify and baseline speculative decoding (Multi-Token Prediction / MTP) on your local hardware before starting a full model autotuning run.

---

## 1. MTP Verification Concept

MTP speeds generation by drafting multiple tokens per step. Packaging differs by model family:

1. **Embedded `nextn`:** tensors inside the main GGUF (e.g. local `Qwen3.5-9B-UD-Q4_K_XL.gguf`, Ornith/Mythos `*-MTP*.gguf`). Flags: `--spec-type draft-mtp --spec-draft-n-max N` (no draft file).
2. **External draft:** separate assistant GGUF (Gemma-4 E4B → `models/draft/mtp-gemma-4-E4B-it.gguf`). Main UD file has **no** `nextn`. Flags: add `--spec-draft-model <draft>`.

Detect: scan GGUF metadata for `nextn` / `gemma4-assistant`. Inventory + measured TPS: [small-model-mtp-tps.md](./small-model-mtp-tps.md).

Prefer the harness gate (`run_llama_bench_validation`) over raw `llama-bench` — bench binaries do not accept MTP draft flags. Canonical `n_max` on the operator host for speed matrix: **4** (not 2).

The `n_max=4` canon is for **dense** targets. On small-active MoE with `--n-cpu-moe` (Qwen3.6-35B-A3B, Gemma-4-26B-A4B), MTP is a measured **loss** — keep n ≤ 1 or skip spec; see [speculative-decoding-formats.md](./speculative-decoding-formats.md) §4b.

---

## 2. Test MTP Speedup via `llama-cli`

> Paths below assume a prebuilt release under `llama.cpp-releases/upstream/<tag>/build-cuda/bin/` (release first — see [`docs/llamacpp-toolset.md`](../llamacpp-toolset.md) → Install). Adjust `<tag>` to the release you extracted.

### Embedded MTP (Qwen) — WITH:
```bash
llama.cpp-releases/upstream/<tag>/build-cuda/bin/llama-cli.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --single-turn
```

### Gemma external draft — WITH:
```bash
llama.cpp-releases/upstream/<tag>/build-cuda/bin/llama-cli.exe \
  -m models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --spec-draft-model models/draft/mtp-gemma-4-E4B-it.gguf \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --single-turn
```

### WITHOUT MTP (Baseline):
```bash
llama.cpp-releases/upstream/<tag>/build-cuda/bin/llama-cli.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -p "Explain quantum computing in one sentence." \
  -n 64 -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --single-turn
```
(Omit `--spec-type` / draft flags entirely, or set `--spec-type none` if your build requires an explicit disable.)

If the MTP run throws `failed to create MTP context` (or similar), that GGUF has no usable MTP heads — download a `*-MTP-GGUF` variant or the matching Gemma draft.

---

## 3. Base Benchmarking via `llama-bench`

To benchmark the base raw performance of the models at a specific context depth (e.g., 65k context) without speculative decoding, use `llama-bench`.

### Test 9B Model at 65k context depth:
```bash
llama.cpp-releases/upstream/<tag>/build-cuda/bin/llama-bench.exe \
  -m models/Qwen3.5-9B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -d 65000 -p 0 -n 128
```

### Test 35B MoE Model at 65k context depth (CPU Offload):
```bash
llama.cpp-releases/upstream/<tag>/build-cuda/bin/llama-bench.exe \
  -m models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \
  -ngl 99 -ncmoe 40 -fa on -ctk q4_0 -ctv q4_0 -d 65000 -p 0 -n 128
```

### Test 26B MoE Model at 65k context depth (CPU Offload):
```bash
llama.cpp-releases/upstream/<tag>/build-cuda/bin/llama-bench.exe \
  -m models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf \
  -ngl 99 -ncmoe 30 -fa on -ctk q4_0 -ctv q4_0 -d 65000 -p 0 -n 128
```
*Note: Gemma-4-26B requires `--n-cpu-moe 30` on 8 GB GPUs to prevent VRAM swapping and thrashing.*

---

## 4. Verdict (2026-08-22): dense win, MoE+CPU-offload loss

Dense MTP nearly **doubles** throughput on this 8 GB-class rig: Ornith 38.7→56.3 (+46%), Qwen3.5-9B 38.7→57.3 (+48%), Gemma-4 E4B 67.6→122.0 (+80%) — consistent with external ~1.9× on dense ([PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), [Frontier Lab](https://thefrontierlab.ai/mtp-defaults-are-a-trap/)).

MoE with `--n-cpu-moe` inverts it (Qwen3.6-35B-A3B): 65k n0 **27.8** → n1 **27.6** (−0.7%) → n2 **24.6** (−11%); 131k n4 **18.1** (−34%, below harness `TPS_FLOOR` 20). Acceptance 0.54 → 0.11. Mechanism: every draft/verify token fetches CPU-offloaded expert weights over PCIe (~144 MB/token), MTP adds a separate KV cache (~2.5 GiB), and acceptance collapses with context depth. The workspace estimator was also fixed (est 9104 → direct 4243 MB; MoE MTP workspace now 0).

**Rule of thumb:** tune MTP only on dense; on MoE+`n_cpu_moe`, `--spec-draft-n-max 1` at most, `none` preferred at deep context. Full evidence, URLs, and the falsifiable sweep probe: [speculative-decoding-formats.md](./speculative-decoding-formats.md) §4b.
