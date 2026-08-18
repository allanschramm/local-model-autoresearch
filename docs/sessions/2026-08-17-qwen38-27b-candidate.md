# 2026-08-17 — Qwen3.8-27B coding-loop candidate research

Research + planning session. No Trials run. Handoff plan → `models/TASK.md`.

## Goal

Find the best model to serve the coding harness issue-loop (Night profile: 65k+ ctx,
`agentic_coding`) on this rig, and the fastest Fingerprint for it. Two win paths:
(a) a bigger dense model with decent TPS and (b) higher TPS / bigger ctx on supported
models. Scope decisions locked this session:

- Objective: **max inference performance (TPS ↑, ctx ↑, VRAM/RAM ↓) without losing IQ**.
  IQ is measured by the repo axes (coding-10 + agentic), never assumed. Engine flags
  cannot raise IQ; quantization can lower it — the guardrail is a same-task shootout
  against the incumbent.
- **Gemma-4 family excluded** from candidates (operator decision: agentic behavior in
  the harness — rushes/gives up instead of iterating; not an inference-flags problem).
- Harness-side IQ extraction (J-Space-style led state, verifier loops) **parked** until
  a harness fork (Pi Agent / OMP). Evaluated: the J-Space capability report is a
  technique reference (report-only repo, CC BY-ND, single-run numbers, API-only target
  models), not a dependency.

## Hardware

discrete 8 GB-class NVIDIA (Baseline `VRAM_LIMIT_MB` 7900), Windows, 32 GB-class RAM,
upstream llama.cpp `b10375` (CUDA 13.3).

## Findings

### Qwen3.8-27B (candidate)

- Apache-2.0, released 2026-08-13/14. `Qwen3_5ForConditionalGeneration` (GGUF arch
  tag `qwen35` — same family already run on b10375, Qwen3.5-9B-UD precedent).
- 27B dense, 64 layers, **hybrid SSM + attention (full attention every 4th layer)** —
  delta-net layers carry recurrent state, not KV → ctx is memory-cheap.
- Native ctx 262144 (YaRN → 1M). Thinking mode by default (`reasoning_effort`
  xhigh/medium/low; harness `REASONING_BUDGET` / `REASONING_PRESERVE` knobs apply).
- **Embedded MTP block** (`nextn_predict_layers=1`) per the quantizer pack →
  `--spec-type draft-mtp`. Verify per-file with `gguf_has_mtp()` after download.
- Text-only GGUF: vision projector (`mmproj`) is a separate file — no ViT in VRAM.

### quimmedes/Qwen3.8-27B-XYZ quant map (v2 recipe, imatrix, upstream llama.cpp)

| File | BPW | GiB | Fits 7900 MiB? |
|---|---:|---:|---|
| `Qwen3.8-27B-Q1Q-XYZ-v2.gguf` | 1.98 | 6.15 | ✅ ~1.7 GiB KV/headroom |
| `Qwen3.8-27B-Q1Z-XYZ-v2.gguf` | 2.24 | 7.12 | ⚠️ borderline |
| `Qwen3.8-27B-Q2-XYZ-v2.gguf` | 2.56 | 8.15 | ❌ over (no-spill) |
| `Qwen3.8-27B-Q3-XYZ.gguf` | 3.26 | 10.39 | ❌ |
| `Qwen3.8-27B-ULTRA-XYZ-v2.gguf` | 4.93 | 15.76 | ❌ |

- **Key tensors kept BF16** (attention K/V, SSM params, MTP head) — low-bit files keep
  the reasoning-critical tensors at full precision. Not brute-force 1-bit.
- Author warning: below ~2.5 BPW a reasoning model can get stuck in the thinking loop
  on some prompts. Mitigation kit: DRY sampler (`--dry-multiplier 0.8 --dry-base 1.75
  --dry-allowed-length 2`), `--temp 0.6 --top-p 0.9 --top-k 40 --repeat-penalty 1.10`.
- Author's reference configs use `--parallel 1`, `-ctk/-ctv q8_0`, `-fa on`, `-kvu`,
  `--spec-type draft-mtp,ngram-mod` composite, `--spec-ngram-mod-n-*`, `--reasoning-budget`.

### Flag probes (TBD vs b10375 `--help` / setup-check)

- Composite `--spec-type draft-mtp,ngram-mod` — contradicts
  `docs/discovery/speculative-decoding-formats.md` ("one spec-type only"); verify.
- `-kvu`; `--spec-ngram-mod-n-match/min/max`; `--spec-draft-p-min`; DRY sampler flags.
- Same author ships a Pi-harness thinking-budget extension (harness-side, no engine change).

### KV estimate

Hybrid arch (full attention 1-in-4 layers) → KV per token ≈ 25% of a dense 27B; 64k
feasible at q4_0-class KV, 128k needs turbo KV. Exact bytes via `gguf_kv_f16_mb()`
post-download. Hard ctx target: ≥100k, prefer 131072.

### Dead ends (already measured, do not re-run)

DFlash / DSpark on MoE+CPU-offload: −49% / −48.5% (2026-08-07/12 sessions). Separate
draft models fail on hybrid/MoE+SSM. DSpark-in-SGLang only for dense on-GPU targets.

## Plan (handoff)

1. Download `Qwen3.8-27B-Q1Q-XYZ-v2.gguf` → `models/quimmedes/qwen3.8-27b-xyz/`
   (text-only, skip mmproj). Optional Q1Z for the IQ-vs-fit tier.
2. Validation: `gguf_has_mtp`, `gguf_kv_f16_mb` @64k/128k, engine load on b10375,
   flag probes (composite spec, `-kvu`, DRY, reasoning-budget).
3. 3-way shootout (same tasks, one Fingerprint each): Qwen3.8-27B-Q1Q vs
   `Kwaipilot_KAT-Coder-V2.5-Dev-IQ4_XS.gguf` (claw 0.6000 / cod 0.640) vs
   `POCKET-35B-Q3_K_M.gguf` (claw 0.6667 / cod 0.615) — coding-10 + Claw full.
4. Winner → performance pass: MTP n_max 1–6, KV formats, CTX 64k→128k, threads,
   reasoning-budget.
5. Night profile selection on the winner (`agentic_coding` axis) for issue-loop serving.

## Errors

None — research only, no Trials run this session.

## Decisions

- Scope locked: performance-without-IQ-loss; IQ guardrail = same-task shootout.
- Gemma-4 excluded from candidates.
- Q1Q primary download; Q2 excluded (over budget); no mmproj (text-only use case).
- No harness-side extraction work until a Pi harness fork.

## Open questions (TBD)

- Composite `draft-mtp,ngram-mod` support on b10375.
- Q1Q vs Q1Z IQ delta — worth the extra ~1 GiB VRAM? Answerable only in the shootout.
- MTP acceptance rate at long ctx on this hybrid arch.

## Sources

- quimmedes/Qwen3.8-27B-XYZ (quant pack README + tree)
- Qwen/Qwen3.8-27B model card
- unsloth/Qwen3.8-27B-GGUF · AtomicChat/Qwen3.8-27B-GGUF (quant availability)
