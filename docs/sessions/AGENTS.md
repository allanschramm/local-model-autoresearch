# AGENTS.md — docs/sessions

## Purpose
Single-day empirical session logs. Captures what was run, on which hardware, with which config, what was measured, decisions taken, and errors encountered. Used as **reproducibility evidence** for the user-facing guides in `docs/discovery/` and as a memory of approaches that did or did not work.

## Ownership
- Owned by: `local-model-autotuning` developers and operators.
- Stable contracts: file naming (`YYYY-MM-DD-<topic>.md`), section shape (Goal / Hardware / Commands / Findings / Errors).

## Local Contracts
- **One file per session or per significant sub-iteration of a session.** Don't merge multiple sessions.
- **Verbatim tool outputs** are preferred over paraphrased summaries. The point is reproducibility.
- **Errors and corrections are first-class.** When an approach was wrong, log it explicitly so future operators don't repeat.
- **No absolute user/checkout paths** (`C:\Users\…`, `/mnt/c/Users/…`). Use repo-relative paths.
- **No machine inventory.** Do not record which GGUFs/aliases are present, kept, or deleted on disk. Scores + GGUF basenames only (root AGENTS). Alias names/ports stay out — use basenames in tables.
- **No operator hardware fingerprint.** No GPU SKU (e.g. product marketing names), exact `nvidia-smi` MiB totals, hostnames, or personal paths. Hardware section: `discrete_gpu` / `unified_memory`, Baseline `VRAM_LIMIT_MB`, OS family, llama.cpp engine/tag — not which SKU sits in the machine.
- **No operator PII.** No personal names, hostnames, or identity links. Say "operator" / "the operator host" if a person must be referenced.
- **Private leftovers** (disk lists, alias registry, absolute paths, identity) go into existing gitignored `models/` notes (`REMOVED.md`, `aliases/REMOVED.md`) — never duplicate session files there.
- **Do not edit a session log after the session is complete** except to fix typos or scrub contract violations above. Add a follow-up file for new work.
- **No external-source URLs in technique claims** (per `docs/models/` rules — methodology names allowed, citations not).

## Work Guidance
- New session → new file with `YYYY-MM-DD-<topic>.md` filename.
- Captured data: tool output (results.tsv excerpts, logs), measured TPS, config tested, errors hit, decisions taken, who approved what.
- Cross-link to related session logs and to model cards / ADRs / discovery guides when relevant.

## Verification
- Each file has: Goal, Hardware, Setup, Commands (reproducible), Findings, Errors, Decisions.
- Hard numbers (TPS, score, VRAM) reported as measured, not estimated.
- "Correções M3" or similar self-correction sections are encouraged.

## Child DOX Index
- [`2026-08-12-qwen36-dflash-tps.md`](./2026-08-12-qwen36-dflash-tps.md) — Qwen3.6-35B Q3 DFlash vs MTP vs no-spec @ 65k; DFlash dead end on 8 GB-class + `n-cpu-moe`.
- [`2026-08-08-thinking-claw-harness-fix.md`](./2026-08-08-thinking-claw-harness-fix.md) — Agentic ignored `reasoning_content` + `max_tokens=512`; Ornith UD claw 0.3333→0.9333; thinking remasure policy.
- [`2026-08-07-qwen36-35b-dflash-tps.md`](./2026-08-07-qwen36-35b-dflash-tps.md) — Qwen3.6-35B Q4 DFlash vs MTP vs no-spec TPS smokes @ 32k; harness MoE VRAM fixes; max-TPS brainstorm.
- [`2026-08-02-qwen36-35b-unsloth-100k.md`](./2026-08-02-qwen36-35b-unsloth-100k.md) — Complete Qwen3.6 35B-A3B Unsloth no-spec pipeline at 100k with full expert CPU offload.
- [`2026-08-02-qwythos-claude-mythos-100k.md`](./2026-08-02-qwythos-claude-mythos-100k.md) — Complete Claude-Mythos pipeline at 100k and the feasible Qwythos comparison boundary.
- [`2026-08-02-qwythos-v2-mtp-b16-vram.md`](./2026-08-02-qwythos-v2-mtp-b16-vram.md) — Strict Qwythos v2 MTP matching-batch A/B rejected at the physical-VRAM gate.
- [`2026-08-02-qwythos-v2-normal-100k.md`](./2026-08-02-qwythos-v2-normal-100k.md) — Complete no-MTP Qwythos v2 pipeline at 100k with Turbo2 and batch/ubatch 16/8.
- [`2026-08-02-qwythos-v2-normal-100k-vram.md`](./2026-08-02-qwythos-v2-normal-100k-vram.md) — Qwythos v2 normal 100k Trial rejected at the physical-VRAM gate with batch/ubatch 32/16.
- [`2026-08-02-qwythos-v2-mtp-n4-preflight.md`](./2026-08-02-qwythos-v2-mtp-n4-preflight.md) — Qwythos v2 MTP n=4 rejection at 100k by the physical-VRAM preflight gate.
- [`2026-08-02-qwythos-v2-mtp-100k.md`](./2026-08-02-qwythos-v2-mtp-100k.md) — Complete Qwythos v2 embedded-MTP pipeline at 100k with Turbo2 and batch/ubatch 32/16.
- [`2026-08-02-research-gap-closure.md`](./2026-08-02-research-gap-closure.md) — Primary-source closure of the repo's web-resolvable TBD/data gaps (model cards, vLLM deep-dive, 8 GB guide); local-only gaps listed separately.
- [`2026-08-01-ornith-turboquant-131k.md`](./2026-08-01-ornith-turboquant-131k.md) — Complete Ornith 131k pipeline on TurboQuant+ Turbo2, including MTP failures and the 7.2 GB no-spec winner.
- [`2026-08-01-ornith-turboquant-100k.md`](./2026-08-01-ornith-turboquant-100k.md) — TurboQuant+ release Trials for Ornith embedded MTP at 100k, including effective KV/MTP flags and VRAM-gate results.
- [`2026-08-01-turboquant-release-research.md`](./2026-08-01-turboquant-release-research.md) — Official TurboQuant+ prebuilt release, Windows/CUDA assets, KV tiers, MTP support, and upstream relationship.
- [`2026-06-19-alias-system.md`](./2026-06-19-alias-system.md) — Alias system setup and design.
- [`2026-06-19-mtp-baseline.md`](./2026-06-19-mtp-baseline.md) — MTP baseline benchmarking and verification.
- [`2026-06-19-whichllm-coding.md`](./2026-06-19-whichllm-coding.md) — whichllm evaluation on coding benchmarks.
- [`2026-06-19-whichllm-plan.md`](./2026-06-19-whichllm-plan.md) — whichllm search and selection planning.
- [`2026-06-19-whichllm-source-deepdive.md`](./2026-06-19-whichllm-source-deepdive.md) — whichllm source code analysis.
- [`2026-06-23-4bench-integration.md`](./2026-06-23-4bench-integration.md) — 4bench evaluation harness integration.
- [`2026-06-26-ornith-baseline-and-validation.md`](./2026-06-26-ornith-baseline-and-validation.md) — Ornith model baseline validation.
- [`2026-06-29-beellama-tcq-copyspec-dflash-iq3.md`](./2026-06-29-beellama-tcq-copyspec-dflash-iq3.md) — BeeLlama TCQ, CopySpec, and DFlash experiments.
- [`2026-07-01-dense-model-validation.md`](./2026-07-01-dense-model-validation.md) — Dense model execution validation.
- [`2026-07-01-gemma4-v2-q3km-validation.md`](./2026-07-01-gemma4-v2-q3km-validation.md) — Gemma 4 v2 Q3_K_M validation.
- [`2026-07-01-ornith-1.0-9b-analysis.md`](./2026-07-01-ornith-1.0-9b-analysis.md) — Ornith 1.0 9B detailed benchmark analysis.
- [`2026-07-06-windows-model-up.md`](./2026-07-06-windows-model-up.md) — Windows model launcher (`model-up`) validation.
- [`2026-07-20-llama-cli-validation.md`](./2026-07-20-llama-cli-validation.md) — llama-cli execution & validation log.
- [`2026-07-20-root-memory-archive.md`](./2026-07-20-root-memory-archive.md) — Root memory archive and empirical notes.
- [`2026-07-20-small-model-tps-matrix.md`](./2026-07-20-small-model-tps-matrix.md) — Small-model MTP TPS empirical matrix (8 GB).
- [`2026-07-23-nanbeige42-tps-matrix.md`](./2026-07-23-nanbeige42-tps-matrix.md) — Nanbeige4.2-3B arch fork + KV/batch TPS matrix (8 GB).
- [`2026-07-23-lfm2.5-8b-a1b-validation.md`](./2026-07-23-lfm2.5-8b-a1b-validation.md) — LFM2.5-8B-A1B Q4_K_M validation + full-VRAM vs exps→CPU A/B.
- [`2026-07-24-claw-full-smoke-high.md`](./2026-07-24-claw-full-smoke-high.md) — Claw-Eval full queue + historical Val Score ceiling (Laguna 0.6667).
- [`2026-07-24-claw-full-top-tps.md`](./2026-07-24-claw-full-top-tps.md) — Claw-Eval full on top-TPS trio (LFM 1.2B/8B, Gemma E4B).
- [`2026-07-24-coding-10-claw-leaders.md`](./2026-07-24-coding-10-claw-leaders.md) — coding-10 on Laguna / LFM 1.2B / Ornith-9B (+ VRAM-kill @ 65k).
- [`2026-07-24-lcb-patch-gambiarra.md`](./2026-07-24-lcb-patch-gambiarra.md) — LCB-only remeasure + in-place results.tsv patch (symlink/timeout fixes).
- [`2026-07-24-lfm2.5-1.2b-ctx-kv-matrix.md`](./2026-07-24-lfm2.5-1.2b-ctx-kv-matrix.md) — LFM2.5-1.2B claw-quick ctx/KV matrix (65k f16 preferred).
- [`2026-07-25-claw-full-pending-queue.md`](./2026-07-25-claw-full-pending-queue.md) — Claw-Eval full on six pending aliases.
- [`2026-07-26-pocket-35b-pipeline.md`](./2026-07-26-pocket-35b-pipeline.md) — POCKET-35B Q3_K_M validation → claw-full 0.6667 → coding 0.615.
- [`2026-07-26-bonsai-coding-vs-pocket.md`](./2026-07-26-bonsai-coding-vs-pocket.md) — Bonsai coding-10 0.455 vs POCKET 0.615.
- [`2026-07-26-pocket-26b-pipeline.md`](./2026-07-26-pocket-26b-pipeline.md) — POCKET-26B Q4_K_M; claw-full 0.20 / coding 0.49.
- [`2026-07-27-incomplete-vectors-pareto.md`](./2026-07-27-incomplete-vectors-pareto.md) — complete incomplete vectors; Ornith Q3 A/B; Day ADR 0008; Qwen coding reject.
- [`2026-07-27-kat-coder-v2.5-dev-pipeline.md`](./2026-07-27-kat-coder-v2.5-dev-pipeline.md) — KAT-Coder IQ4_XS; claw-full 0.6000 / coding 0.640.
- [`2026-07-28-qwen35-4b-mtp-claw-full.md`](./2026-07-28-qwen35-4b-mtp-claw-full.md) — Qwen3.5-4B-MTP claw-full 0.2667; vector complete (coding 0.385).
- [`2026-07-28-ornith-9b-deepreinforce-claw-full.md`](./2026-07-28-ornith-9b-deepreinforce-claw-full.md) — ornith-1.0-9b-Q4_K_M claw 0.4000 @ 65k; vector complete.
- [`2026-07-28-qwen35-9b-mtp-claw-full.md`](./2026-07-28-qwen35-9b-mtp-claw-full.md) — Qwen3.5-9B-MTP claw 0.2000; vector complete (coding 0.495).
- [`2026-07-31-day-model-candidates-100k.md`](./2026-07-31-day-model-candidates-100k.md) — primary-source shortlist for new 100k+ DAY candidates on the 8 GB rig; no downloads or Trials.
- [`2026-08-01-day-models-131k-pipeline.md`](./2026-08-01-day-models-131k-pipeline.md) — full 131k throughput, coding-10, and Claw full results for Nemotron 3 Nano and Granite 4.0/4.1 candidates.
- [`2026-08-01-ornith-mtp-100k-preflight.md`](./2026-08-01-ornith-mtp-100k-preflight.md) — Ornith 9B embedded-MTP rejection at the 100k context floor by the physical-VRAM hard gate.
