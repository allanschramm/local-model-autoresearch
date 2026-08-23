# 2026-08-23 — NEW Models Post-2026-08-02 (8 GB-class / 100K) — Qwen3.8-4B Distill Only Fit

## Goal
Find **NEW** models worth a download + `benchmark_search.py --validation` that fit the 8 GB-class rig at 100K ctx, biasing to post-2026-08-02 HF releases (newest-first) and not re-anchoring to the stale 2026-08-02 queue (`Qwen3.5-4B`, `Gemma 4 E4B`, `Qwen3-8B`, `LFM2.5-8B-A1B`). Desk research only — no GPU, no Trial.

## Hardware
- `discrete_gpu`, 8 GB VRAM class, Windows host, WSL2 Ubuntu 24.04 available
- Baseline `VRAM_LIMIT_MB` 7900 (preflight guard), target 100K ctx (prefer 131072)
- Engine pin: `llama.cpp-releases/upstream/b10549` (Gated DeltaNet / MTP-capable) per root `AGENTS.md`

## Setup (reads before web)
- Root `AGENTS.md` + `CONTEXT.md` (Pareto / Day-Night / Baseline contracts)
- `docs/discovery/best-model-8gb-vram.md` (2026-08-02, stale) + `docs/discovery/discover-models.md` (Steps 0-5)
- `models/TASK.md` queues (research-derived 2026-08-02 + requested candidates 2026-08-06)
- `results.tsv` (recent gemma-4-E4B ~73-76 TPS incomplete, Qwythos ~35-43 TPS — no complete vector for stale picks)
- `docs/sessions/2026-08-18-qwen38-27b-rejected.md` (Q1Q rejected: 27.9 TPS Day fail, ~61K Night ceiling)

## Method
1. Walked repo docs first — collected TBDs (§ Open questions).
2. Clarity gate: stale queue already answers 2026-08-02, so research gate = **no** → newest-first web fact-check required.
3. `web_search` against current sources (official HF cards, vendor blogs, leaderboards) on 2026-08-23 — every claim below cites URL + access date. External measurements labeled **external / unverified on this rig**.
4. Doc lands in `docs/sessions/` (single-day capture); no code/config/alias edits. Commit on operator go-ahead only.

## Commands (not executed — desk only)
```powershell
# Hardware gate (do first — operator-only, not run this session)
.\venv\Scripts\python.exe scripts\check_hardware.py
# Candidate planning (candidate list only)
# uvx whichllm@latest --profile coding
# llmfit plan "Qwen3.8-4B"

# Download — the one NEW fit (run only after hardware confirm)
hf download empero-ai/Qwen3.8-4B-Distill-GGUF Qwen3.8-4B-Q4_K_M.gguf --local-dir models/empero-ai/Qwen3.8-4B-Distill-GGUF
# optional A/B same dir: Qwen3.8-4B-Q5_K_M.gguf (3.161 GB), Q6_K (3.563 GB), Q8_0 (4.611 GB)

# Validation — one model at a time, no autoloop
.\venv\Scripts\python.exe benchmark_search.py --validation --desc "validate Qwen3.8-4B-Distill Q4_K_M"
# then on same Fingerprint:
# .\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "claw-full Qwen3.8-4B-Distill"
# .\venv\Scripts\python.exe benchmark_search.py --coding ... (same Fingerprint) — until complete vector for Pareto read
```
`autoloop.py` omitted — operator-only per root `AGENTS.md` ("No autonomous autoloop").

## Findings

### NEW candidates evaluated (post-2026-08-02, newest-first bias)

| Repo (HF) | Release | Arch / ctx | GGUF file (hf --dry-run 2026-08-23) | Fits 8GB + 100K? | Coding signal (primary) |
|---|---|---|---|---|---|
| **`empero-ai/Qwen3.8-4B-Distill-GGUF`** | ~2026-08-18 (5d old) | Distill of Qwen3.8-2.4T-A95B into Qwen3.5-4B arch, Gated DeltaNet 8/32 full-attn, 262K-equivalent | **2.8 GB** Q4_K_M, 3.2 GB Q5, 3.6 GB Q6, 4.6 GB Q8 (verified) | **yes** — 4-6 GB tier, ~5 GB left for KV @131K | Card headline **mixed vs Qwen3.5-4B base:** `mmlu CoT +0.199` but `gsm8k_cot -0.065`. No SWE-bench Verified / Aider / LCB — **unverified for coding agent**. |
| `deepgrove/maple-preview-GGUF` | 2026-08 update | 20B-A1B MoE, 24L 256×8, 3:1 SWA-512:GA, MIT | **5.0 GB** TQ1_0-Q4_K / **5.9 GB** TQ2_0-Q4_K (+ F16 5.4/6.3G heads) — **hf-verified** | **yes** — 5-6 GB fits; 100K TBD (ctx not published) | Reasoning MoE (ternary TQ1/TQ2 quants), no SWE/Aider — **unverified** |
| `HuggingFaceTB/SmolLM3-3B` (+ `unsloth/SmolLM3-3B-GGUF`) | 2026-07 (GGUF May-2026, updated) | dense 3B, 64K train → **128K YARN** | **1.9 GB** Q4_K_M / 1.8 GB Q4_0 (verified) | **yes** — 1.9 GB + KV @128K | LCB v4 **30.0% thinking / 15.2% no-thinking**, no SWE-bench — **external / unverified** |
| `ai9stars/G9v3-3B` | 2026-07-23 | dense 3B, 131K | **no GGUF** in `ai9stars/G9v3-3B` (safetensors 5.4+0.6 GB only — hf-verified) | yes via safetensors but **no first-party GGUF** | AA Intell 16 / coding 9.9 — weak |
| `amd/Instella-MoE-16B-A3B-Think` | 2026-08-01 | 16B/2.8B active, 27L Gated MLA, 64K max | **no GGUF** (6×5.4G safetensors — hf-verified) | **no** — 64K <100K, needs quant | Base avg 76.7 std; no SWE-bench Verified |
| `microsoft/Fara1.5-9B` | May-Jun 2026 | Qwen3.5-9B CUA fine-tune (vision→click/type), 262K | **no GGUF** (4×5.9-6.0G safetensors — hf-verified) | fits via transformers but **CUA workload** | Online-Mind2Web 63% — not coding-agent |
| `Qwen/Qwen3.8-27B` (+ GGUF forks) | 2026-08-14 | dense 27B, 262K, Apache-2.0 | **16.8 GB** Q4_K_M + mmproj (C Chungulus) | **no** — 21GB+ with KV | Vendor SWE-bench Pro 61.7 — **external / unverified** |
**Global context 2026-08-23 (web + hf):** `state-of-open-models-summer-2026` confirms Qwen dominance but weekly new-model roundup is only MiniMax-H3 + Qwen3.8-27B. **hf-verified now:** `maple-preview` is NOT 11-13 GB — actual 5.0/5.9 GB TQ quants fit 8GB class (ternary, not plain Q4). `LiquidAI/LFM2.5-DSpark` (2026-08-20) also NEW — drafters ~300M for `LFM2.5-1.2B/2.6B/8B-A1B`, 3.18× H100 / 2.87× M4 Max, identical greedy output — **engine adjacency, not a NEW base model**.

### Verdict (updated with hf-verified files)
**Two NEW GGUFs clear 8GB:** `empero-ai/Qwen3.8-4B-Distill-GGUF` Q4_K_M 2.8 GB (**primary** — 262K-equiv hybrid, architecture-fit bet) and `deepgrove/maple-preview-GGUF` TQ1_0-Q4_K 5.0 GB (**secondary** — ternary MoE reasoning bet, **unverified ctx/IQ**). `SmolLM3-3B` Q4_K_M 1.9 GB is a lightweight 128K YARN control (30% LCB thinking) — quality far below Qwen frontier, speed only. `G9v3/Instella/Fara` have **no first-party GGUF** per `hf --dry-run` — not `llama.cpp` GGUF candidates without community quant. All three are safetensors-only (require transformers/SGLang + different VRAM math).

Stale 2026-08-02 queue (`Qwen3.5-4B` etc.) remains **not re-recommended** — superseded unless both NEW bets fail locally.

### Fit math (hf-verified)
`Qwen3.8-4B-Distill` Q4_K_M 2.8 GB + hybrid KV @131K fits with 5GB+ headroom (card tier 4-6 GB). `maple-preview` TQ1_0 5.0 GB leaves ~2-3 GB for KV — 100K ceiling **TBD** (card doesn't publish ctx, TQ is ternary 1.58-bit so KV math differs). `SmolLM3` 1.9 GB leaves ~6 GB for KV @128K. No desktop 8GB tps published for any — ordering is hf-verified file size + arch heuristic, **unverified on this rig**.

## Errors / Corrections
- Initial `hf download` path `empero-ai/Qwen3.8-4B-GGUF` is 404 — **corrected** to `empero-ai/Qwen3.8-4B-Distill-GGUF` per card + `hf --dry-run` 2026-08-23.
- Initial `maple-preview` estimate 11-13 GB Q4 — **corrected** to 5.0/5.9 GB TQ1_0/TQ2_0-Q4_K per `hf --dry-run` (ternary TQ quants, not plain Q4).
- Initial prose "direct upgrade path" overstated — mixed deltas (+0.199 mmlu, -0.065 gsm8k_cot) — **softened** to architecture-fit bet; coding remains unverified.
- Anchoring to stale `best-model-8gb-vram.md` names rejected — **biased to post-2026-08 newest-first** + `hf --dry-run` verification this pass.

## Decisions
- **Primary:** `empero-ai/Qwen3.8-4B-Distill-GGUF` Q4_K_M 2.8 GB — download + `benchmark_search.py --validation` → complete vector.
- **Secondary (hf-verified fits):** `deepgrove/maple-preview-GGUF` TQ1_0-Q4_K 5.0 GB (ternary MoE) — secondary bet, ctx/IQ unverified; `HuggingFaceTB/SmolLM3-3B` Q4_K_M 1.9 GB (128K YARN) — 30% LCB, weak coding — lightweight control only.
- **Do not GGUF-trial** `ai9stars/G9v3-3B` / `amd/Instella-MoE-16B-A3B-Think` / `microsoft/Fara1.5-9B` — **no first-party GGUF** per `hf --dry-run` (safetensors-only) and/or wrong workload.
- Next: `check_hardware` → validation on primary → complete Objective Vector on same Fingerprint → `scripts/recompute_status.py` → operator decides hill-climb (no autoloop).

## Open questions
- **TBD:** Local Objective Vector for `Qwen3.8-4B-Distill` Q4_K_M @131K `q4_0` (Claw-full + coding-10, same Fingerprint) — falsifiable via validation + claw-full + coding-10 rows with identical `config_json`.
- **TBD:** Local Objective Vector for `maple-preview` TQ1_0-Q4_K 5.0 GB — ctx ceiling not published, ternary TQ impact on IQ/TPS **unverified on this rig** (falsifiable via same 3-axis vector).
- **TBD:** Whether Qwen3.8-4B distill retains 262K or 131K YaRN practical ceiling via `llama.cpp` b10549 — verify from effective server log (`--ctx-size` + KV reservation).

## References
- `huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF` (hf --dry-run 2026-08-23: 2.8/3.2/3.6/4.6 GB, tiers 4-6 GB) + README mixed eval deltas, Apache-2.0
- `huggingface.co/deepgrove/maple-preview-GGUF` (hf --dry-run 2026-08-23: TQ1_0-Q4_K 5.0G, TQ2_0-Q4_K 5.9G)
- `huggingface.co/HuggingFaceTB/SmolLM3-3B` + `unsloth/SmolLM3-3B-GGUF` (hf --dry-run 2026-08-23: Q4_K_M 1.9G, LCB 30% thinking / 15.2% no-thinking)
- `huggingface.co/ai9stars/G9v3-3B` (hf --dry-run 2026-08-23: no GGUF, safetensors 5.4+0.6G) + `artificialanalysis.ai/models/g9v3-3b` (AA 16 / coding 9.9)
- `huggingface.co/amd/Instella-MoE-16B-A3B-Think` (hf --dry-run 2026-08-23: no GGUF, 6×5.4G) + `rocm.blogs.amd.com/...`
- `huggingface.co/microsoft/Fara1.5-9B` (hf --dry-run 2026-08-23: no GGUF, 4×~6G) + `arxiv.org/abs/2606.20785` — CUA
- `huggingface.co/Qwen/Qwen3.8-27B` + `Chungulus/Qwen3.8-27B-Q4_K_M-GGUF` (16.8 GB Q4_K_M)
- `huggingface.co/blog/state-of-open-models-summer-2026` + `huggingface.co/blog/LiquidAI/lfm25-dspark` (2026-08-20: DSpark 3.18× H100, 2.87× M4 Max)
Cross-links: `docs/discovery/best-model-8gb-vram.md` (stale baseline), `models/TASK.md` queues, `docs/sessions/2026-08-18-qwen38-27b-rejected.md`, `docs/discovery/discover-models.md` Steps 0-5, `docs/sessions/AGENTS.md`.

## Verification
- Desk research only — no `llama-server`/`llama-cli` probe, no `venv` run, no Trial.
- Every claim web fact-checked 2026-08-23; external measurements labeled external/unverified.
- File naming `YYYY-MM-DD-<topic>.md`, sections per `docs/sessions/AGENTS.md`, no SKU/hostname/absolute path/PII (memory class only).
