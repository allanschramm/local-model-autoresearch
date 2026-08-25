# 2026-08-25 — Qwen3.8-4B-Distill Q4_K_M KV-quant A/B (@65k) + q8_0@131k + TurboQuant turbo3 — all `on_front`

## Goal
Single-variable A/B of KV cache precision for the winning `Qwen3.8-4B-Q4_K_M.gguf`: same ctx across variants for real comparison (per operator directive), ladder `q4_0 → q8_0 → f16`, context floor 65536, full Trials (Claw-15 + coding-10). Follow-ups: `q8_0 @131072` (operator: "Run 131k ctx q8") and `turbo3 @65536` on the TurboQuant+ fork engine (operator: "Trial turbo quant to A/B too if it works with this model").

## Hardware
- `discrete_gpu`, 8 GB VRAM class, Windows host
- Baseline `VRAM_LIMIT_MB` 8000 → clamp 7676 (physical − keepout 512); runtime kill ceiling = `min(limit, physical − keepout)` = 7676 MB **device-wide NVML used** (20 ms sampling) — NOT free-at-start−headroom (that is preflight-only, issue #10)
- Engine: upstream `b10549` for q4_0/q8_0/f16; `turboquant@tqp-v0.3.0` (fork commit `30d6881`, CUDA 12.4) for turbo3
- Dense `block_count 33` Gated DeltaNet hybrid (≈¼ dense KV footprint), `fa on`, `b 512 / ub 128 / t 8 / tbd 8`, `jinja`, `cont-batching`, `cache-reuse 256`
- Preflight bypass: `AUTORESEARCH_SKIP_FREE_CLAMP=1` (documented operator escape; effective budget = configured 7676; runtime monitor untouched)

## Setup
- Model: `models/empero-ai/Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf` (2.8G)
- Sampler unchanged from card: `TEMP 0.6 / TOP_P 0.95 / TOP_K 20 / MIN_P 0.0 / REP 1.0`
- Fingerprints: every variant its own `config_json` (ctx × KV in fingerprint); store point identity = GGUF basename per bucket (ADR 0012)

## Commands
```powershell
# one per variant; only KV_CACHE / KV_CACHE_K / KV_CACHE_V and CTX_SIZE edited in config.py Baseline
$env:AUTORESEARCH_SKIP_FREE_CLAMP = "1"
# q4_0 / q8_0 / f16 @65536 and q8_0 @131072 — engine b10549 (default AUTORESEARCH_LLAMA_CPP_ROOT)
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "trial Qwen3.8-4B-Q4_K_M kv-<type> ctx<size>"
# turbo3 @65536 — fork engine
$env:AUTORESEARCH_LLAMA_CPP_ROOT = "<repo>\llama.cpp-releases\turboquant\tqp-v0.3.0"
.\venv\Scripts\python.exe benchmark_search.py --agentic-full --desc "trial Qwen3.8-4B-Q4_K_M kv-turbo3 ctx65536 tqp"
```
`--agentic-full` defaults: Claw full = 15 tasks + coding-10 (HE/MBPP/LCB/BC × 10). No `autoloop`; no command timeouts.

## Findings

### A/B @65536 (same ctx, engine b10549)

| KV | bench tg | agentic | coding (HE/MBPP/LCB/BC) | min | peak | trial |
|----|----------|---------|--------------------------|-----|------|-------|
| q4_0 | 73.4 | 0.7333 (11/15) | 0.5800 (1.0/0.7/0.4/0.1) | 0.5800 | 4.5 GB | `621d0719` |
| q8_0 | 72.9 | **0.8000** (12/15) | 0.5900 (0.9/0.7/0.5/0.1) | 0.5900 | 5.0 GB | `5a70ee91` |
| f16 | **74.4** | 0.7333 (11/15) | **0.6250** (1.0/0.8/0.5/0.0) | **0.6250** | 5.8 GB | `55aa739e` |
| turbo3 | 71.8 | **0.8000** (12/15) | 0.5150 (0.8/0.7/0.4/0.0) | 0.5150 | 4.7 GB | `38fc1cec` |

- q8_0: best agentic at 65k (+0.067 vs q4_0), +0.01 coding, −0.5 t/s
- f16: best coding + best min-axis (0.6250) + fastest bench; +1.3 GB VRAM vs q4_0
- turbo3 (fork engine): ties q8_0 agentic, lowest coding + TPS of the set; VRAM ≈ q4_0 — turbo compression buys nothing on this arch at 65k (GDN KV already ≈¼ dense). Row is **not engine-isolated** (type exists only in the fork; `binary_version` = `turboquant@tqp-v0.3.0`)

### q8_0 @131072 (b10549)

| KV | bench tg | agentic | coding | min | peak | trial |
|----|----------|---------|--------|-----|------|-------|
| q4_0 @131072 (ref) | 74.9 | 0.8667 | 0.6400 | **0.6400** | 5.4 GB | `6069530a` |
| q8_0 @131072 | 73.8 | 0.8667 | 0.4900 | 0.4900 | 6.4 GB | `54e49297` |

- q8_0@131k ties agentic (T050 passed 1.00 — it failed all three 65k runs; task-level variance), loses coding (HE 6/10 vs 10/10) and TPS. Ran clean under the 7676 monitor ceiling (est 7601 passed with SKIP_FREE_CLAMP).

### VRAM estimator vs measured (the preflight-math miss)
- Preflight `estimate_vram_mb` over-reads this hybrid-attention arch's f16 KV ≈1.5× (f16@131072 est 11403 MB vs measured ~7.6 GB device-inclusive, ~6.8 GB model+server). `VRAM_KB_PER_TOKEN_F16 = 80.0` flat base not corrected by `gguf_kv_f16_mb` here; measured ≈30.5 KB/token f16 on this arch.
- Consequence: f16@131072 is physically loadable (measured, no OOM, generation OK) but unlaunchable via harness (est > hard-clamped budget 7676; no skip-preflight flag/env exists — verified both CLIs, all `AUTORESEARCH_*` reads, config template, repo-wide grep; only `AUTORESEARCH_SKIP_FREE_CLAMP=1` exists and it cannot overcome the estimator). q8_0@131072 est 7601 passed only with the clamp bypass.
- Runtime monitor confirmed: device-wide NVML used vs `min(limit, physical−keepout)`; the 65k set peaked 4.5–5.8 GB (1.9 GB slack at f16), q8_0@131k peaked 6.4 GB (1.2 GB slack). The razor-thin case (f16@131k, 13 MB under the ceiling) never ran.

### Store semantics (why every row reads `on_front`)
- Statuses recomputed after the set (`scripts/recompute_status.py`); all rows stay `on_front` **by design**: recompute merges per GGUF basename per budget bucket (ADR 0012, `recompute.py` merge on basename) — all five KV rows share the merged super-vector (ctx max, tps 95.7, agentic 0.8667, coding 0.64) and no other model dominates it. Compare per-row values, not status.

## Errors / Corrections
- First VRAM estimate batch used the harness estimator → wrong by design (see above); operator directed real measurement (nvidia-smi / direct server probe) — done, tables above.
- Initial plan tried 131072 for the set; f16@131k deterministically MODEL_REJECTED at preflight (est 11403 > 7676, no env override) → per operator "restart with 65k, don't wait for approval" the whole A/B ran at 65536; q8_0@131k follow-up then ran standalone.
- Task variance is visible across the set: T046/T050/T053 flip pass/fail between runs (ctx-exceed at 65k, tool-call JSON parse errors, length-stops). Single-run scores are noisy at ±1–2 tasks (±0.07 agentic); treat rank order as direction, not fine resolution.
- TurboQuant: upstream has no turbo types (b10549 `--help` allowed list + `./llama.cpp` source grep zero hits + `docs/llamacpp-toolset.md`). Fork release `tqp-v0.3.0` supports turbo2/3/4 for K and V; turbo3 loaded the qwen35 GDN GGUF and generated cleanly (probe), turbo2 has a prior native-crash history (`2026-08-01-ornith-turboquant-100k.md`).

## Decisions
- **Family pick unchanged: `Qwen3.8-4B-Q4_K_M.gguf` @131072 `q4_0`** — min-axis 0.6400 beats every 65k variant (f16 0.6250 closest) and q8_0@131k (0.4900). Baseline restored to this Fingerprint after the set (verified `load_config`).
- No KV-quant change to the daily-driver recipe; alias notes updated with the A/B table.
- TurboQuant not adopted for this model: no win at 65k; fork-only engine confounds future comparisons. Candidate venue: huge-ctx or KV-dominated models on a dedicated VITRIOL-engine A/B.

## Open questions
- **TBD:** f16 @131072 (measured-fits, 13 MB monitor margin) — unlaunchable until the estimator is fixed; operator owns the fix (`VRAM_KB_PER_TOKEN_F16` / `gguf_kv_f16_mb` path for hybrid attention).
- **TBD:** turbo4 / turbo2 on this model at 65k (turbo4 untested; turbo2 crashed on Ornith in the 2026-08-01 session).
- **TBD:** KV-precision effect at 131k once the estimator fix lands (q8_0@131k measured; f16@131k pending).

## References
- Card: `docs/models/qwen3.8-4b-distill.md`
- Sessions: `2026-08-23-qwen38-4b-distill-full-trial.md` (winning ref row), `2026-08-01-ornith-turboquant-100k.md` (turbo history)
- Logs: `autoresearch/runners/logs/llama-server-20260825-*.log` (5 server logs), `agentic-20260825-*.json` (5)
- Store: `results.db` / `results.tsv` — `621d0719`, `5a70ee91`, `55aa739e`, `54e49297`, `38fc1cec`

## Verification
- Measured: bench tg, per-task agentic PASS/FAIL, coding per-dataset, NVML peaks, store rows, recomputed statuses. VRAM probes: direct engine launches + nvidia-smi (real peaks above). No SKU/PII/alias names; memory-class + engine tags only.
