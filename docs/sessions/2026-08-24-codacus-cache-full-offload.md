# Codacus cache, literal config on 8 GB-class: full-offload + 48 slots (2026-08-24)

Follow-up to [`2026-08-24-codacus-fork-validation.md`](./2026-08-24-codacus-fork-validation.md). That session tested the MoE expert profile cache at partial offload (`-ncmoe 36`), which put the test in the dead zone (16 slots max = 6.25 % coverage) and lost. This run executes the creator's actual serve topology — full expert offload — adapted to the operator rig.

## Goal

Does the cache help when configured as documented (`-ncmoe 99 --no-mmap` + large slot budget), once VRAM is spent on slots instead of resident experts?

## Hardware

- discrete_gpu, 8 GB-class VRAM, Windows, 32 GB host RAM; RAM circuit-breaker watchdog active.

## Setup

- Same fork build (`perf` @ `0ac3d9b`), same merged trace CSV (code + chat prompts, 512 tokens each).
- Deviation from creator guide: `--spec-type draft-mtp` omitted — the plain `UD-Q4_K_M` file carries no `nextn` tensors (verified via gguf-py); his MTP leg requires the separate MTP-GGUF packaging. Context 4096 not 65536 (slot budget priority).

## Commands

```bash
# control (no cache)
llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port <P> -ngl 99 -ncmoe 99 \
  -lm none -fa on -c 4096 --no-mmap --temp 0
# cache
... --moe-cache-profile qwen36-merged.csv --moe-cache-slots 48 -lv 5
# request: /v1/chat/completions, fixed essay prompt, max_tokens 192, temperature 0
```

Activation line: `init_moe_expert_cache: expert cache: 40 layers x 48 slots, 3499.12 MiB uploaded to CUDA0` — all 40 layers eligible under full offload (vs 36/40 at `-ncmoe 36`).

## Findings

| Config (-ncmoe 99, --no-mmap) | decode t/s | prompt t/s |
| --- | --- | --- |
| control, no cache | 27.2 | 13.7 |
| cache, 48 slots (~19 % coverage) | **31.0–31.2 (+14 %)** | 24.4 |

1. **The cache helps at full offload and hurts at partial offload** — direction flips exactly along the split point. At `-ncmoe 36` it cost −9 % (prior session); at `-ncmoe 99` with 48 slots it gains +14 %. Mechanism: slot uploads only pay when the experts they replace would otherwise cross PCIe every token.
2. **Gain magnitude tracks the coverage rule**: ~19 % coverage → +14 % decode, versus creator's +66 % at ~50 % coverage on a larger card. An 8 GB card cannot reach 50 % coverage on a 256-expert 35B — the slot budget caps near ~50/layer.
3. **Token-identity still fails**: control hash stable across configs/splits (`6ecd9857…`), cache runs produce a different hash (`44c97f41…`). The two-pass hot/cold split is not bit-exact on this build regardless of offload mode.
4. Prompt-side numbers are not comparable (cold first-request on control vs warm server on cache run); only the decode column supports conclusions.
5. Host RAM held at full offload without `REGISTER_HOST` pinning (watchdog never fired); the earlier thrash incidents involved pinning or VRAM oversubscription, not full offload per se.

## Addendum: full ladder incl. MTP (same day)

- **Provenance correction**: the plain `unsloth/Qwen3.6-35B-A3B-GGUF` UD-Q4_K_M carries NO MTP head (`block_count=40`, max tensor `blk.39`, zero `nextn` fields). The card line "MTP: trained with multi-steps" is a training note, not shipped draft weights. The head ships in the standalone `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` repo under the SAME basename (`block_count=41`, `nextn_predict_layers`, full `blk.40`). Same-basename-different-content across repos — check provenance before benchmarking.
- Ladder on the `-MTP-GGUF` file, `-ncmoe 99 --no-mmap -c 4096`, fixed prompt, greedy, 192 tokens:

| Config | prompt t/s (1465-tok prefill) | decode t/s |
| --- | --- | --- |
| control (no spec, no cache) | 268.7 | 28.5–28.8 |
| cache only, 48 slots (~19 % coverage) | 239.3 | 30.6–30.7 |
| MTP only (`--spec-type draft-mtp --spec-draft-n-max 2`) | 284.2 | 32.4–34.8 |
| **MTP + cache stacked** | 257.6 | **41.5–42.1 (+46 % vs control)** |

- Prompt-processing verdict: the cache **costs ~11 % prefill** at this coverage (slot-map overhead + dual-pass bookkeeping during large batches); MTP is prefill-neutral (+6 %, within noise); combined −4 %. Decode: levers compound — cache +7 %, MTP +16 %, stacked +46 %.
- **False-crash correction**: three earlier "stacked config crashes" were **self-inflicted** — the RAM circuit-breaker watchdog (2.5 GB floor) force-killed the server during the transient host-RAM dip of a large-prompt prefill (`circuit-brake.log` shows all three kills; exit `0x7FFFFFFF`, no WER record). Re-run without the watchdog: no crash, stable across repeated requests (41.5 / 42.1 t/s). Lesson: an external watchdog can masquerade as target instability — always audit kill logs before blaming the process.
- Free RAM at end of stacked run: ~2.5 GB — full offload + MTP + cache sits at the host-RAM envelope on 32 GB; a watchdog for this config should floor near 1–1.5 GB, not 2.5 GB.
- **Ornith-1.5-35B-Q4_K_M ladder** (same protocol; `nextn_predict_layers=1` with a full MoE `blk.40` draft layer — embedded MTP present, contrary to expectation):

| Config | prompt t/s (1465-tok) | decode t/s |
| --- | --- | --- |
| control | 183.3 | 29.6–31.0 |
| cache only, 48 slots (40 layers packed, ~19 % coverage) | 237.0 | **36.3–36.9 (+18–22 %)** |
| MTP only | 262.7 | 21.4–24.2 (**−25 %**) |
| MTP + cache stacked | 234.0 | 29.4–29.8 (≈ control — MTP cancels the cache gain) |

- Model-dependent spec verdict: embedded MTP *helps* Qwen3.6 (+16 %) but *hurts* Ornith-1.5 (−25 %, low draft acceptance) under full offload. The cache gain is the reliable lever on both; stacking must be validated per model, not assumed.
- Cache packs 40 of 41 layers (draft layer excluded from routing during normal decode).
- **Why Ornith MTP loses — acceptance stats** (`-lv 5`, server-reported): Qwen3.6 draft acceptance **0.734**, mean accepted len 2.47; Ornith **0.380**, mean 1.76, per-position `(0.660, 0.100)` — the first drafted token accepts decently, the second almost never (10 %), while `n-max 2` pays draft+verify for both every cycle. `--spec-draft-n-max 1` recovers most of the loss (26.8–27.7 t/s vs −25 %) but still nets ~−5 % vs control: even 66 % pos-0 acceptance does not cover the draft-pass overhead at this model's decode speed. Recipe: Ornith → cache only, no spec; Qwen3.6 → stack both.

## Decisions
