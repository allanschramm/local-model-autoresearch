# 2026-08-31 — Qwen3.5-9B EXL3 probe: base measurement + fair-MTP attempt (hardware-blocked)

## Goal

Operator linked `turboderp/Qwen3.5-9B-exl3` and asked (1) what EXL3 is, (2) how to test it on this rig. Research pass (engine-side) landed in [`docs/discovery/fastest-tps-inference-engine.md`](../discovery/fastest-tps-inference-engine.md) §3/§3.1; this log captures the measurement session: EXL3 3.00bpw base TPS, the fair EXL3-MTP attempt via self-quantization, and the two hardware walls that blocked it. Operator approved the manual (out-of-harness) measurement pass explicitly; no autoloop, no harness edits.

## Hardware

- `discrete_gpu`, Baseline `VRAM_LIMIT_MB` 7676-class (±200 MB WDDM variance near ceiling), Windows, 32 GB host RAM.
- Harness engine pinned `upstream/b10549` — untouched all session (ExLlamaV3 runs outside it).

## Setup

Fresh dedicated venv (`.auto/venv-exl3/`, Python 3.12; the default launcher Python is 3.14 — no wheel exists for it):

```
py -3.12 -m venv .auto/venv-exl3
.auto/venv-exl3/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
.auto/venv-exl3/Scripts/python.exe -m pip install https://github.com/turboderp-org/exllamav3/releases/download/v1.4.5/exllamav3-1.4.5%2Bcu128.torch2.10.0-cp312-cp312-win_amd64.whl
.auto/venv-exl3/Scripts/python.exe -m pip install triton-windows
```

Pack download (hf CLI; sizes verified via `--dry-run --format json`, byte-identical after):

```
./venv/Scripts/hf.exe download turboderp/Qwen3.5-9B-exl3 --revision 3.00bpw --local-dir models/turboderp/Qwen3.5-9B-exl3/3.00bpw
```

Measurement instrument: `.auto/exl3_bench.py` (Job API, greedy, batch 1, 511–512 tokens, 32k q4/q4 cache, `nvidia-smi` peak watcher at 250 ms). BF16 source for the self-quant: `models/Qwen/Qwen3.5-9B` (4 shards ≈ 18 GiB, hf CLI, shard sizes verified).

## Commands

```
# base leg (turboderp pack)
.auto/venv-exl3/Scripts/python.exe .auto/exl3_bench.py models/turboderp/Qwen3.5-9B-exl3/3.00bpw

# self-quantize (final recipe: head unquantized, vision excluded)
.auto/venv-exl3/Scripts/python.exe .auto/convert_runner.py -i models/Qwen/Qwen3.5-9B -o models/selfquant/Qwen3.5-9B-exl3-mtp-3.00bpw -w .auto/exl3-convert-work -b 3.0 -hb 16 -vb 16

# fair-MTP leg (failed at load)
.auto/venv-exl3/Scripts/python.exe .auto/exl3_bench.py models/selfquant/Qwen3.5-9B-exl3-mtp-3.00bpw --mtp
```

Note: convert.py cannot run via `python -m` from the wheel; the runner wrapper imports `exllamav3.conversion.convert_model` (`prepare` → `main`). Long runs must be hub-start managed processes — the bash device hard-kills background jobs at 300 s (this session's first quant died that way mid-run).

## Findings

### Base leg (turboderp pack, MTP-off — pack carries zero MTP tensors)

```
RESULT {"mtp": false, "new_tokens": 511, "time_generate": 10.30, "tps": 49.61,
        "peak_vram_mb": 5437, "stop_reason": null}
```

- **EXL3 3.00bpw base = 49.6 t/s**, peak VRAM **5437 MB** (incl. torch context) vs keepout 7676 MB — big headroom.
- Repo targets (results store, fair matrix 2026-07-20): llama.cpp base **38.7** / MTP **57.3**; bench_tg **67.5** @32k+MTP.
- → EXL3 base **+28 %** over llama.cpp base, **−13 %** vs llama.cpp-with-MTP.
- Load 35.9 s first run (triton JIT), 3.4 s warm.
- Pack audit: 1363 tensors, **zero `mtp`/`nextn`** — turboderp's conversion predates converter MTP support (card requires ≥ v0.0.23).

### Fair-MTP attempt (self-quant with head kept)

- convert.py v1.4.5 wires the MTP component automatically for `qwen3_5` (source-verified); a completed run produced a 1399-tensor pack with **39 `mtp.*` tensors** (3 bpw text / 4 bpw MTP), zero vision.
- **Wall 1 — head quantization exceeds 32 GB host RAM on every path.** Head default 6 bpw (mul1 Viterbi): 55 GB process commit, progress frozen at 0 % for 45 min. 8 bpw retry: free RAM 24 → 0.8 GB in ~2 min, pagefile 5.8 GB, killed per circuit-breaker protocol (free RAM recovered to 24.2 GB in 3 s). The quantizer's head-stage state scales with vocab size; 248K vocab ≈ 2× Llama-class.
- **Wall 2 — unquantized head doesn't place on 8 GB VRAM.** `-hb 16` pack = 7.18 GiB (bf16 head 2.03 + bf16 embed 2.03 — embeddings are hard-coded 16-bit copies — + 3 bpw layers ≈ 3.2) and load fails:

```
RuntimeError("Insufficient VRAM in split for model and cache")   # model_ls.py _load_autosplit, 16 s in
```

  Turboderp's own pack fits only because its vision tower is skipped at text-gen and its head is 4-bpw quantized — exactly the quantization Wall 1 forbids.
- **No escape hatches in v1.4.5**: `Model.load` exposes no embedding-offload (only MoE CPU-offload exists in `model_ls.py`); convert.py has no embedding-bits option (full flag list checked).
- **Ecosystem scan** (`hf models ls`, all authors): the only MTP-bearing EXL3 pack for the family is `komeijishiki/DeepSeek-V4-Pro-Qwen3.5-9B-EXL3-6.50bpw-H8-V8-MTP8` (published 2026-09-01T01:08Z, unsloth finetune base) — 9.6 GB total, ~3 GB over the placement ceiling. No ≤4 bpw base-model pack with a head exists.

## Errors

1. **ImportError at first load**: `bc_dsa.py` imports triton kernels unconditionally — `triton-windows` is mandatory on Windows (README says "suboptimal"; reality is "won't import"). Fix: `pip install triton-windows`.
2. **Missing calibration corpus**: the wheel omits `exllamav3/conversion/standard_cal_data/*.utf8` (c4/code/multilingual/technical/tiny/wiki) — fetch from the GitHub tag into the installed package, else calibration fails.
3. **300 s background-job kill**: first quantizer run died mid-run ("Command timed out after 300 seconds") — checkpointed, but motivated the hub-start migration. Lesson: any multi-minute run goes through hub start, never bare bash background.
4. **Head-quant thrash (twice)**: 55 GB commit (6 bpw) and 0.8 GB free (8 bpw) — both killed. Same pattern as the 2026-08-24 codacus incidents; circuit-breaker protocol applies to quantizers, not just llama-server.
5. **Bench OOM on the hb16 pack** (Wall 2 above).
6. **Self-corrections**: a work-dir wipe cost ~45 min of reusable layer tensors (`-r --override_anyway` would have reused them); edit-tool hunk drift corrupted the bench script three times (GEN_TOKENS eaten, `ids` assignment eaten, `out = {` opening eaten) — each caught and repaired, `py_compile` verified; an ad-hoc watcher script violated the no-temp-scripts rule — killed and deleted, harness-native flow restored.

## Decisions

- Vision excluded from the self-quant (`-vb 16`): dead weight for a TPS test (~1–1.5 GiB VRAM + ~10 min quant).
- Head left unquantized in the final pack (only path that completes on 32 GB RAM); pack retained at `models/selfquant/Qwen3.5-9B-exl3-mtp-3.00bpw/` for the day embedding-offload/quant lands upstream.
- Circuit-breaker kill authorized by protocol (free RAM 0.8 GB < 2.5 GB threshold); operator host recovered fully.
- **Verdict unchanged: llama.cpp stays the daily config.** EXL3 base +28 % is real but loses to llama.cpp-MTP; the fair EXL3-MTP number is hardware-blocked on this rig.
- Retention pending operator decision: BF16 source (`models/Qwen/Qwen3.5-9B/`, ≈ 19.3 GB) and the self-quant pack (≈ 7.2 GB) — deletable to reclaim ~26 GB, or keep for a bigger-RAM retry.

## Open questions

- Does upstream ExLlamaV3 add embedding quantization or embedding-offload? (Would make the hb16 pack fit: ~5 GiB load.)
- Will turboderp re-convert Qwen3.5-9B with a current convert.py (head kept)? The ecosystem's first MTP-bearing pack appeared the same night — the recipe works on bigger-RAM hosts.
- Is the 8 bpw head path viable on a 64 GB host, and does the resulting pack (≈ 6.5 GiB) place on 8 GB VRAM? Unverified.

## Cross-links

- [`docs/discovery/fastest-tps-inference-engine.md`](../discovery/fastest-tps-inference-engine.md) §3/§3.1 — research + measured table + verdict.
- [`docs/models/qwen3.5-9b.md`](../models/qwen3.5-9b.md) — llama.cpp card (MTP daily config, 61.6 t/s @80k).
- [`2026-08-24-codacus-fork-validation.md`](./2026-08-24-codacus-fork-validation.md) — circuit-breaker protocol origin.
