# Codacus MoE-fork validation: Ornith-1.5-35B + Qwen3.6-35B-A3B (2026-08-24)

## Goal

Validate the `thecodacus/llama.cpp` `perf` branch (MoE expert profile cache + host-register/prefetch env levers) on the operator rig: does the claimed prefill gain replicate, does the expert profile cache help decode, and is output token-identical?

## Hardware

- discrete_gpu, 8 GB-class VRAM (`VRAM_LIMIT_MB` ≈ 8192), Windows
- 32 GB host RAM — central to this session: full-expert-offload configs sit at the host-RAM envelope

## Setup

- Fork: `thecodacus/llama.cpp` branch `perf`, HEAD `0ac3d9b` ("overlap CPU cold-expert and GPU hot-expert computation in the MoE cache (#9)"), shallow-cloned to gitignored vendor dir `llama.cpp-codacus/`
- Build: MSVC v143 BuildTools + CUDA toolkit v13.3, `cmake -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`; targets `llama-cli llama-server llama-bench llama-moe-trace`; 485/485 OK
- Stock reference: pinned upstream release b10549 (`AUTORESEARCH_LLAMA_CPP_ROOT` untouched throughout)
- Models: `Ornith-1.5-35B-Q4_K_M.gguf` (20.21 GiB weights, experts 18.65 GiB = 41 layers × 256 exp × 8 active), `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (20.60 GiB, experts 18.22 GiB = 40 layers × 256 × 8); both arch `qwen35moe`

## Commands

```bash
# baseline benches (fork, features off)
./llama.cpp-codacus/build-cuda/bin/llama-bench.exe -m <model>.gguf \
  -ngl 99 -ncmoe <N> -fa 1 [-lm none] -p 2048 -n 0 -r 3 -b 2048 -ub 2048   # prefill
  ... -p 512 -n 128 -r 3 -b 2048 -ub 512                                   # decode

# env levers
GGML_CUDA_REGISTER_HOST=1 [GGML_SCHED_PREFETCH_EXPERTS=1] <same bench>

# trace profiles (2 prompts merged by concat)
MOE_TRACE_OUT=<csv> ./llama.cpp-codacus/build-cuda/bin/llama-moe-trace.exe \
  -m <model>.gguf -ngl 99 -ncmoe 36 -lm none -fa 1 -c 4096 -n 512 -p "<prompt>"

# cache serve (server only — see Findings #4)
./llama.cpp-codacus/build-cuda/bin/llama-server.exe -m <model>.gguf --port <P> \
  -ngl 99 -ncmoe 36 -lm none -fa on -c 4096 --load-mode none --temp 0 \
  --moe-cache-profile <merged.csv> --moe-cache-slots <N>
```

## Findings

1. **Prefill env levers replicate at FULL offload** — Ornith-1.5-35B `-ncmoe 41`: pp2048 704 → **1257 t/s (+78 %)** with both env vars (±7 tight variance); +53 % over stock b10549 (820). Decode tg128 28.6 → 30.2 (+5.6 %).
2. **`GGML_SCHED_PREFETCH_EXPERTS=1` is harmful at PARTIAL offload** — Qwen3.6 @ `-ncmoe 32`: pp2048 908–958 baseline → **562 t/s (−40 %)** with prefetch alone; `REGISTER_HOST` alone neutral (933 ± 36 vs 958 ± 4). The second-stream prefetch only pays when every layer's experts cross PCIe.
3. **Stock b10549 beats the fork's own features-off baseline** on Ornith (pp2048 820 vs 704; tg128 32.2 vs 28.6) — fork base predates b10549 gains; quote feature gains vs the fork-off control, not vs stock.
4. **Expert profile cache activates via llama-server only** — `common_model_params_to_llama()` plumbs `--moe-cache-*` in server/batched-bench/fit-params but **not llama-cli** (flags accepted, silently ignored). Activation line at `-lv 5`: `init_moe_expert_cache: expert cache: 36 layers x 16 slots, 1046.12 MiB uploaded to CUDA0`.
5. **Cache is NOT token-identical on this build and slows decode at achievable coverage** — Qwen3.6 @ `-ncmoe 36`, slots=16 (6.25 % coverage): deterministic controls reproduce byte-identical text across restarts (sha `6ecd9857…`), both cache runs reproduce a different hash (`3c427fef…`, first divergence char 59) and decode drops 32.4 → 29.4 t/s (−9 %). Hot/cold two-pass split changes FP reduction order → not bit-exact here; 16 slots is deep in the <15–20 % coverage dead zone per the scaling rule, so slower is consistent.
6. **Memory envelope is the governing constraint on this rig**: 20 G-class MoE with `-ncmoe = block_count` puts ~18.7 GiB on host RAM → total commit at physical-RAM edge; any extra allocation tips into pagefile thrash (SSD 100 %, CPU/GPU idle). Safe fast path found: **`-ncmoe 36 --load-mode none`** (VRAM peak 5407 MiB, run 29 s vs 585–785 s while thrashing).

## Errors

- Operator PC crash (0xe0000008): cancelled a mid-load background job → orphaned `llama-cli.exe` kept allocating; next launch overlapped a second 20 G load. Protocol now: verify no llama processes + VRAM baseline before every launch; never cancel mid-load; strictly one model-resident process.
- Two pagefile-thrash episodes during cache experiments at `-ncmoe 32` (VRAM oversubscription backed by WDDM shared memory → process commit hit ~40 GB, working set ~14–22 GB). Mitigation built: PowerShell watchdog (`circuit-breaker`) kills llama processes when free RAM < 2500 MB; fired twice, machine survived.
- `llama-bench` rejects `--no-mmap` (use `-lm none`); fork CLI uses `--temp` (double dash) and `-no-cnv` (single dash); `llama-moe-trace` CMake target is `llama-moe-trace`.
- Port collision took down two servers once (started second before stopping first) — hub-managed servers must be stopped before relaunch on same port.

## Decisions

- Fork stays evaluation-only in `llama.cpp-codacus/` (gitignored); `AUTORESEARCH_LLAMA_CPP_ROOT` and the b10549 pin untouched; nothing entered `results.db` (engine eval, not a Trial).
- Adopted rig rules: env-lever sweeps only at full offload; partial-offload configs need VRAM-headroom check (peak ≤ ~5.4 GiB of ~6.9 usable) + circuit breaker running; cache trials only above ~25 % slot coverage are worth GPU time on 256-expert models.
- Follow-up candidate (not scheduled): re-test cache bit-exactness claim at higher coverage on a smaller expert set where ≥50 % slots fit.
