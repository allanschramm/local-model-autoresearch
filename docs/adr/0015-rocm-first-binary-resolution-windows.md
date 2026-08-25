# ADR 0015 — ROCm-first binary resolution on Windows

- **Status:** accepted
- **Date:** 2026-08-16
- **Context:** `llama_runner._candidate_binary()` had a `has_discrete_amd() and not IS_WINDOWS` guard on line 125 that prevented ROCm-first binary ordering on Windows. This meant AMD GPU systems on Windows always resolved to the CUDA binary first — which silently fell back to CPU-only inference because CUDA cannot use AMD hardware. A secondary issue: a hardcoded `--device Vulkan0 --no-host -sm layer` block in `_build_cmd()` forced the inferior Vulkan backend on AMD Windows systems, bypassing native ROCm/HIP even when available.

## Decision

1. **Remove the `not IS_WINDOWS` guard** from `_candidate_binary()`. AMD GPUs on Windows now get ROCm-first binary ordering, matching the existing Linux behavior.
2. **Remove the Vulkan0 hardcode** from `_build_cmd()`. The ROCm backend handles device selection natively — the hardcode was a pre-ROCm workaround that interfered with proper HIP GPU compute.
3. **Add `--no-reasoning-preserve`** for `reasoning_preserve=False`. Previously only `True` emitted the flag; `False` was silently ignored (the command never contained the negation flag).

## Consequences

- AMD GPUs on Windows with the HIP SDK installed now use ROCm/HIP natively (measured: 33.4 t/s tg on an RDNA 2 8 GB-class card vs ~5 t/s CPU-only and ~13 t/s on Vulkan).
- Systems without the HIP SDK still fall back gracefully — `ggml-hip.dll` fails to load and llama.cpp uses the next available backend (CPU).
- NVIDIA-only systems are unaffected: `has_discrete_amd()` returns False, so the CUDA-first ordering is preserved.
- The `reasoning_preserve=False` fix resolves the pre-existing test failure `test_build_cmd_reasoning_preserve_off`.
