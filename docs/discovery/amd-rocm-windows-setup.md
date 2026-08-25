# AMD ROCm GPU Setup for llama.cpp on Windows

## Problem

AMD Radeon GPUs (RDNA 2/3) on Windows require the **HIP SDK** for llama.cpp's ROCm backend (`ggml-hip.dll`) to function. Without it, inference silently falls back to CPU-only — the GPU appears in Task Manager as "Shared GPU memory" (system RAM mapped through PCI-e), not as actual GPU compute. Symptoms:

- "VRAM" usage in Task Manager is actually Shared GPU memory (RAM via PCI-e BAR)
- System RAM + pagefile thrash → PC freeze under load
- `llama-bench --list-devices` reports `Available devices: (none)`
- `ggml-hip.dll` fails to load (Windows error 126 = missing dependency)
- Generation speed ~5 t/s (CPU) instead of ~33+ t/s (GPU)

## Prerequisites

| Component | Requirement |
|-----------|-------------|
| **GPU** | AMD Radeon RDNA 2 or RDNA 3 (gfx1030+ / gfx1100+) |
| **OS** | Windows 10/11 |
| **Driver** | AMD Adrenalin (latest) |
| **HIP SDK** | AMD ROCm HIP SDK 7.x+ |
| **llama.cpp** | Build with ROCm/HIP backend (`ggml-hip.dll`) |

## Setup Steps

### 1. Install AMD HIP SDK

Download from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html

Install at minimum: **HIP Runtime**, **rocBLAS**, **hipBLAS**.

Default install path: `C:\Program Files\AMD\ROCm\<version>\`

### 2. Add ROCm bin to PATH

The HIP SDK installs DLLs into its own `bin/` directory but does **not** add it to the system PATH. Without this, `ggml-hip.dll` cannot find its dependencies (`rocblas.dll`, `amdhip64_7.dll`, etc.).

```powershell
# Permanent (user-level, survives restarts)
$rocmBin = "C:\Program Files\AMD\ROCm\7.2\bin"
$currentPath = [Environment]::GetEnvironmentVariable("PATH", "User")
[Environment]::SetEnvironmentVariable("PATH", "$rocmBin;$currentPath", "User")
```

Adjust `7.2` to your installed version.

### 3. Set AUTORESEARCH_LLAMA_CPP_ROOT

Point to the parent directory containing `build-rocm/`:

```powershell
# Permanent (user-level)
[Environment]::SetEnvironmentVariable(
    "AUTORESEARCH_LLAMA_CPP_ROOT",
    "c:\Dev\local-model-autotuning\llama.cpp-releases\upstream\b10448",
    "User"
)
```

### 4. Verify GPU detection

```powershell
# Open a NEW terminal (picks up PATH changes)
& "$env:AUTORESEARCH_LLAMA_CPP_ROOT\build-rocm\bin\llama-bench.exe" --list-devices
```

Expected output:
```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 8 GB-class):
  Device 0: RDNA 2, gfx1032, VMM: no, Wave Size: 32, VRAM: 8 GB-class
load_backend: loaded ROCm backend from ...\ggml-hip.dll
Available devices:
  ROCm0: RDNA 2 8 GB-class (~8 GB total, ~7.8 GB free)
```

If devices still show `(none)`, check:
- PATH includes the ROCm `bin/` directory
- `rocblas.dll` exists in that directory
- The `ggml-hip.dll` version matches the HIP SDK version

## Performance Reference (RDNA 2 8 GB-class, Qwen3.5-9B-MTP-Q4_K_M, ROCm b10448)

| Test | No MTP | draft-mtp n=4 |
|------|--------|---------------|
| pp512 | 551 t/s | — |
| tg128 | 33.4 t/s | 40.4 t/s |

## Known Limitations

- **VMM: no** on RDNA 2 (gfx1032) — HIP Virtual Memory Management unsupported. Extra host-side staging buffers are allocated, adding ~2–4 GB RAM overhead.
- **Memory bandwidth** is the tg bottleneck (224 GB/s on RDNA 2 vs 272 GB/s on an 8 GB-class Ada card), not FLOPS.
- **MTP boost is lower on ROCm** (~21%) compared to CUDA (~48%) for the same model — speculative decoding kernels are less optimized.
- **`NO_MMAP = False`** (mmap) recommended when all layers fit in VRAM — OS can reclaim file-backed pages after GPU transfer, reducing steady-state RAM.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ggml-hip.dll` error 126 | Missing ROCm runtime DLLs | Add ROCm `bin/` to PATH |
| Devices: `(none)` | HIP backend failed to load | Check PATH + DLL versions |
| Resolves to `build-cuda` | Binary resolution prefers CUDA | Ensure `has_discrete_amd()` returns True; check `llama_runner.py` ROCm priority |
| High RAM (~12 GB) despite GPU | `NO_MMAP=True` or VMM:no staging | Set `NO_MMAP=False`; ~2–4 GB overhead is unavoidable on RDNA 2 |
| "Shared GPU" in Task Manager | CPU-only fallback, WDDM mapping | Fix GPU backend (this guide) |
