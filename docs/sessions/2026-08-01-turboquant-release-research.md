# TurboQuant+ prebuilt release research

## Goal

Determine whether the official TurboQuant+ release can provide a disposable Windows/CUDA runtime for long-context KV compression, without keeping another source checkout or building locally.

## Hardware

- Target: Windows x64 with an NVIDIA GPU.
- No binary was downloaded or executed in this research pass.

## Setup

- Repository: [`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)
- Latest release observed on 2026-08-01: [`tqp-v0.3.0`](https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0), published 2026-07-12.
- Release commit: [`30d6881eb97be0844b77ff7bc93175e15972d689`](https://github.com/TheTom/llama-cpp-turboquant/commit/30d6881eb97be0844b77ff7bc93175e15972d689).

## Commands

Primary-source metadata was queried through the GitHub API:

```powershell
gh api repos/TheTom/llama-cpp-turboquant/releases/latest
gh api repos/TheTom/llama-cpp-turboquant/git/ref/tags/tqp-v0.3.0
gh api repos/TheTom/llama-cpp-turboquant/contents/common/arg.cpp?ref=tqp-v0.3.0
gh api repos/TheTom/llama-cpp-turboquant/contents/common/speculative.cpp?ref=tqp-v0.3.0
```

## Findings

### Prebuilt Windows runtime

The release publishes `turboquant-plus-tqp-v0.3.0-windows-x64-cuda12.4.zip` (835,899,907 bytes). The [release notes](https://github.com/TheTom/llama-cpp-turboquant/releases/tag/tqp-v0.3.0) identify it as Windows x64 / CUDA 12.4; the [tagged README](https://github.com/TheTom/llama-cpp-turboquant/blob/tqp-v0.3.0/README.md) says CUDA runtime DLLs are bundled.

Therefore, this target does not require cloning or compiling the fork: download and unpack the release, then point the harness at its `llama-cli`, `llama-server`, and related binaries. A source build remains necessary for ROCm/HIP or a custom CUDA architecture.

Other published binaries are Linux x64 CPU, Linux x64 Vulkan, and macOS arm64 Metal. There is no Windows Vulkan or Windows CPU-only asset in this release.

### KV TurboQuant flags

The fork exposes the KV types through the standard `-ctk` / `--cache-type-k` and `-ctv` / `--cache-type-v` arguments. The [tagged implementation](https://github.com/TheTom/llama-cpp-turboquant/blob/tqp-v0.3.0/common/arg.cpp) includes:

- `turbo2`: about 2 bits per element; most aggressive.
- `turbo3`: about 3.5 bits; documented default V-side sweet spot.
- `turbo4`: about 4.5 bits; least aggressive turbo tier.

The [official recommendation](https://github.com/TheTom/llama-cpp-turboquant/blob/tqp-v0.3.0/README.md#kv-cache-quantization-runtime) is asymmetric KV, keeping K more precise than V:

```text
safe:        --cache-type-k f16  --cache-type-v turbo4
default:     --cache-type-k q8_0 --cache-type-v turbo3
aggressive:  --cache-type-k q8_0 --cache-type-v turbo2
```

The project explicitly discourages beginning with turbo-compressed K. For Ornith at 100k, `q8_0/turbo3` is the documented first useful memory-focused candidate; `q8_0/turbo2` is the next candidate only after quality validation.

### MTP and speculative decoding

The release tag contains `draft-mtp` in the accepted `--spec-type` values and a concrete MTP implementation in [`common/speculative.cpp`](https://github.com/TheTom/llama-cpp-turboquant/blob/tqp-v0.3.0/common/speculative.cpp). The release notes also include an upstream fix for a Gemma-4 E4B MTP + Flash Attention crash. Thus the binary contains MTP support; this is not only TurboQuant KV support.

This source evidence does not prove that Ornith embedded MTP plus `q8_0/turbo3` is correct or faster on the operator host. That exact model/backend/fingerprint still requires the normal validation and full pipeline. The release also adds DFlash, but that is a separate speculative mode and does not replace embedded `draft-mtp`.

### Relationship to upstream llama.cpp

The [official README](https://github.com/TheTom/llama-cpp-turboquant/blob/tqp-v0.3.0/README.md#status) describes the fork as additive, continuously synced from `ggml-org/llama.cpp`, approximately 300 commits ahead, and not upstreamed. Existing llama.cpp formats and model families are intended to remain supported, but `tqp-v0.3.0` is a fixed fork snapshot, not interchangeable with whatever commit the local upstream submodule currently uses.

Keeping only upstream cloned is therefore viable if the TurboQuant release is treated as a versioned external binary toolchain rather than a source tree. Trial fingerprints must identify the TurboQuant release/tag so its results are not mixed with upstream-engine results.

## Errors

- No runtime validation was performed.
- The GitHub release web page reports six assets because GitHub also exposes automatic source archives; the API lists four uploaded prebuilt binaries.
- No primary-source result was found for the exact Ornith 9B MTP + TurboQuant + Windows CUDA 12.4 combination.

## Decisions

- Use the official `tqp-v0.3.0` Windows CUDA 12.4 archive instead of cloning/building this fork.
- Keep the existing upstream `llama.cpp/` checkout as the only source checkout.
- Treat TurboQuant as a distinct engine fingerprint.
- Start Ornith MTP at 100000 context with `q8_0/turbo3`; escalate to `q8_0/turbo2` only if the first full-quality vector passes and more KV reduction is needed.
