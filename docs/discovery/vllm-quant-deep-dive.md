# vLLM Quantization Deep Dive — Per-Format Mechanics (Primary Sources)

Research date: 2026-08-02, against vLLM `main` @ `06018507` (2026-08-02), vLLM docs in-tree (`docs/features/quantization/`), `vllm-gguf-plugin` `main`, and llm-compressor docs. Companion to the overview ref [vllm-quantization.md](./vllm-quantization.md); NVFP4 format structure lives in [nvfp4-quantization.md](./nvfp4-quantization.md).

Every claim cites its primary source. Source-code paths are relative to `github.com/vllm-project/vllm/blob/main/` unless stated. Hardware notes use SM (compute capability): Volta 7.0, Turing 7.5, Ampere 8.0/8.6, Ada 8.9, Hopper 9.0, Blackwell 10.0/12.0.

---

## 1. Format mechanics

### 1.1 Quant method registry (source of truth)

`vllm/model_executor/layers/quantization/__init__.py` maps every accepted `quantization=` name to a config class:

| `quantization=` name(s) | Config |
|---|---|
| `awq`, `awq_marlin`, `auto_awq` | `AutoAWQConfig` |
| `gptq`, `gptq_marlin`, `auto_gptq` | `AutoGPTQConfig` |
| `fp8` | `Fp8Config` |
| `fbgemm_fp8`, `fp_quant` | deprecated (still loadable) |
| `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed` | ModelOpt configs |
| `compressed-tensors` | `CompressedTensorsConfig` (llm-compressor output) |
| `bitsandbytes`, `experts_int8`, `quark`, `moe_wna16`, `torchao`, `inc`, `humming` | one each |
| `mxfp4`, `gpt_oss_mxfp4`, `deepseek_v4_fp8` | MXFP4/DeepSeek-specific |
| `online` + shorthands `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `int8_per_channel_weight_only`, `nvfp4_per_token` | `OnlineQuantizationConfig` |

Source: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/__init__.py and https://github.com/vllm-project/vllm/blob/main/vllm/config/quantization.py (`_ONLINE_SHORTHANDS`).

### 1.2 AutoAWQ — 4-bit weight-only (W4A16)

- Scheme: weights only, INT4 with zero-point, group size 128 typical; activations stay FP16/BF16. `TYPE_MAP = {4: uint4}` → 4-bit only. Reference: AWQ paper (arXiv 2306.00978). Source: `.../quantization/auto_awq.py` (`AutoAWQConfig`, `TYPE_MAP`).
- Kernel dispatch: on CUDA, if `check_marlin_supported(quant_type, group_size, zero_point)` and not batch-invariant → Marlin-class kernels via `choose_mp_linear_kernel` (Conch SM80+, Exllama, or Marlin); else Triton fallback `AutoAWQLinearMethod`. XPU/CPU get dedicated paths (CPU → CPUWNA16LinearKernel). MoE layers → `AutoAWQMoEMarlin` or WNA16 MoE kernels. Source: `auto_awq.py` `get_quant_method`.
- AWQ checkpoints pack weights in a non-standard bit order; vLLM repacks them to the standard GPTQ-like layout at load (`_convert_awq_to_standard_format`). Source: `auto_awq.py`.
- Min SM: 75 (Turing). Config file: `quant_config.json` (with `quant_method: "awq"`). Source: `auto_awq.py` (`get_min_capability`, `get_config_filenames`).
- Tooling: the AutoAWQ library is **deprecated**; AWQ is now produced via llm-compressor (`examples/awq`). Source: https://docs.vllm.ai/en/latest/features/quantization/auto_awq.html (warning box).
- Ecosystem: 6500+ AWQ repos on HF (`https://huggingface.co/models?search=awq`), legacy e.g. `TheBloke/Llama-2-7B-Chat-AWQ`. Source: auto_awq doc page.

### 1.3 GPTQ — 4/8-bit weight-only (W4A16/W8A16)

- Scheme: INT4 or INT8 weights, group sizes −1/32/64/128, symmetric or asymmetric, `desc_act` (activation ordering). Per-module dynamic overrides (`bits`, `group_size`, `desc_act` per layer) supported. Sources: `.../quantization/auto_gptq.py`, `.../quantization/utils/gptq_utils.py` (`override_config`), and the GPTQModel doc page (dynamic quantization).
- Kernel dispatch: Marlin-class via `choose_mp_linear_kernel`; **Machete** (SM90+, Hopper) and **Conch** (SM80+) are the high-perf paths; Exllama fallback. Sources: `auto_gptq.py`, `.../kernels/linear/mixed_precision/machete.py` (`get_min_capability` 90, "requires compute capability of 90"), `.../kernels/linear/mixed_precision/conch.py` (`get_min_capability` 80, group sizes −1/128).
- Marlin kernel constraints: group sizes [−1, 32, 64, 128]; weight types gated per device capability (`query_marlin_supported_quant_types`). Source: `.../quantization/utils/marlin_utils.py`.
- **Marlin-24 kernels are gone from vLLM main.** The dedicated `gptq_marlin_24.py` (act-order GPTQ via Marlin-24: `uint4b8`/`uint8b128` weight types, group sizes [−1, 128], tile 16) still existed at tag `v0.9.0` (`vllm/model_executor/layers/quantization/gptq_marlin_24.py`) but has been removed from current main. Act-order GPTQ now routes through the generic GPTQ/Marlin path.
- Min SM: 60 (Pascal). Config file: `quantize_config.json`. Source: `auto_gptq.py`.
- Tooling: GPTQModel (ModelCloud) — `pip install gptqmodel`. Ecosystem: 5000+ GPTQ repos on HF; example `ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2`. Source: https://docs.vllm.ai/en/latest/features/quantization/gptqmodel.html.

### 1.4 bitsandbytes — 4/8-bit, dequant-at-runtime

- Two modes read from HF config `quantization_config`: `load_in_8bit` (LLM.int8 style, outlier threshold 6.0, fp32 CPU offload option) and `load_in_4bit` (`nf4` or `fp4` quant type, double-quant flag). Both pre-quantized checkpoints and in-flight quantization (`--quantization bitsandbytes`) are supported; no calibration needed. Sources: `.../quantization/bitsandbytes.py`, https://docs.vllm.ai/en/latest/features/quantization/bnb.html.
- Execution: **dequantizes weights to FP16 at runtime** (`dequantize_4bit` / 8-bit dequant helpers) then runs a normal GEMM — the slowest path in vLLM, not a native tensor-core path. Source: `bitsandbytes.py` (`_apply_8bit_dequant`, `dequantize_4bit`).
- Min SM: 70 (Volta). No config file. Source: `bitsandbytes.py`.
- Lifecycle: RFC #39583 "[RFC]: Migrate bitsandbytes and GGUF quantization support to OOT plugin" proposes moving bnb out-of-tree too; as of main, bnb is still in-tree. Source: https://github.com/vllm-project/vllm/issues/39583.
- Ecosystem: `https://huggingface.co/models?search=bitsandbytes`; example `unsloth/tinyllama-bnb-4bit` (pre-quantized). Source: bnb doc page.

### 1.5 GGUF — migrated out-of-tree (vllm-gguf-plugin)

- **vLLM main contains zero GGUF code.** GGUF support now lives in the OOT plugin `vllm-gguf-plugin` (`pip install vllm-gguf-plugin`). Migration tracked in RFC #39583. Sources: https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/gguf.md, https://github.com/vllm-project/vllm/issues/39583, plugin repo https://github.com/vllm-project/vllm-gguf-plugin.
- Loading: `vllm serve <hf_repo>:<quant_type>` or a local `.gguf` path; `--tokenizer <base-model>` recommended (GGUF tokenizer conversion is slow/unstable); `--hf-config-path` fallback for architectures HF can't convert; `--tensor-parallel-size` works. Source: gguf doc page.
- Which quants: the plugin accepts GGML tensor types, file-level types with no tensor type (e.g. `IQ2_M`, `MXFP4_MOE`), extended suffixes (`Q4_K_M` → `Q4_K`), and dash-prefixed custom names (e.g. `UD-Q4_K_XL`). Execution: Triton dequant kernels (standard `Q4_0`–`Q8_1`, K-quants `Q2_K`–`Q6_K`, IQ quants `IQ1_S`–`IQ4_XS`) that dequantize to FP16/BF16 before GEMM, plus llama.cpp-style CUDA vec-dot kernels with quantized (q8_1) activations (`csrc/gguf/gguf_kernel.cu`, `mmvq.cuh`, `mmq.cuh`). Sources: `vllm_gguf_plugin/gguf_utils.py`, `vllm_gguf_plugin/triton/dequantize/`, `vllm_gguf_plugin/csrc/gguf/` in the plugin repo.
- Hardware: **GPU-only** (CUDA or ROCm toolkit required; build needs the same PyTorch as vLLM). Hardware matrix row: AMD GPU ✅, x86/Arm CPU ❌. Sources: plugin README (prerequisites), https://docs.vllm.ai/en/latest/features/quantization/index.html (matrix).
- Model coverage: generic GGUF weight adapter, not an allowlist — but "not every vLLM-supported architecture works". Tested in plugin CI: Qwen 2.5 (Q6_K), Qwen 3 (Q8_0), Phi 3.5 (IQ4_XS), GPT-2 (Q4_K_M), StableLM (Q4_K_M), Gemma 3 (Q4_0), OLMoE (Q4_0), plus VLM/diffusion (Gemma 3, Z-Image-Turbo, FLUX.2-klein). Source: plugin README (tested model coverage table).
- Status: doc page warns "highly experimental and under-optimized … incompatible with other features". Source: gguf doc page.

### 1.6 LLM Compressor formats (compressed-tensors format)

Produced by llm-compressor (`pip install llmcompressor`), consumed by vLLM via `quantization="compressed-tensors"` (auto-detected from `config.json` + `model.safetensors.index.json`). Scheme reference: https://docs.vllm.ai/projects/llm-compressor/en/latest/ and https://github.com/vllm-project/llm-compressor/blob/main/docs/guides/compression_schemes.md.

| Scheme | Weights | Activations | Group size | Min SM | Notes |
|---|---|---|---|---|---|
| FP8 W8A8 (FP8_DYNAMIC) | FP8 E4M3, per-channel or per-tensor scale | dynamic per-token | — | 89 (Ada/Hopper native); SM75+ runs W8A16 weight-only via FP8 Marlin | ~2× memory cut, up to 1.6× throughput; E4M3 vs E5M2 tradeoff (range 448 vs 57344) |
| FP8 W8A8 block (FP8_BLOCK) | FP8, 128×128 block scales | dynamic per-group(128) | 128 | 89+ | requires fp8-serialized checkpoint + dynamic activation scheme (`Fp8Config.weight_block_size` constraint) |
| INT4 W4A16 | INT4, per-group | FP16/BF16 | 128 typical | >80 (Ampere+) | GPTQ/AWQ calibration; ~3.7× compression |
| INT8 W4A8 | INT4 | INT8 | — | **stale matrix** (see below) | two source paths: CUTLASS W4A8 (SM90) and XPU; CPU/Arm via dynamic4bit kernel |
| INT8 W8A8 | INT8, per-channel/per-group | INT8 dynamic/static | — | 75 (Turing+) | **not supported on Blackwell SM≥10** — use FP8 there |

Sources: in-tree docs `docs/features/quantization/llm_compressor/{fp8,int4,int8_w4a8,int8_w8a8}.md`, llm-compressor `compression_schemes.md`, `.../quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py` (`get_min_capability` 1), `.../compressed_tensors_w4a8_fp8.py` (90), `.../compressed_tensors_w8a8_int8.py` (75), `.../kernels/linear/mixed_precision/cutlass.py` (CUTLASS W4A8 SM90), `.../kernels/linear/mixed_precision/dynamic_4bit.py` (CPU-only, Arm fp32/bf16/fp16). W4A8 matrix row in the quantization README ("Arm CPU only") is **stale** — source now ships CUTLASS W4A8 on Hopper; flagged explicitly.

KV-cache and attention quantization are part of the same scheme system (`kv_cache_scheme`, `QuantizationScheme(targets=["LlamaAttention"])`); calibration done with llm-compressor one-shot. Source: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html.

### 1.7 NVIDIA Model Optimizer (ModelOpt)

- Checkpoints identified by `hf_quant_config.json` `quantization.quant_algo`. Supported: `FP8` (per-tensor weight scale, optional static act scale), `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_PB_WO` (block-scaled FP8 weight-only, 128×128), `NVFP4`, `MXFP8`, `MIXED_PRECISION`. Source: https://docs.vllm.ai/en/latest/features/quantization/modelopt.html + `.../quantization/modelopt.py`.
- Config names: `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed`. Min SMs: 80 (modelopt), 75 (modelopt_fp4 — NVFP4 experts via Marlin W4A16 on SM75+, "validated end-to-end on Tesla T4 (SM75) and A100 (SM80)" per source comment), 80 (modelopt_mxfp8), 75 (modelopt_mixed). Source: `modelopt.py`.
- Example checkpoint: `nvidia/Llama-3.1-8B-Instruct-FP8` (`quantization="modelopt"`). Source: modelopt doc page.

### 1.8 Online quantization (runtime)

- Quantizes a plain BF16/FP16 checkpoint **at load time**; no pre-quantized weights, no calibration; activations scaled dynamically each forward. Source: https://docs.vllm.ai/en/latest/features/quantization/online.html.
- CLI shorthands (`--quantization <name>`, desugared by `_ONLINE_SHORTHANDS` in `vllm/config/quantization.py`): `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `mxfp8`, `int8_per_channel_weight_only` (MoE-only), `nvfp4_per_token` (MoE-only, Blackwell + FlashInfer). Full control via `--quantization online --quantization-config '{...}'`.

### 1.9 AMD Quark

- Checkpoint-format consumer: `quantization="quark"`, min SM 70. Quark (AMD) produces weight/activation/kv-cache quantized checkpoints with AWQ/GPTQ/Rotation/SmoothQuant algorithms. Source: https://docs.vllm.ai/en/latest/features/quantization/quark.html + `.../quantization/quark/quark.py`.
- `QuarkNVFP4` scheme: group size 16, loads NVFP4 checkpoints, kernel via `init_nvfp4_linear_kernel()` — only the emulation backend is validated ("Correctness is not validated" for others per source). Source: `.../quantization/quark/schemes/quark_nvfp4.py`.

### 1.10 TorchAO

- `quantization="torchao"`, min SM 75. Consumes torchao-quantized HF checkpoints (`TorchAoConfig` in transformers, e.g. `Int8WeightOnlyConfig`); torchao nightly ≥ 10.0. Source: https://docs.vllm.ai/en/latest/features/quantization/torchao.html + `.../quantization/torchao.py`.

### 1.11 Intel Neural Compressor

- `quantization="inc"`, min SM 60. AutoRound (Intel) exports INT2/3/4/8, MXFP8, MXFP4, NVFP4 and GGUF formats. Source: https://docs.vllm.ai/en/latest/features/quantization/inc.html + `.../quantization/inc/inc.py`.

### 1.12 Others

- `humming` (Microsoft): min SM 75, GEMM kernels for FP8/FP4-class data. Source: `.../quantization/humming.py`.
- `experts_int8`: INT8 MoE experts only. Source: `.../quantization/experts_int8.py`.
- `moe_wna16`: W4A16/W8A16 MoE-only config, min SM 70. Source: `.../quantization/moe_wna16.py`.
- `mxfp4` / `gpt_oss_mxfp4`: MXFP4 (OCP MX spec, E8M0 exponent-only scales, group 32, no calibration needed for RTN) — Blackwell-native W4A4 via FlashInfer CUTLASS (SM100+) with emulation fallback; min SM 80. Sources: `.../quantization/mxfp4.py`, `.../kernels/linear/mxfp4/{flashinfer,emulation}.py`, llm-compressor `compression_schemes.md` (MXFP4 section).
- `deepseek_v4_fp8`: model-specific FP8 config. Source: `.../quantization/__init__.py`.

### 1.13 Quantized KV cache

`--kv-cache-dtype` accepted values (from `CacheDType` literal in `vllm/config/cache.py`): `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`, `fp8_ds_mla` (DeepSeek MLA), `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`, `int4_per_token_head`, `int8_per_token_head`, `fp8_per_token_head`, `nvfp4`.

- Scales: per-tensor (`q/k/v_scale=[1]`) or per-attention-head (`[num_heads]`). Per-head quantization only on the FlashAttention backend and requires llm-compressor calibration. Source: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html.
- Calibration: (1) no calibration, scales = 1.0 (`calculate_kv_scales=False`); (2) on-the-fly random-token warmup (`calculate_kv_scales=True`); (3) dataset calibration via llm-compressor (recommended). Source: quantized_kvcache doc page.
- `--kv-cache-dtype-skip-layers` skips sensitive layers (e.g. `sliding_window`). Source: quantized_kvcache doc page.
- Backend gates (source: `vllm/v1/attention/backends/`): FlashAttention → `auto/float16/bfloat16/fp8/fp8_e4m3`; FlashInfer → adds `fp8_e5m2`, `nvfp4`; Triton attention → adds `int4/int8/fp8_per_token_head`. With FA3, attention math itself runs in FP8 (queries also quantized). Source: quantized_kvcache doc page.
- `fp8` (E4M3) KV cache: CUDA 11.8+ and ROCm; `fp8_e5m2`: CUDA 11.8+. Source: quantized_kvcache doc page.
- FP8 KV memory math: ½ the KV bytes vs FP16 (no explicit size figure in doc — doc sells "store more tokens", ~2× KV capacity). Source: quantized_kvcache doc page (qualitative claim only).
- TurboQuant: 3/4-bit KV cache via Hadamard rotation + per-coordinate Lloyd-Max quantization of keys, uniform quantization of values. Source: `.../quantization/turboquant/__init__.py` + `turboquant_attn.py` backend.

---

## 2. Config flags & loading requirements

| Need | Mechanism | Pre-quantized required? |
|---|---|---|
| Pick format | `--quantization <name>` (registry §1.1) or auto-detect from checkpoint config | No — `online` + bnb in-flight quantize at load |
| FP8 KV cache | `--kv-cache-dtype fp8|fp8_e4m3|fp8_e5m2|nvfp4|int8_per_token_head|...` | No (runtime) |
| KV scale source | `--calculate-kv-scales [True|False]`, llm-compressor dataset calibration | No |
| Skip KV layers | `--kv-cache-dtype-skip-layers sliding_window\|0 1 23` | No |
| FP8 block scales | weights must be fp8-serialized checkpoint; `activation_scheme="dynamic"` required | Yes |
| NVFP4 kernel choice | `--linear-backend cutlass\|marlin\|flashinfer_*\|emulation\|...` (replaces `VLLM_NVFP4_GEMM_BACKEND`) | No (checkpoint already NVFP4) |
| Online quant config | `--quantization online --quantization-config '{...}'` | No |
| GGUF | `pip install vllm-gguf-plugin`; model `repo:quant_type` or local `.gguf`; `--tokenizer` base model | Yes (GGUF file) |
| AWQ/GPTQ/bnb/ModelOpt/Quark/TorchAO/INC | checkpoint config files (`quant_config.json`, `quantize_config.json`, `hf_quant_config.json`, `quantization_config`) auto-detected | Yes (except bnb/online in-flight) |

Sources: registry `__init__.py`, `vllm/config/cache.py`, quantized_kvcache + modelopt + online doc pages (all cited above).

---

## 3. Model / architecture constraints

- **Min SM per format** (§1 tables): GPTQ 60, INC 60, bnb 70, quark 70, moe_wna16 70, AWQ/FP8/torchao/humming/compressed-tensors-WNa16 75, mxfp4/modelopt/ModelOpt-MXFP8 80, Machete 90, FP8-W8A8 native 89+, NVFP4-W4A4 native 100+.
- **Marlin kernels**: group sizes [−1, 32, 64, 128]; FP4-Marlin (NVFP4 W4A16) group size 16 only and SM75+. Sources: `marlin_utils.py`, `marlin_utils_fp4.py`.
- **INT8 W8A8 has no Blackwell support** (SM ≥ 10) — use FP8. Source: `llm_compressor/int8_w8a8.md` warning.
- **FP8 W8A8 native compute** needs SM ≥ 8.9; below that, FP8 checkpoints run weight-only W8A16 via FP8 Marlin. Source: `llm_compressor/fp8.md`.
- **Per-head KV scales** only via FlashAttention backend + llm-compressor. Source: quantized_kvcache doc.
- **AWQ is 4-bit only**; **GPTQ is 4/8-bit**; **Marlin-24 removed** from main (§1.3).
- **GGUF model coverage is adapter-based, not universal** (§1.5). llama.cpp-class GGML quants all work via plugin Triton/CUDA kernels, but the architecture must map GGUF tensor names → HF model.
- **MoE**: AWQ/GPTQ experts run via Marlin-MoE or WNA16 kernels; NVFP4/MXFP4 MoE via `nvfp4_blockwise_moe_kernel.cu` (SM100) / FlashInfer / TRTLLM paths; `int8_per_channel_weight_only` and `nvfp4_per_token` online shorthands are MoE-only. Sources: `auto_awq.py`, `auto_gptq.py`, `vllm/config/quantization.py`, `csrc/libtorch_stable/quantization/fp4/`.
- **Pre-quantized checkpoint sources (HF)**: NeuralMagic FP8 collection `huggingface.co/collections/neuralmagic/fp8-llms-for-vllm-666742ed2b78b7ac8df13127`; INT4 `.../int4-llms-for-vllm-668ec34bf3c9fa45f857df2c`; INT8 `.../int8-llms-for-vllm-668ec32c049dca0369816415`; AWQ/GPTQ legacy `TheBloke/*`; GPTQModel `ModelCloud/*`; bnb `unsloth/*-bnb-4bit`; ModelOpt `nvidia/*-FP8`; GGUF `unsloth/*-GGUF`. Sources: per-format doc pages (cited above).

---

## 4. NVFP4 in vLLM — real status (source-verified)

**Format.** NVFP4 = NVIDIA 4-bit float (E2M1: 1 sign + 2 exponent + 1 mantissa). vLLM's layout: weights packed `uint8` (2 FP4 values/byte), per-block weight scales in **FP8-E4M3** with **group size 16**, plus scalar global scales for weights and activations. Exact statement in source: `vllm/model_executor/kernels/linear/nvfp4/base.py` (`NvFp4LinearLayerConfig` docstring: "packed uint8 weights (2 FP4 values per byte), FP8-E4M3 per-block weight scales (group size 16), and scalar global scales for both weights and activations"). Also llm-compressor `compression_schemes.md` (NVFP4 section: global per-tensor scale + per-group-16 local scales stored in `torch.float8_e4m3fn`; dynamic per-group-16 activation quantization; calibration dataset required for activation global scales).

**How it enters vLLM — there is no standalone `quantization="nvfp4"`.** NVFP4 checkpoints are consumed through four channels:
1. **compressed-tensors** (llm-compressor NVFP4 recipes) → `CompressedTensorsW4A4Fp4` scheme (`compressed_tensors_w4a4_nvfp4.py`; W4A4, or W4A16 via `use_a16=True`).
2. **ModelOpt** → `quantization="modelopt_fp4"`; `quant_algo NVFP4` → W4A4 CUTLASS path, `W4A16_NVFP4` → FP4 Marlin W4A16 (`modelopt.py`, `ModelOptNvFp4Config`).
3. **Quark** → `QuarkNVFP4` scheme (emulation-validated) (`quark/schemes/quark_nvfp4.py`).
4. **Online** → `nvfp4_per_token` shorthand, MoE-only, Blackwell + FlashInfer (`vllm/config/quantization.py`).

The "nvfp4/, nvfp4/base/, nvfp4/cutlass/ pages" in vLLM docs are **API autodoc pages for the kernel package** `vllm/model_executor/kernels/linear/nvfp4/` — not a format guide: `https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/` (subpages `base/`, `cutlass/`, `emulation/`, `fbgemm/`, `flashinfer/`, `humming/`, `marlin/`; plus MoE autodoc pages `nvfp4_moe/`, `nvfp4_emulation_moe/`, `trtllm_nvfp4_moe/`). Source: sitemap of docs.vllm.ai (2026-08-02).

**Kernel backends** (`kernels/linear/nvfp4/*.py` + `init_nvfp4_linear_kernel()` in `kernels/linear/__init__.py`):

| Kernel | Mode | Hardware gate |
|---|---|---|
| `FlashInferCuteDslNvFp4LinearKernel` | W4A4 | SM 10x + flashinfer (Blackwell) |
| `FlashInferCutlassNvFp4LinearKernel` | W4A4 | Blackwell |
| `CutlassNvFp4LinearKernel` (`cutlass_scaled_fp4_mm`) | W4A4 | SM 100–129, CUDA ≥ 12.8, compiled `ENABLE_NVFP4_SM100/SM120` (`csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu`) |
| `MarlinNvFp4LinearKernel` | W4A16 (weight-only) | CUDA SM75+; group size 16 only |
| `FlashInferTrtllmNvFp4LinearKernel`, `FlashInferCudnnNvFp4LinearKernel` | W4A4 | Blackwell + FlashInfer |
| `FbgemmNvFp4LinearKernel` | W4A4 | needs `fbgemm_gpu` |
| `EmulationNvFp4LinearKernel` | dequant→BF16 matmul | always (slow last-resort fallback) |
| `HummingNvFp4LinearKernel` | W4A4 | SM75+, CUDA |

CUDA auto-selection priority: FlashInferCuteDsl → FlashInferCutlass → Cutlass → Marlin → FlashInferTrtllm → FlashInferCudnn → FBGEMM → Emulation → Humming; ROCm: emulation only. `--linear-backend` overrides auto-selection. MoE NVFP4: `csrc/.../fp4/nvfp4_blockwise_moe_kernel.cu` (SM100), `compressed_tensors_moe_w4a4_nvfp4.py`, FlashInfer FP4 MoE utils, TRTLLM NVFP4 MoE.

**NVFP4 vs Marlin FP4.** NVFP4 W4A4 = both weights **and activations** in FP4 (requires Blackwell tensor cores). Marlin FP4 = weights FP4, activations stay FP16/BF16 (W4A16) — runs on SM75+ (Turing and later), explicitly the fallback for non-Blackwell GPUs ("Your GPU does not have native support for FP4 computation … weight-only FP4 compression … may degrade performance" — `kernels/linear/nvfp4/marlin.py`). So: *NVFP4 in vLLM is a Blackwell-native W4A4 format, with a Marlin W4A16 weight-only fallback for older hardware*. This is source-verified, not from the NVIDIA blog.

**Related**: `nvfp4` is also a KV-cache dtype (`CacheDType`), served by the FlashInfer attention backend (§1.13). NVFP4 is distinct from **MXFP4** (OCP MX spec: E8M0 scales, group 32, RTN without calibration, also Blackwell-native W4A4 in vLLM). Sources: llm-compressor `compression_schemes.md` (MXFP4 section), `mxfp4.py`.

**Format structure reference**: repo guide [nvfp4-quantization.md](./nvfp4-quantization.md).

---

## 5. Relevance to this repo

Rig context: consumer GPU with **8 GB VRAM, older NVIDIA (Turing SM75 / Ampere SM80 class)**; harness backend is llama.cpp (`llama-server`), not vLLM. See [inference-engines-landscape.md](./inference-engines-landscape.md).

- **Usable on the operator host (SM75/SM80)**: GPTQ + AWQ (Marlin W4A16), bitsandbytes (SM70+), INT8 W8A8 (SM75+), FP8 weight-only W8A16 via FP8 Marlin (SM75+), INT4 W4A16 compressed-tensors (SM80+), GGUF via plugin (GPU-only), quantized FP8 KV cache. These are the formats a vLLM serving path on this GPU could realistically use; AWQ/GPTQ/INT8 are the practical picks (real kernel speedups vs bnb's dequant path).
- **Not usable on the operator host**: FP8 W8A8 (SM89+), NVFP4 W4A4 (SM100+), MXFP4 native (Blackwell; only emulation on SM80), INT8 W4A8 CUTLASS (SM90), Machete (SM90). On Blackwell all of FP8/NVFP4/MXFP4 become first-class.
- **GGUF: vLLM vs llama.cpp.** This repo's models are GGUF, and llama.cpp executes them natively with per-quant CUDA kernels — vLLM would require the experimental OOT plugin, dequant-to-FP16/BF16 paths (Triton) that are slower and "under-optimized", and it is **GPU-only with no CPU fallback** — worse than llama.cpp on every axis for this workflow (single-user, local, offload-friendly). No reason to switch GGUF serving to vLLM; llama.cpp stays the Trial backend. vLLM GGUF only matters as a second server option if an OpenAI-API-compatible endpoint on the same GGUF files is ever wanted, and even then plugin maturity is the risk.
- **Blackwell future**: when this repo's hardware moves to Blackwell, the vLLM-relevant formats flip: FP8 W8A8, NVFP4 (W4A4 + `nvfp4` KV cache + FlashInfer attention), MXFP4 all become native and are the high-throughput serving path (vLLM or TensorRT-LLM for static graphs). Model-card docs for future Blackwell GGUFs won't change (llama.cpp still needs GGUF), but any future non-GGUF serving experiment should target NVFP4/FP8 via llm-compressor + compressed-tensors, not AWQ/GPTQ. Ecosystem signal (2026-07-10): Unsloth shipped Qwen3.6 **NVFP4 GGUFs** (Blackwell-only W4A4, ~2.5× faster, MTP tensors in-quant) — the llama.cpp side of the same NVFP4 wave; relevant to [`nvfp4-quantization.md`](./nvfp4-quantization.md).
- **Key maintenance notes for docs**: GGUF support is OOT since RFC #39583 (existing `vllm-quantization.md` matrix still lists GGUF as in-tree format — matrix row survives but loading requires the plugin); Marlin-24 kernels removed from main; W4A8 matrix row stale (CUTLASS SM90 exists); INT8 not on Blackwell.

---

## 6. Primary sources

**vLLM source (main @ 06018507, 2026-08-02)** — `https://github.com/vllm-project/vllm/blob/main/...`:
- `vllm/model_executor/layers/quantization/__init__.py` (registry, deprecated list, online shorthands)
- `vllm/model_executor/layers/quantization/auto_awq.py`, `auto_gptq.py`, `bitsandbytes.py`, `fp8.py`, `mxfp4.py`, `modelopt.py`, `torchao.py`, `humming.py`, `experts_int8.py`, `moe_wna16.py`, `kv_cache.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/` (schemes incl. `compressed_tensors_w4a4_nvfp4.py`, `w4a4_mxfp4.py`, `w4a8_int.py`, `w8a8_int8.py`, `wNa16.py`; MoE variants)
- `vllm/model_executor/layers/quantization/online/` (base, fp8, int8, nvfp4)
- `vllm/model_executor/layers/quantization/quark/` (`quark.py`, `schemes/quark_nvfp4.py`)
- `vllm/model_executor/layers/quantization/utils/` (`marlin_utils.py`, `marlin_utils_fp4.py`, `nvfp4_utils.py`, `nvfp4_emulation_utils.py`)
- `vllm/model_executor/kernels/linear/__init__.py` (MPLinear kernel selection, `init_nvfp4_linear_kernel`, `_POSSIBLE_NVFP4_KERNELS`), `vllm/model_executor/kernels/linear/nvfp4/*.py`, `.../mixed_precision/{conch,exllama,machete,cutlass,dynamic_4bit}.py`, `.../linear/mxfp4/*.py`
- `vllm/config/quantization.py` (`_ONLINE_SHORTHANDS`), `vllm/config/cache.py` (`CacheDType`)
- `vllm/v1/attention/backends/{flash_attn,flashinfer,triton_attn}.py` (KV cache dtype gates)
- `csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_entry.cu` (CUTLASS FP4 SM gate), `nvfp4_blockwise_moe_kernel.cu`, `mxfp4_blockwise_moe_kernel.cu`
- Tag `v0.9.0`: `vllm/model_executor/layers/quantization/gptq_marlin_24.py` (Marlin-24, removed from main)

**vLLM docs (in-tree `docs/features/quantization/` and docs.vllm.ai)**:
- https://docs.vllm.ai/en/latest/features/quantization/index.html (format list + hardware matrix + plugin API)
- `auto_awq.md`, `gptqmodel.md`, `bnb.md`, `gguf.md`, `inc.md`, `modelopt.md`, `online.md`, `quantized_kvcache.md`, `quark.md`, `torchao.md`, `fp8_vit_attn.md`
- `llm_compressor/{README,fp8,int4,int8_w4a8,int8_w8a8}.md`
- API autodoc NVFP4 pages: https://docs.vllm.ai/en/latest/api/vllm/model_executor/kernels/linear/nvfp4/ (+ `base/`, `cutlass/`, `emulation/`, `fbgemm/`, `flashinfer/`, `humming/`, `marlin/`)

**GGUF plugin**: https://github.com/vllm-project/vllm-gguf-plugin (README; `vllm_gguf_plugin/gguf_utils.py`; `triton/dequantize/`; `csrc/gguf/`); deprecation RFC https://github.com/vllm-project/vllm/issues/39583

**LLM Compressor**: https://github.com/vllm-project/llm-compressor/blob/main/docs/guides/compression_schemes.md; https://docs.vllm.ai/projects/llm-compressor/en/latest/; scheme list https://github.com/vllm-project/compressed-tensors/blob/main/src/compressed_tensors/quantization/quant_scheme.py

**Format specs**: AWQ paper https://arxiv.org/abs/2306.00978 (cited in `auto_awq.py`); OCP MX spec via llm-compressor MXFP4 section (E8M0 scales); NVFP4 layout per vLLM `nvfp4/base.py` docstring + llm-compressor NVFP4 section.

**Repo guides**: [vllm-quantization.md](./vllm-quantization.md) (overview matrix), [nvfp4-quantization.md](./nvfp4-quantization.md) (NVFP4 format structure), [inference-engines-landscape.md](./inference-engines-landscape.md).

---

## Open questions

None as of 2026-08-02. The three prior TBDs were resolved against primary sources (see the research file [`../sessions/2026-08-02-research-gap-closure.md`](../sessions/2026-08-02-research-gap-closure.md) §7):
- bitsandbytes OOT migration (RFC #39583): **still open**; bnb remains in-tree (GGUF already lives in the `vllm-gguf-plugin`). Revisit when the RFC closes or deprecation lands.
- W4A8 hardware matrix: **confirmed stale** — the docs row says Arm CPU only while source ships the CUTLASS W4A8 (SM90) kernel path; docs lag source.
- GGUF `IQ2_M`/`IQ3_M`: **file-type-only** acceptance (`is_valid_gguf_quant_type`); execution is tensor-type driven — no dedicated dequant module.
