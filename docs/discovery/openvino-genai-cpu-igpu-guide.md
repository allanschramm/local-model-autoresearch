# OpenVINO GenAI CPU and iGPU Inference Guide

OpenVINO GenAI is a practical path for Intel CPU and integrated-GPU inference. It uses OpenVINO IR models rather than GGUF files, so export and benchmark each model/device combination separately.

## 1. Install the Runtime

Create or reuse the repository virtual environment:

```bash
# Windows
venv/Scripts/python.exe -m pip install optimum[openvino] openvino-genai
# Linux/macOS
./venv/bin/python -m pip install 'optimum[openvino]' openvino-genai
```

The CPU plugin is broadly available. GPU/iGPU execution requires a supported Intel GPU, matching graphics drivers, and an OpenVINO build exposing the GPU plugin. Use `CPU` when `GPU` is unavailable; do not assume that a discrete non-Intel GPU is supported by this path.

## 2. Export a Model

Export a Hugging Face-compatible causal language model to OpenVINO IR with `optimum-cli`:

```bash
optimum-cli export openvino --model <model-id> --task text-generation-with-past ov-model
```

For lower memory use, export weight compression where supported:

```bash
optimum-cli export openvino --model <model-id> --task text-generation-with-past \
  --weight-format int8 ov-model-int8
```

INT8 is a good CPU starting point. INT4 can reduce memory further, but may reduce quality and is model/transformer-version dependent. Validate quality before selecting it for production. Keep the exported directory, tokenizer, and configuration together.

## 3. Select a Device

OpenVINO device names are normally `CPU` and `GPU`:

```bash
# CPU
venv/Scripts/python.exe scripts/bench_openvino.py ov-model "Write a haiku" --device CPU

# Intel integrated GPU, if the GPU plugin and driver are installed
venv/Scripts/python.exe scripts/bench_openvino.py ov-model "Write a haiku" --device GPU
```

On Linux, verify the Intel graphics stack and permissions before diagnosing model performance. On Windows, install a current Intel graphics driver. Unsupported hardware, missing drivers, or a model unsupported by the selected plugin must fail clearly rather than silently falling back.

## 4. Reproducible Benchmark Workflow

Run from the repository root. The benchmark reports prefill and decode throughput separately:

```bash
venv/Scripts/python.exe scripts/bench_openvino.py ov-model "Explain photosynthesis in two sentences" \
  --new-tokens 128 --device CPU
```

Repeat each configuration after a warm-up, use the same prompt and token limit, and record device, precision, model revision, and runtime versions. Compare CPU and iGPU only with identical exported weights and generation settings. Prefill TPS reflects prompt processing; decode TPS reflects generated-token throughput. Do not compare these directly with llama.cpp GGUF results without documenting the different model format and tokenization path.

If `openvino_genai` is absent, the script exits nonzero with an installation command. Missing model directories and invalid token limits also exit nonzero.

## 5. Practical Tuning

- Start with INT8 on CPU; test INT4 when memory pressure matters more than quality.
- Keep prompts and generated-token limits fixed for comparisons.
- Benchmark short and long prompts because prefill scales with prompt length while decode is primarily generation-bound.
- Use `CPU` as the reliable baseline, then test `GPU` on supported Intel iGPUs.
- Leave host-memory headroom for the operating system and application; a smaller quantization is safer than paging.
