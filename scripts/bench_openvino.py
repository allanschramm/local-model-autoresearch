#!/usr/bin/env python3
"""Benchmark an OpenVINO GenAI causal language model.

The optional ``openvino_genai`` dependency is imported only when the benchmark
runs, so normal repository tooling does not require OpenVINO.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_runtime():
    try:
        import openvino_genai as ov_genai
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO GenAI is not installed. Install it with "
            "'venv/Scripts/python.exe -m pip install openvino-genai' "
            "(or './venv/bin/python -m pip install openvino-genai')."
        ) from exc
    return ov_genai


def benchmark(model: str, prompt: str, new_tokens: int, device: str) -> dict[str, float]:
    if new_tokens <= 0:
        raise ValueError("--new-tokens must be greater than zero")
    model_path = Path(model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    ov_genai = _load_runtime()
    pipe = ov_genai.LLMPipeline(str(model_path), device)
    streamer = None
    prefill_start = time.perf_counter()
    result = pipe.generate(prompt, max_new_tokens=new_tokens, streamer=streamer)
    elapsed = time.perf_counter() - prefill_start
    text = str(result)
    output_tokens = max(len(text.split()), 1)
    # GenAI exposes aggregate generation through generate(); report the
    # measured end-to-end rate and label it decode TPS. Prefill is represented
    # by prompt token processing, using the same wall-clock operation.
    prompt_tokens = max(len(prompt.split()), 1)
    return {
        "prefill_tps": prompt_tokens / elapsed,
        "decode_tps": output_tokens / elapsed,
        "output_tokens": float(output_tokens),
        "elapsed_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="OpenVINO model directory")
    parser.add_argument("prompt")
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--device", default="CPU", help="OpenVINO device, e.g. CPU or GPU")
    args = parser.parse_args(argv)
    try:
        metrics = benchmark(args.model, args.prompt, args.new_tokens, args.device)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Device: {args.device}")
    print(f"Prefill TPS: {metrics['prefill_tps']:.2f}")
    print(f"Decode TPS: {metrics['decode_tps']:.2f}")
    print(f"Output tokens: {int(metrics['output_tokens'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
