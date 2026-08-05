#!/usr/bin/env python3
"""Benchmark an OpenVINO GenAI causal language model.

The optional ``openvino_genai`` dependency is imported only when the benchmark
runs, so normal repository tooling does not require OpenVINO.

Prefill TPS is measured from a single-token generation (wall-clock dominated
by prompt processing); decode TPS derives from subtracting the measured
prefill time out of a full generation run. Token counts come from the model
tokenizer when available, with a word-count fallback.
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


def _get_tokenizer(pipe):
    try:
        return pipe.get_tokenizer()
    except Exception:
        return None


def _token_count(text: str, tokenizer) -> int:
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text)
            return max(len(list(getattr(ids, "input_ids", ids))), 1)
        except Exception:
            pass
    return max(len(text.split()), 1)


def benchmark(model: str, prompt: str, new_tokens: int, device: str) -> dict[str, float]:
    if new_tokens <= 0:
        raise ValueError("--new-tokens must be greater than zero")
    model_path = Path(model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    ov_genai = _load_runtime()
    pipe = ov_genai.LLMPipeline(str(model_path), device)
    tokenizer = _get_tokenizer(pipe)
    prompt_tokens = _token_count(prompt, tokenizer)

    pipe.generate(prompt, max_new_tokens=1)
    prefill_start = time.perf_counter()
    pipe.generate(prompt, max_new_tokens=1)
    prefill_elapsed = time.perf_counter() - prefill_start
    prefill_tps = prompt_tokens / prefill_elapsed

    total_start = time.perf_counter()
    result = pipe.generate(prompt, max_new_tokens=new_tokens)
    total_elapsed = time.perf_counter() - total_start
    output_tokens = _token_count(str(result), tokenizer)
    decode_elapsed = max(total_elapsed - prefill_elapsed, 1e-9)
    decode_tps = output_tokens / decode_elapsed

    return {
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "output_tokens": float(output_tokens),
        "elapsed_seconds": total_elapsed,
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
