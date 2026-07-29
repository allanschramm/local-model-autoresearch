#!/usr/bin/env python3
"""LCB-only smoke against Baseline in config.py (gambiarra for patching results.tsv).

Starts llama-server from current ENGINE_/SAMPLER_ Baseline, runs LiveCodeBench
(task_limit=10), prints JSON: {model, lcb, tokens, seconds, peak_vram_gb}.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\lcb_only.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.benchmarks.benchmark_coding import LiveCodeBenchTask, run_coding_eval
from autoresearch.core.config import load_config
from autoresearch.core.llama_client import GenerationParams, LlamaClient
from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent


def main() -> int:
    cfg = load_config()
    models_dir = REPO_ROOT / "models"
    intent, _ = ServerIntent.from_config(cfg, models_dir)
    gen = GenerationParams(
        temp=float(cfg.get("TEMP", 0.4)),
        top_p=cfg.get("TOP_P"),
        top_k=cfg.get("TOP_K"),
        min_p=cfg.get("MIN_P"),
        repeat_penalty=cfg.get("REPEAT_PENALTY"),
        presence_penalty=cfg.get("PRESENCE_PENALTY"),
        frequency_penalty=cfg.get("FREQUENCY_PENALTY"),
    )
    vram_limit = cfg.get("VRAM_LIMIT_MB")
    with LlamaServerRunner(intent, vram_limit_mb=vram_limit) as runner:
        if runner.port is None:
            print("ERROR: server failed to start", file=sys.stderr)
            return 1
        client = LlamaClient(runner.port)
        pass1, tokens, seconds = run_coding_eval(
            client, LiveCodeBenchTask(), gen_params=gen, task_limit=10
        )
        out = {
            "model": Path(str(cfg.get("MODEL", ""))).name,
            "lcb": round(float(pass1), 4),
            "tokens": int(tokens),
            "seconds": round(float(seconds), 1),
            "peak_vram_gb": round(float(runner.peak_vram_mb) / 1024.0, 1),
            "vram_killed": bool(runner.vram_killed),
        }
        print(json.dumps(out), flush=True)
        return 1 if runner.vram_killed else 0


if __name__ == "__main__":
    raise SystemExit(main())
