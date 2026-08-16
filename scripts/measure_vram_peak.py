"""Measure real peak VRAM for the Baseline model, bypassing the preflight gate.

Calibration for issue #10 / issue #2 underestimate case: does the estimator
(weights + KV + overhead) match real committed VRAM? Uses the harness
LlamaServerRunner (same lifecycle, flags, and NVML/nvidia-smi peak sampler as
Trials) but skips the preflight gate on purpose. A real generation pushes
allocation past load. Not a Trial — no results.tsv row written.

Usage:
    .\\venv\\Scripts\\python.exe scripts/measure_vram_peak.py [--n-gen N]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from autoresearch.core import config as core_config
from autoresearch.core.llama_client import GenerationParams, LlamaClient
from autoresearch.core.llama_runner import LlamaServerRunner, ServerIntent
from autoresearch.core.model_arch import resolve_model_file


def nvidia_used_free() -> tuple[float, float]:
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        parts = [p.strip() for p in (res.stdout or "").split(",")]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return 0.0, 8192.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=256, help="generated tokens for the probe")
    args = ap.parse_args()

    models_dir = BASE_DIR / "models"
    cfg = dict(core_config.DEFAULTS)
    intent, norm = ServerIntent.from_config(cfg, models_dir)
    resolved = resolve_model_file(cfg["MODEL"], models_dir)
    model_path = resolved or (models_dir / cfg["MODEL"])
    print(
        f"model={model_path.name} ctx={intent.ctx_size} kv={intent.kv_cache} n-cpu-moe={intent.n_cpu_moe}"
    )

    used0, free0 = nvidia_used_free()
    print(f"[pre]         used={used0:.0f}MB free={free0:.0f}MB")

    runner = LlamaServerRunner(intent, vram_limit_mb=norm.get("vram_limit_mb"))
    with runner as r:
        used1, free1 = nvidia_used_free()
        print(f"[post-load]   used={used1:.0f}MB free={free1:.0f}MB (delta +{used1 - used0:.0f}MB)")
        client = LlamaClient(r.port)
        t0 = time.time()
        resp = client.complete(
            "Write a detailed explanation of quantum computing with a Python simulation example.",
            gen=GenerationParams(temp=0.2, top_k=80, repeat_penalty=1.05),
            max_tokens=args.n_gen,
        )
        dt = time.time() - t0
        used2, free2 = nvidia_used_free()
        toks = resp["usage"]["total_tokens"]
        print(
            f"[post-gen]    used={used2:.0f}MB free={free2:.0f}MB toks={toks} tps={toks / dt:.1f} ({dt:.1f}s)"
        )
        time.sleep(0.4)  # let the sampler catch the last peak
        print(f"[peak]        runner.peak_vram_mb={r.peak_vram_mb:.0f}MB (sampler)")

    used3, free3 = nvidia_used_free()
    print(f"[post-exit]   used={used3:.0f}MB free={free3:.0f}MB")
    limit = norm.get("vram_limit_mb") or 7900.0
    print(f"[note] configured VRAM_LIMIT_MB={limit:.0f}MB; real peak is ground truth vs estimator")


if __name__ == "__main__":
    main()
