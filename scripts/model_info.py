#!/usr/bin/env python3
"""Read-only GGUF metadata query through the harness.

Prints the GGUF ground truth an agent needs to seed a model card or Baseline —
arch class (MoE vs dense), block count, KV cache sizing, resolved path, and the
effective `--n-cpu-moe` for Baseline `N_CPU_MOE=None` — without loading or
serving the model.

This is the ONLY blessed way for an agent to learn model details. Agents must
NOT parse the raw `.gguf` (or scrape tensor dumps) — run this script instead.

Usage:
    ./venv/Scripts/python.exe scripts/model_info.py <basename|path> [--ctx N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch.core.model_arch import (
    gguf_block_count,
    gguf_kv_f16_mb,
    is_moe_model,
    resolve_model_file,
    resolve_n_cpu_moe,
)

CTX_DEFAULT = 65536


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only GGUF metadata query (harness-backed).")
    ap.add_argument(
        "model",
        help="Model basename (e.g. Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf) or path",
    )
    ap.add_argument(
        "--ctx",
        type=int,
        default=CTX_DEFAULT,
        help="Context size for KV cache sizing (default %(default)s)",
    )
    ap.add_argument(
        "--tensors",
        action="store_true",
        help="Dump the full tensor inventory (count + names) read-only. "
        "For GGUF forensics (e.g. embedded-MTP nextn tensors).",
    )
    args = ap.parse_args()

    path = resolve_model_file(args.model)
    if path is None:
        print(f"ERROR: model not found under models/: {args.model}")
        return 2

    print(f"ref         = {args.model}")
    print(f"resolved    = {path}")
    if path.is_file():
        print(f"size_mib    = {path.stat().st_size / (1024 * 1024):.1f}")
    else:
        print("size_mib    = (not a regular file)")
        return 3

    print(f"arch        = {'MoE' if is_moe_model(args.model) else 'dense'}")

    try:
        print(f"block_count = {gguf_block_count(path)}")
    except Exception as exc:  # noqa: BLE001 — report, don't crash the query
        print(f"block_count = unreadable ({exc})")

    try:
        kv = gguf_kv_f16_mb(path, args.ctx)
        print(f"kv_f16_mb@{args.ctx} = {kv if kv is None else round(kv, 2)}")
    except Exception as exc:  # noqa: BLE001
        print(f"kv_f16_mb@{args.ctx} = unreadable ({exc})")

    try:
        n_cpu, auto = resolve_n_cpu_moe(path, None)
        print(f"n_cpu_moe(None) = {n_cpu} (auto={auto})")
    except Exception as exc:  # noqa: BLE001
        print(f"n_cpu_moe(None) = unreadable ({exc})")

    if args.tensors:
        try:
            from gguf import GGUFReader

            reader = GGUFReader(str(path))
            names = [t.name for t in reader.tensors]
            print(f"tensor_count = {len(names)}")
            for name in sorted(names):
                print(f"  {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"tensors = unreadable ({exc})")
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
