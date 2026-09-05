#!/usr/bin/env python3
"""Apply a Fingerprint file to the mutable Baseline (issue #50, ADR 0014 phase 1).

Copies the file's engine (+ optional sampler) into config.py Baseline via
write_baseline, so existing Claw / SWE-lite / coding-10 runs use that engine.
An omitted sampler leaves the Baseline sampler alone. No eval harness
rewrite: benches keep reading Baseline. No GPU required.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\apply_fingerprint.py fingerprints\\model.json
    .\\venv\\Scripts\\python.exe scripts\\apply_fingerprint.py --model model.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch.core import fingerprint
from autoresearch.core.fingerprint import FingerprintError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy a Fingerprint file into the mutable Baseline.",
    )
    parser.add_argument(
        "fingerprint",
        nargs="?",
        help="path to the Fingerprint JSON file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="GGUF basename; resolves fingerprints/<stem>.json",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline config.py to update (default: autoresearch/core/config.py)",
    )
    args = parser.parse_args(argv)

    if args.model:
        target = fingerprint.path_for(args.model)
    elif args.fingerprint:
        target = Path(args.fingerprint)
    else:
        parser.error("give a Fingerprint file or --model <basename>")
    if not target.exists():
        print(f"fingerprint not found: {target}", file=sys.stderr)
        return 1

    try:
        data = fingerprint.load(target)
        result = fingerprint.apply(target, baseline_path=args.baseline)
    except FingerprintError as exc:
        print(f"invalid Fingerprint: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # ConfigError + OSError: actionable, not a traceback
        print(f"cannot apply {target}: {exc}", file=sys.stderr)
        return 1

    sampler_note = "left alone"
    if data.get("sampler") is not None:
        sampler_note = f"applied ({len(data['sampler'])} keys)"
    print(f"applied {target} -> Baseline MODEL={result['MODEL']}")
    print(f"engine keys: {len(data['engine'])}; sampler: {sampler_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
