#!/usr/bin/env python3
"""Rank models from results.tsv (Pareto / Day / Night / claw / coding).

Ground truth stays TSV. This CLI is the agent-facing query surface so ranking
never needs ad-hoc temp scripts.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode claw
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode coding
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode agentic-coding
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --day-tps-floor 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ensure_repo_root_on_sys_path() -> None:
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_root_on_sys_path()

from autoresearch.core.classify import fp_from_config_json
from autoresearch.core.pareto import pareto_set

DEFAULT_TSV = REPO_ROOT / "results.tsv"

_DESC_TPS_RE = re.compile(r"(?:bench_tg|TPS)=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_DESC_CTX_RE = re.compile(r"\bctx=([0-9]+)\b", re.IGNORECASE)

DEFAULT_DAY_TPS_FLOOR = 50.0
DEFAULT_NIGHT_CTX_FLOOR = 65536

OK_OUTCOMES = {"", "OK"}


@dataclass(frozen=True)
class Point:
    model: str
    ctx: int
    tps: float
    agentic: float
    coding: float
    fp: str | None = None
    agentic_coding: float | None = None

    @property
    def iq_min(self) -> float:
        return min(self.agentic, self.coding)

    @property
    def night_iq(self) -> float:
        if self.agentic_coding is None:
            return self.iq_min
        return min(self.agentic, self.coding, self.agentic_coding)

    @property
    def complete(self) -> bool:
        return self.agentic >= 0.0 and self.coding >= 0.0


def _fnum(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _valid_score(raw: Any) -> float | None:
    value = _fnum(raw)
    if value is None or value < 0.0 or value > 1.0:
        return None
    return value


def _ctx_from_description(description: str) -> int | None:
    match = _DESC_CTX_RE.search(description or "")
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _tps_from_description(description: str) -> float | None:
    # Prefer bench_tg= over TPS= when both appear.
    matches = _DESC_TPS_RE.findall(description or "")
    if not matches:
        return None
    # Last numeric wins when both TPS= and bench_tg= present (bench_tg usually last).
    value = float(matches[-1])
    return value if value > 0 else None


def _ctx_of(row: dict[str, str]) -> int | None:
    direct = _fnum(row.get("ctx"))
    if direct is not None and direct > 0:
        return int(direct)
    raw = (row.get("config_json") or "").strip()
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            for key in ("CTX_SIZE", "ctx_size", "ctx"):
                value = _fnum(data.get(key))
                if value is not None and value > 0:
                    return int(value)
    return _ctx_from_description(row.get("description") or "")


def _tps_of(row: dict[str, str]) -> float | None:
    for key in ("bench_tg", "tps"):
        value = _fnum(row.get(key))
        if value is not None and value > 0:
            return value
    return _tps_from_description(row.get("description") or "")


def _is_measurement_row(row: dict[str, str]) -> bool:
    outcome = (row.get("outcome") or "").strip()
    return outcome in OK_OUTCOMES


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _axis_values(row: dict[str, str]) -> tuple[float | None, float | None]:
    """agentic/coding axis values of one row: columns when populated, else category.

    The modern write path records a combined vector (agentic-full + coding-10
    measured in one run) with both columns populated; the split form stores
    agentic-full / 10-task category rows. Legacy rows have neither.
    """
    agentic = _valid_score(row.get("agentic"))
    coding = _valid_score(row.get("coding"))
    agentic_coding = _valid_score(row.get("agentic_coding"))
    score = _valid_score(row.get("val_score"))
    category = (row.get("category") or "").strip()
    if agentic is None and category == "agentic-full":
        agentic = score
    if coding is None and category == "10-task":
        coding = score
    if agentic_coding is None and category == "agentic-coding":
        agentic_coding = score
    return agentic, coding, agentic_coding


def build_vectors(
    rows: Sequence[dict[str, str]],
) -> tuple[list[Point], list[Point]]:
    """Merge best valid agentic + coding + TPS + ctx per GGUF basename.

    Point identity is the model basename (quant/version file). Engine flags and
    sampler changes hill-climb the same point: max claw, max coding, max TPS,
    max ctx across all OK Trials for that file (ADR 0012). Fingerprint is kept
    only as a pick hint (config_json of the best-claw row) for Baseline load.
    Uses outcome=OK (or empty); status cells are not measurement validity.
    """
    ag_best: dict[str, dict[str, Any]] = {}
    cod_best: dict[str, dict[str, Any]] = {}
    ac_best: dict[str, float] = {}
    tps_best: dict[str, float] = {}
    ctx_best: dict[str, int] = {}

    for row in rows:
        model = (row.get("model") or "").strip()
        if not model:
            continue
        fp = fp_from_config_json(row.get("config_json"))
        tps = _tps_of(row)
        if tps is not None:
            prev = tps_best.get(model)
            if prev is None or tps > prev:
                tps_best[model] = tps
        ctx = _ctx_of(row)
        if ctx is not None and ctx > 0:
            prev_c = ctx_best.get(model)
            if prev_c is None or ctx > prev_c:
                ctx_best[model] = ctx

        if not _is_measurement_row(row):
            continue
        agentic, coding, agentic_coding = _axis_values(row)
        if agentic is None and coding is None and agentic_coding is None:
            continue
        if agentic is not None:
            prev = ag_best.get(model)
            if prev is None or agentic > prev["agentic"]:
                ag_best[model] = {"agentic": agentic, "fp": fp}
        if coding is not None:
            prev = cod_best.get(model)
            if prev is None or coding > prev["coding"]:
                cod_best[model] = {"coding": coding, "fp": fp}
        if agentic_coding is not None:
            prev_ac = ac_best.get(model)
            if prev_ac is None or agentic_coding > prev_ac:
                ac_best[model] = agentic_coding

    complete: list[Point] = []
    incomplete: list[Point] = []
    for model in sorted(set(ag_best) | set(cod_best) | set(ac_best)):
        ag = ag_best.get(model)
        cod = cod_best.get(model)
        agentic = float(ag["agentic"]) if ag else -1.0
        coding = float(cod["coding"]) if cod else -1.0
        ac = ac_best.get(model)
        tps = float(tps_best.get(model, 0.0))
        ctx = int(ctx_best.get(model, 0))
        fp = None
        if ag and ag.get("fp"):
            fp = ag["fp"]
        elif cod and cod.get("fp"):
            fp = cod["fp"]
        point = Point(
            model=model,
            ctx=ctx,
            tps=tps,
            agentic=max(agentic, 0.0) if not ag else agentic,
            coding=max(coding, 0.0) if not cod else coding,
            fp=fp,
            agentic_coding=ac,
        )
        if ag and cod:
            complete.append(point)
        else:
            incomplete.append(point)
    return complete, incomplete


# Pareto Set owned by the nucleus (issue #1); alias keeps callers/tests stable.
# Input order preserved — Day/Night tables sort internally (ADR 0008).
pareto_front = pareto_set


def pick_day(
    front: Sequence[Point],
    day_tps_floor: float = DEFAULT_DAY_TPS_FLOOR,
) -> Point | None:
    ranked = day_table(front, day_tps_floor=day_tps_floor)
    return ranked[0] if ranked else None


def pick_night(
    front: Sequence[Point],
    night_ctx_floor: int = DEFAULT_NIGHT_CTX_FLOOR,
) -> Point | None:
    ranked = night_table(front, night_ctx_floor=night_ctx_floor)
    return ranked[0] if ranked else None


def day_table(
    front: Sequence[Point],
    day_tps_floor: float = DEFAULT_DAY_TPS_FLOOR,
) -> list[Point]:
    """Front points clearing Day TPS floor, sorted by maximin IQ then TPS (ADR 0009)."""
    if not front:
        return []
    eligible = [p for p in front if p.tps >= day_tps_floor]
    pool = eligible if eligible else list(front)
    if eligible:
        return sorted(pool, key=lambda p: (-p.iq_min, -p.tps, -p.ctx, p.model))
    return sorted(pool, key=lambda p: (-p.tps, -p.iq_min, -p.ctx, p.model))


def night_table(
    front: Sequence[Point],
    night_ctx_floor: int = DEFAULT_NIGHT_CTX_FLOOR,
) -> list[Point]:
    """Front points clearing Night ctx floor, sorted by maximin IQ (ADR 0013)."""
    if not front:
        return []
    eligible = [p for p in front if p.ctx >= night_ctx_floor]
    measured = [p for p in eligible if p.agentic_coding is not None]
    if measured:
        return sorted(measured, key=lambda p: (-p.night_iq, -p.ctx, -p.tps, p.model))
    if eligible:
        return sorted(eligible, key=lambda p: (-p.iq_min, -p.ctx, -p.tps, p.model))
    return sorted(front, key=lambda p: (-p.ctx, -p.iq_min, -p.tps, p.model))


def _rank_axis(
    rows: Sequence[dict[str, str]],
    category: str,
    *,
    score_column: str | None = None,
) -> list[tuple[str, float, float | None, int | None]]:
    best: dict[str, tuple[float, float | None, int | None]] = {}
    for row in rows:
        if not _is_measurement_row(row):
            continue
        model = (row.get("model") or "").strip()
        score = _valid_score(row.get(score_column)) if score_column else None
        if score is None and (row.get("category") or "").strip() == category:
            score = _valid_score(row.get("val_score"))
        if not model or score is None:
            continue
        prev = best.get(model)
        if prev is None or score > prev[0]:
            best[model] = (score, _tps_of(row), _ctx_of(row))
    ranked = [(model, score, tps, ctx) for model, (score, tps, ctx) in best.items()]
    ranked.sort(key=lambda item: (-item[1], -(item[2] or 0.0), item[0]))
    return ranked


def _fmt_ctx(ctx: int) -> str:
    if ctx <= 0:
        return "-"
    # Repo leaderboards use 65k/32k/131k (= ctx // 1000), not KiB.
    if ctx >= 1000:
        return f"{ctx // 1000}k"
    return str(ctx)


def _md_table(points: Sequence[Point]) -> list[str]:
    if not points:
        return ["(none)"]
    headers = ("#", "Model", "ctx", "TPS", "agentic", "coding")
    rows: list[tuple[str, ...]] = []
    for i, p in enumerate(points, 1):
        rows.append(
            (
                str(i),
                p.model,
                _fmt_ctx(p.ctx),
                f"{p.tps:.1f}",
                f"{p.agentic:.4f}",
                f"{p.coding:.4f}",
            )
        )
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            # left-align model; right-align numeric-ish cols
            if i == 1:
                parts.append(f" {cell:<{widths[i]}} ")
            else:
                parts.append(f" {cell:>{widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(row) for row in rows)
    return out


def _fmt_axis_row(
    rank: int,
    model: str,
    score: float,
    tps: float | None,
    ctx: int | None,
) -> tuple[str, str, str, str, str]:
    tps_s = f"{tps:.1f}" if tps is not None else "-"
    ctx_s = _fmt_ctx(ctx) if ctx is not None else "-"
    return (str(rank), model, ctx_s, tps_s, f"{score:.4f}")


def _md_axis_table(
    ranked: Sequence[tuple[str, float, float | None, int | None]],
) -> list[str]:
    if not ranked:
        return ["(none)"]
    headers = ("#", "Model", "ctx", "TPS", "score")
    rows = [_fmt_axis_row(i, m, s, t, c) for i, (m, s, t, c) in enumerate(ranked, 1)]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: Sequence[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 1:
                parts.append(f" {cell:<{widths[i]}} ")
            else:
                parts.append(f" {cell:>{widths[i]}} ")
        return "|" + "|".join(parts) + "|"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [fmt_row(headers), sep]
    out.extend(fmt_row(row) for row in rows)
    return out


def format_report(
    complete: Sequence[Point],
    incomplete: Sequence[Point],
    *,
    day_tps_floor: float = DEFAULT_DAY_TPS_FLOOR,
    night_ctx_floor: int = DEFAULT_NIGHT_CTX_FLOOR,
    mode: str,
    rows: Sequence[dict[str, str]] | None = None,
) -> str:
    del incomplete  # kept in signature for callers; default view is Day/Night only
    lines: list[str] = []
    front = pareto_front(complete)

    if mode in ("pareto", "day", "all"):
        day_rows = day_table(front, day_tps_floor=day_tps_floor)
        lines.append(f"DAY  (TPS >= {day_tps_floor:.1f})  pick=#1")
        lines.extend(_md_table(day_rows))

    if mode in ("pareto", "night", "all"):
        if lines:
            lines.append("")
        night_rows = night_table(front, night_ctx_floor=night_ctx_floor)
        lines.append(f"NIGHT  (CTX >= {night_ctx_floor})  pick=#1")
        lines.extend(_md_table(night_rows))

    if mode in ("claw", "all") and rows is not None:
        if lines:
            lines.append("")
        lines.append("CLAW-FULL")
        lines.extend(_md_axis_table(_rank_axis(rows, "agentic-full")))

    if mode in ("coding", "all") and rows is not None:
        if lines:
            lines.append("")
        lines.append("CODING-10")
        lines.extend(_md_axis_table(_rank_axis(rows, "10-task")))

    if mode in ("agentic-coding", "all") and rows is not None:
        if lines:
            lines.append("")
        lines.append("AGENTIC-CODING")
        lines.extend(
            _md_axis_table(_rank_axis(rows, "agentic-coding", score_column="agentic_coding"))
        )

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank models from results.tsv (ADR 0006/0009).",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_TSV,
        help="Path to results.tsv (default: repo root)",
    )
    parser.add_argument(
        "--mode",
        choices=("pareto", "day", "night", "claw", "coding", "agentic-coding", "all"),
        default="pareto",
        help="Report view (default: pareto = Day + Night markdown tables)",
    )
    parser.add_argument(
        "--day-tps-floor",
        type=float,
        default=DEFAULT_DAY_TPS_FLOOR,
        help=f"ADR 0009 Day TPS floor (default {DEFAULT_DAY_TPS_FLOOR})",
    )
    parser.add_argument(
        "--night-ctx-floor",
        type=int,
        default=DEFAULT_NIGHT_CTX_FLOOR,
        help=f"Night CTX floor (default {DEFAULT_NIGHT_CTX_FLOOR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.tsv
    if not path.is_file():
        print(f"ERROR: missing {path}", file=sys.stderr)
        return 1
    rows = load_rows(path)
    complete, incomplete = build_vectors(rows)
    report = format_report(
        complete,
        incomplete,
        day_tps_floor=args.day_tps_floor,
        night_ctx_floor=args.night_ctx_floor,
        mode=args.mode,
        rows=rows,
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
