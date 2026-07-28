#!/usr/bin/env python3
"""Rank models from results.tsv (Pareto / Day / Night / claw / coding).

Ground truth stays TSV. This CLI is the agent-facing query surface so ranking
never needs ad-hoc temp scripts.

Usage (repo root):
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode claw
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --mode coding
    .\\venv\\Scripts\\python.exe scripts\\rank_results.py --day-iq-ratio 0.8
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_DESC_TPS_RE = re.compile(r"(?:bench_tg|TPS)=([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_DESC_CTX_RE = re.compile(r"\bctx=([0-9]+)\b", re.IGNORECASE)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TSV = REPO_ROOT / "results.tsv"

DEFAULT_DAY_IQ_RATIO = 0.75
DEFAULT_NIGHT_CTX_FLOOR = 65536

OK_OUTCOMES = {"", "OK"}


@dataclass(frozen=True)
class Point:
    model: str
    ctx: int
    tps: float
    agentic: float
    coding: float

    @property
    def iq_min(self) -> float:
        return min(self.agentic, self.coding)

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


def build_vectors(
    rows: Sequence[dict[str, str]],
) -> tuple[list[Point], list[Point]]:
    """Merge best valid agentic-full + coding-10 per model basename.

    Uses outcome=OK (or empty) and val_score in [0, 1]. Ignores legacy
    keep/discard — that flag is Search history, not measurement validity.
    """
    ag_best: dict[str, dict[str, Any]] = {}
    cod_best: dict[str, dict[str, Any]] = {}
    tps_fallback: dict[str, float] = {}

    for row in rows:
        model = (row.get("model") or "").strip()
        if not model:
            continue
        tps = _tps_of(row)
        if tps is not None:
            prev = tps_fallback.get(model)
            if prev is None or tps > prev:
                tps_fallback[model] = tps

        if not _is_measurement_row(row):
            continue
        score = _valid_score(row.get("val_score"))
        if score is None:
            continue
        category = (row.get("category") or "").strip()
        ctx = _ctx_of(row)
        payload = {"score": score, "tps": tps, "ctx": ctx}

        if category == "agentic-full":
            prev = ag_best.get(model)
            if prev is None or score > prev["score"]:
                ag_best[model] = payload
        elif category == "10-task":
            prev = cod_best.get(model)
            if prev is None or score > prev["score"]:
                cod_best[model] = payload

    complete: list[Point] = []
    incomplete: list[Point] = []
    for model in sorted(set(ag_best) | set(cod_best)):
        ag = ag_best.get(model)
        cod = cod_best.get(model)
        agentic = float(ag["score"]) if ag else -1.0
        coding = float(cod["score"]) if cod else -1.0
        tps = 0.0
        if ag and ag["tps"] is not None:
            tps = float(ag["tps"])
        elif cod and cod["tps"] is not None:
            tps = float(cod["tps"])
        else:
            tps = float(tps_fallback.get(model, 0.0))
        ctx_candidates = [c for c in (
            ag["ctx"] if ag else None,
            cod["ctx"] if cod else None,
        ) if c]
        ctx = max(ctx_candidates) if ctx_candidates else 0
        point = Point(model=model, ctx=ctx, tps=tps, agentic=max(agentic, 0.0), coding=max(coding, 0.0))
        if ag and cod:
            complete.append(Point(model=model, ctx=ctx, tps=tps, agentic=agentic, coding=coding))
        else:
            incomplete.append(point)
    return complete, incomplete


def dominates(a: Point, b: Point) -> bool:
    """A dominates B on maximize axes ctx, TPS, agentic, coding."""
    ge = (
        a.ctx >= b.ctx
        and a.tps >= b.tps
        and a.agentic >= b.agentic
        and a.coding >= b.coding
    )
    gt = (
        a.ctx > b.ctx
        or a.tps > b.tps
        or a.agentic > b.agentic
        or a.coding > b.coding
    )
    return ge and gt


def pareto_front(points: Sequence[Point]) -> list[Point]:
    front: list[Point] = []
    for candidate in points:
        if any(dominates(other, candidate) for other in points if other is not candidate):
            continue
        front.append(candidate)
    front.sort(key=lambda p: (-p.iq_min, -p.tps, -p.ctx, p.model))
    return front


def pick_day(front: Sequence[Point], day_iq_ratio: float = DEFAULT_DAY_IQ_RATIO) -> Point | None:
    ranked = day_table(front, day_iq_ratio=day_iq_ratio)
    return ranked[0] if ranked else None


def pick_night(
    front: Sequence[Point],
    night_ctx_floor: int = DEFAULT_NIGHT_CTX_FLOOR,
) -> Point | None:
    ranked = night_table(front, night_ctx_floor=night_ctx_floor)
    return ranked[0] if ranked else None


def day_table(
    front: Sequence[Point],
    day_iq_ratio: float = DEFAULT_DAY_IQ_RATIO,
) -> list[Point]:
    """Front points in Day IQ band, sorted by TPS desc (ADR 0008)."""
    if not front:
        return []
    iq_best = max(p.iq_min for p in front)
    floor = day_iq_ratio * iq_best
    band = [p for p in front if p.iq_min >= floor]
    pool = band if band else list(front)
    if band:
        return sorted(pool, key=lambda p: (-p.tps, -p.iq_min, -p.ctx, p.model))
    return sorted(pool, key=lambda p: (-p.iq_min, -p.tps, -p.ctx, p.model))


def night_table(
    front: Sequence[Point],
    night_ctx_floor: int = DEFAULT_NIGHT_CTX_FLOOR,
) -> list[Point]:
    """Front points clearing Night ctx floor, sorted by maximin IQ."""
    if not front:
        return []
    eligible = [p for p in front if p.ctx >= night_ctx_floor]
    if eligible:
        return sorted(eligible, key=lambda p: (-p.iq_min, -p.ctx, -p.tps, p.model))
    return sorted(front, key=lambda p: (-p.ctx, -p.iq_min, -p.tps, p.model))


def _rank_axis(
    rows: Sequence[dict[str, str]],
    category: str,
) -> list[tuple[str, float, float | None, int | None]]:
    best: dict[str, tuple[float, float | None, int | None]] = {}
    for row in rows:
        if (row.get("category") or "").strip() != category:
            continue
        if not _is_measurement_row(row):
            continue
        model = (row.get("model") or "").strip()
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
    day_iq_ratio: float,
    night_ctx_floor: int,
    mode: str,
    rows: Sequence[dict[str, str]] | None = None,
) -> str:
    del incomplete  # kept in signature for callers; default view is Day/Night only
    lines: list[str] = []
    front = pareto_front(complete)

    if mode in ("pareto", "day", "all"):
        day_rows = day_table(front, day_iq_ratio=day_iq_ratio)
        iq_best = max((p.iq_min for p in front), default=0.0)
        floor = day_iq_ratio * iq_best
        lines.append(f"DAY  (min >= {day_iq_ratio:.2f}*IQ_best={iq_best:.4f} -> {floor:.4f})  pick=#1")
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

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank models from results.tsv (ADR 0006/0008).",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_TSV,
        help="Path to results.tsv (default: repo root)",
    )
    parser.add_argument(
        "--mode",
        choices=("pareto", "day", "night", "claw", "coding", "all"),
        default="pareto",
        help="Report view (default: pareto = Day + Night markdown tables)",
    )
    parser.add_argument(
        "--day-iq-ratio",
        type=float,
        default=DEFAULT_DAY_IQ_RATIO,
        help=f"ADR 0008 Day IQ band ratio (default {DEFAULT_DAY_IQ_RATIO})",
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
        day_iq_ratio=args.day_iq_ratio,
        night_ctx_floor=args.night_ctx_floor,
        mode=args.mode,
        rows=rows,
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
