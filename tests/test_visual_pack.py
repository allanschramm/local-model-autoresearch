"""Visual pack stub guard (issue #54, ADR 0014 phase 4).

The visual pack is docs-only: one frozen workspace-shaped prompt plus a
short operator guide. It runs in Pi against the `model-up` server on the
same Fingerprint file as daily Pi, and it must never elect a champion or
write rank (`results.tsv` / `results.db` / `on_front` / Pareto).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUIDE = REPO_ROOT / "docs" / "discovery" / "visual-pack.md"
PROMPT = REPO_ROOT / "docs" / "discovery" / "visual-pack-prompt.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_guide_and_prompt_exist() -> None:
    assert GUIDE.is_file(), "missing operator guide docs/discovery/visual-pack.md"
    assert PROMPT.is_file(), "missing frozen prompt docs/discovery/visual-pack-prompt.md"


def test_guide_pins_same_fingerprint_file_as_daily_pi() -> None:
    text = _read(GUIDE)
    assert "fingerprints/" in text, "guide must pin the Fingerprint file bus"
    assert "model-up" in text or "model_up" in text, "guide must serve via the launcher"


def test_guide_states_no_rank_write_and_optional_rubric() -> None:
    text = _read(GUIDE)
    assert "on_front" in text, "guide must state visual results never write on_front"
    assert "results.tsv" in text, "guide must state no TSV column for visual results"
    assert "rubric" in text.lower(), "guide must note the camera rubric is optional"


def test_prompt_is_workspace_shaped_and_frozen() -> None:
    text = _read(PROMPT)
    assert "workspace" in text.lower(), "prompt must be workspace-shaped"
    assert "notes.py" in text, "prompt must carry its self-contained fixture"
    assert "frozen" in text.lower(), "prompt must be marked frozen"


def test_no_python_rank_writer_for_visual_pack() -> None:
    hits = []
    for root in ("autoresearch", "scripts"):
        for path in (REPO_ROOT / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "visual-pack" in text or "visual_pack" in text:
                hits.append(str(path.relative_to(REPO_ROOT)))
    assert hits == [], f"visual pack must stay docs-only, no rank writer: {hits}"
