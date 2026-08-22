"""Morris elementary-effects screen over engine Search Space knobs.

Pins low-effect engine parameters before Neighbor Search. Not Taguchi;
does not replace SearchStrategy.get_neighbors.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from autoresearch.core.search import SearchStrategy

Config = dict[str, Any]


def varying_space(search_space: dict[str, list]) -> dict[str, list]:
    return {k: v for k, v in search_space.items() if len(v) >= 2}


def one_step(cfg: Config, param: str, candidates: list, rng: random.Random) -> Config | None:
    n = cfg.copy()
    current = n.get(param)
    try:
        idx = candidates.index(current)
    except ValueError:
        others = [v for v in candidates if v != current]
        if not others:
            return None
        n[param] = rng.choice(others)
        return n if SearchStrategy.is_batch_consistent(n) else None
    options = []
    if idx > 0:
        options.append(idx - 1)
    if idx < len(candidates) - 1:
        options.append(idx + 1)
    if not options:
        return None
    n[param] = candidates[rng.choice(options)]
    return n if SearchStrategy.is_batch_consistent(n) else None


def generate_trajectories(
    search_space: dict[str, list], r: int, rng: random.Random, seed_cfg: Config
) -> list[list[tuple[Config, str]]]:
    varying = varying_space(search_space)
    keys = list(varying)
    trajectories: list[list[tuple[Config, str]]] = []
    for _ in range(max(0, int(r))):
        start = seed_cfg.copy()
        ok = False
        for _try in range(50):
            trial = seed_cfg.copy()
            for k, levels in varying.items():
                trial[k] = rng.choice(levels)
            if SearchStrategy.is_batch_consistent(trial):
                start = trial
                ok = True
                break
        if not ok:
            start = seed_cfg.copy()
        traj: list[tuple[Config, str]] = [(start, "")]
        if keys:
            for param in rng.sample(keys, k=len(keys)):
                stepped = one_step(traj[-1][0], param, varying[param], rng)
                if stepped is None:
                    continue
                traj.append((stepped, param))
        trajectories.append(traj)
    return trajectories


def elementary_effects(
    samples: list[tuple[str, float, float]],
) -> dict[str, dict[str, float | int]]:
    by_param: dict[str, list[float]] = {}
    for param, y_before, y_after in samples:
        if not param:
            continue
        by_param.setdefault(param, []).append(y_after - y_before)
    out: dict[str, dict[str, float | int]] = {}
    for param, ees in by_param.items():
        n = len(ees)
        mu_star = statistics.mean(abs(x) for x in ees)
        sigma = statistics.pstdev(ees) if n >= 2 else 0.0
        out[param] = {"mu_star": mu_star, "sigma": sigma, "n": n}
    return out


def pins_from_effects(
    effects: dict[str, dict],
    search_space: dict[str, list],
    best_cfg: Config,
    frac: float = 0.10,
) -> dict[str, object]:
    mu_vals = [
        float(v.get("mu_star", 0.0)) for v in effects.values() if int(v.get("n", 0) or 0) > 0
    ]
    max_mu = max(mu_vals) if mu_vals else 0.0
    if max_mu == 0:
        return {}
    pins: dict[str, object] = {}
    for param, stats in effects.items():
        if int(stats.get("n", 0) or 0) == 0:
            continue
        if param not in search_space:
            continue
        if float(stats.get("mu_star", 0.0)) < frac * max_mu:
            if param in best_cfg:
                pins[param] = best_cfg[param]
    return pins
