"""Hill-Climbing Search Strategy over the per-model Pareto Set (ADR 0006).

Neighbor generation and Random Restarts live here; Trial acceptance no longer
uses scalar Val Score — a Neighbor improves when it joins or improves the
per-model Pareto Set (issue #7). autoloop is not wired yet; the legacy scalar
keep rules stay as a compat shim for it.
"""

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from autoresearch.core.pareto import ObjectiveVector, dominates, pareto_set

Config = dict[str, Any]


@dataclass
class Neighbor:
    config: Config
    changed: str
    old: Any
    new: Any


class SearchStrategy:
    """
    Deep module encapsulating the Hill-Climbing Search logic.
    Provides leverage by standardising Neighbor generation, per-model Pareto Set
    updates, and Random Restarts across different search spaces.
    """

    def __init__(
        self,
        search_space: dict[str, list[Any]],
        *,
        known: Iterable[ObjectiveVector] = (),
        use_pareto_tiebreaker: bool = False,
    ):
        self.search_space = search_space
        # Legacy flag consumed only by the not-yet-wired scalar keep rules
        # (autoloop compat; issue #7 keeps autoloop out of scope).
        self.use_pareto_tiebreaker = use_pareto_tiebreaker
        # Per-model Pareto Set state: every Trial outcome recorded, merged per
        # Fingerprint by the caller. The front is derived, never stored.
        self.known = list(known)

    def get_config_key(self, cfg: Config) -> str:
        """Deterministically serialize the search-space parameters of a config."""
        return str(sorted((k, v) for k, v in cfg.items() if k in self.search_space or k == "MODEL"))

    @staticmethod
    def is_batch_consistent(cfg: Config) -> bool:
        """Reject neighbors where UBATCH_SIZE > BATCH_SIZE."""
        batch = cfg.get("BATCH_SIZE", cfg.get("batch_size"))
        ubatch = cfg.get("UBATCH_SIZE", cfg.get("ubatch_size"))
        if batch is None or ubatch is None:
            return True
        try:
            return int(ubatch) <= int(batch)
        except (TypeError, ValueError):
            return True

    def get_neighbors(self, current_cfg: Config) -> list[Neighbor]:
        """Generate single-parameter perturbations of current config."""
        neighbors = []
        for param, candidates in self.search_space.items():
            current = current_cfg.get(param)
            try:
                idx = candidates.index(current)
            except ValueError:
                # Current value not in search space; try all candidates
                for val in candidates:
                    if val != current:
                        n = current_cfg.copy()
                        n[param] = val
                        if self.is_batch_consistent(n):
                            neighbors.append(
                                Neighbor(config=n, changed=param, old=current, new=val)
                            )
                continue

            # Adjacent neighbors in the ordered list
            if idx > 0:
                n = current_cfg.copy()
                n[param] = candidates[idx - 1]
                if self.is_batch_consistent(n):
                    neighbors.append(
                        Neighbor(config=n, changed=param, old=current, new=candidates[idx - 1])
                    )
            if idx < len(candidates) - 1:
                n = current_cfg.copy()
                n[param] = candidates[idx + 1]
                if self.is_batch_consistent(n):
                    neighbors.append(
                        Neighbor(config=n, changed=param, old=current, new=candidates[idx + 1])
                    )

        random.shuffle(neighbors)
        return neighbors

    def random_restart(
        self, visited: set[str], current_cfg: Config, max_attempts: int = 100
    ) -> Config | None:
        """Generate a random valid configuration not in visited to escape local maxima."""
        for _ in range(max_attempts):
            new_cfg = current_cfg.copy()
            for param, values in self.search_space.items():
                new_cfg[param] = random.choice(values)
            if not self.is_batch_consistent(new_cfg):
                continue
            n_key = self.get_config_key(new_cfg)
            if n_key not in visited:
                return new_cfg
        return None

    @property
    def pareto_set(self) -> list[ObjectiveVector]:
        """Current per-model front: complete, mutually non-dominated vectors.

        Incomplete vectors never join the front (ADR 0006 merge rule) but stay
        in `known` so they compete once their axes fill in.
        """
        return pareto_set(self.known)

    def record(self, vector: ObjectiveVector) -> None:
        """Record a Trial outcome into the per-model Set.

        Every Trial is kept (results-store semantics); statuses derive from the
        front instead of being decided at write time.
        """
        self.known.append(vector)

    def improves_set(self, vector: ObjectiveVector) -> bool:
        """Neighbor acceptance: joins or improves the per-model Pareto Set.

        Replaces the scalar Val Score / Pareto Tie-Breaker keep rules: a
        Neighbor improves the Set iff its Objective Vector is non-dominated by
        the current front. Joining (incomparable) counts, and so does improving
        (dominating older members shrinks the front). Incomplete vectors never
        compete (ADR 0006: incomplete merges, does not dominate). Caller merges
        repeated measurements per Fingerprint before recording, so an exact
        duplicate never reaches the Set (equal vectors would judge each other as
        joining, not dominating).
        """
        return vector.complete and not any(dominates(other, vector) for other in self.pareto_set)

    def is_local_maximum(self, candidates: Iterable[ObjectiveVector]) -> bool:
        """Local Maxima = no Neighbor joins or improves the per-model Set."""
        return not any(self.improves_set(candidate) for candidate in candidates)

    def is_improvement(
        self,
        baseline_score: float,
        baseline_tps: float,
        baseline_vram: float,
        new_score: float,
        new_tps: float,
        new_vram: float,
    ) -> tuple[bool, str]:
        """
        LEGACY scalar keep rules — autoloop is not yet wired to the per-model
        Pareto Set (issue #7 scope). Kept so autoloop keeps running until its
        wiring ticket; the Search keep truth is `improves_set` / the Set.
        Rules (Allan's matrix):
          Score+  Speed+  → KEEP
          Score+  Speed-  → KEEP
          Score+  Speed=  → KEEP
          Score-  Speed+  → DISCARD
          Score-  Speed-  → DISCARD
          Score=  Speed+  → KEEP
          Score=  Speed=  → DISCARD
        Returns (is_improvement, reason_string).
        """
        delta = new_score - baseline_score

        # Score improved → always KEEP
        if new_score > baseline_score + 0.0001:
            return True, f"Score improved (Δ={delta:+.6f})"

        # Score tied → KEEP only if speed improved
        if self.use_pareto_tiebreaker and abs(new_score - baseline_score) <= 0.0001:
            if new_tps > baseline_tps * 1.05:
                return True, f"Score tied, TPS improved (+{new_tps - baseline_tps:.1f})"
            tps_tied = abs(new_tps - baseline_tps) <= abs(baseline_tps) * 0.05
            if tps_tied and baseline_vram > 0 and new_vram < baseline_vram * 0.95:
                return (
                    True,
                    f"Score and TPS tied, VRAM improved (-{baseline_vram - new_vram:.2f}GB)",
                )

        return False, ""

    def format_config_summary(self, cfg: Config) -> str:
        """One-line summary of tunable params for logging."""
        parts = []
        for p in self.search_space:
            v = cfg.get(p)
            if v is not None:
                parts.append(f"{p}={v}")
        return " ".join(parts)
