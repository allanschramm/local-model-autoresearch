import unittest

from autoresearch.core.pareto import ObjectiveVector
from autoresearch.core.search import SearchStrategy

VEC = dict(ctx=131072, tps=30.0, agentic=0.6, coding=0.4)


def vec(**overrides) -> ObjectiveVector:
    """Fake Trial outcome; complete by default, override axes per test."""
    values = dict(VEC)
    values.update(overrides)
    return ObjectiveVector(**values)


class TestSearchStrategy(unittest.TestCase):
    def test_is_improvement_simple_improvement(self):
        # Score improved
        strategy = SearchStrategy({}, use_pareto_tiebreaker=False)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70,
            baseline_tps=30.0,
            baseline_vram=4.0,
            new_score=0.75,
            new_tps=30.0,
            new_vram=4.0,
        )
        self.assertTrue(is_imp)
        self.assertIn("Score improved", reason)

    def test_is_improvement_no_improvement(self):
        # Score regressed/same without tiebreaker
        strategy = SearchStrategy({}, use_pareto_tiebreaker=False)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70,
            baseline_tps=30.0,
            baseline_vram=4.0,
            new_score=0.70,
            new_tps=50.0,
            new_vram=2.0,
        )
        self.assertFalse(is_imp)

    def test_is_improvement_pareto_tps(self):
        # Score tied, TPS improved (> 5%)
        strategy = SearchStrategy({}, use_pareto_tiebreaker=True)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70,
            baseline_tps=30.0,
            baseline_vram=4.0,
            new_score=0.70,
            new_tps=32.0,
            new_vram=4.0,
        )
        self.assertTrue(is_imp)
        self.assertIn("TPS improved", reason)

    def test_is_improvement_pareto_vram(self):
        # Score tied, TPS same, VRAM improved — but VRAM is no longer a tie-breaker
        strategy = SearchStrategy({}, use_pareto_tiebreaker=True)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70,
            baseline_tps=30.0,
            baseline_vram=4.0,
            new_score=0.70,
            new_tps=29.0,
            new_vram=3.5,
        )
        self.assertTrue(is_imp)
        self.assertIn("VRAM improved", reason)

    def test_is_improvement_pareto_no_tps_no_vram(self):
        # Score tied, TPS regressed heavily, VRAM improved (not enough for TPS drop)
        strategy = SearchStrategy({}, use_pareto_tiebreaker=True)
        is_imp, reason = strategy.is_improvement(
            baseline_score=0.70,
            baseline_tps=30.0,
            baseline_vram=4.0,
            new_score=0.70,
            new_tps=20.0,
            new_vram=3.5,
        )
        self.assertFalse(is_imp)

    def test_random_restart(self):
        search_space = {"param1": [1, 2], "param2": [10]}
        strategy = SearchStrategy(search_space)
        current = {"param1": 1, "param2": 10}

        # If we visit the current config, it should pick the other option
        visited = {strategy.get_config_key(current)}
        new_cfg = strategy.random_restart(visited, current)
        self.assertIsNotNone(new_cfg)
        self.assertEqual(new_cfg["param1"], 2)
        self.assertEqual(new_cfg["param2"], 10)

        # If all configurations are visited, it should return None
        visited.add(strategy.get_config_key(new_cfg))
        final_cfg = strategy.random_restart(visited, current, max_attempts=50)
        self.assertIsNone(final_cfg)

    def test_config_key_distinguishes_models(self):
        strategy = SearchStrategy({"THREADS": [8, 12]})

        first = strategy.get_config_key({"MODEL": "first.gguf", "THREADS": 8})
        second = strategy.get_config_key({"MODEL": "second.gguf", "THREADS": 8})

        self.assertNotEqual(first, second)

    def test_is_batch_consistent(self):
        strategy = SearchStrategy({})
        self.assertTrue(strategy.is_batch_consistent({"BATCH_SIZE": 512, "UBATCH_SIZE": 128}))
        self.assertFalse(strategy.is_batch_consistent({"BATCH_SIZE": 256, "UBATCH_SIZE": 512}))

    def test_get_neighbors_filters_invalid_batch(self):
        space = {"BATCH_SIZE": [256, 512], "UBATCH_SIZE": [128, 256, 512]}
        strategy = SearchStrategy(space)
        neighbors = strategy.get_neighbors({"BATCH_SIZE": 256, "UBATCH_SIZE": 256})
        for n in neighbors:
            self.assertLessEqual(n.config["UBATCH_SIZE"], n.config["BATCH_SIZE"])


class TestSearchStrategyParetoSet(unittest.TestCase):
    """Issue #7: Neighbor acceptance = non-dominated improvement on the per-model Set."""

    def test_first_trial_joins_empty_set(self):
        # A single complete point is its own front.
        strategy = SearchStrategy({})
        self.assertTrue(strategy.improves_set(vec()))

    def test_dominating_neighbor_improves_set(self):
        strategy = SearchStrategy({}, known=[vec()])
        # Same tradeoff, strictly faster TPS → dominates the known member.
        self.assertTrue(strategy.improves_set(vec(tps=35.0)))

    def test_dominated_neighbor_does_not_improve_set(self):
        strategy = SearchStrategy({}, known=[vec()])
        # Slower TPS, everything else equal → dominated, no improvement.
        self.assertFalse(strategy.improves_set(vec(tps=25.0)))

    def test_incomparable_neighbor_joins_set(self):
        strategy = SearchStrategy({}, known=[vec()])
        # Better TPS, worse ctx → incomparable → joins the front.
        self.assertTrue(strategy.improves_set(vec(ctx=65536, tps=45.0)))

    def test_incomplete_neighbor_never_competes(self):
        # ADR 0006: incomplete merges, never dominates, never joins the front.
        strategy = SearchStrategy({}, known=[vec()])
        self.assertFalse(strategy.improves_set(vec(agentic=None)))
        self.assertFalse(strategy.improves_set(vec(tps=None, coding=None)))

    def test_record_keeps_every_trial_but_front_only_complete_non_dominated(self):
        strategy = SearchStrategy({}, known=[vec()])
        strategy.record(vec(tps=25.0))  # dominated
        strategy.record(vec(agentic=None))  # incomplete
        front = strategy.pareto_set
        self.assertEqual(len(front), 1)
        self.assertEqual(front[0], vec())

    def test_improvement_shrinks_front_when_it_dominates_members(self):
        strategy = SearchStrategy({}, known=[vec(tps=30.0), vec(tps=35.0, agentic=0.5)])
        candidate = vec(tps=36.0, agentic=0.7)
        self.assertTrue(strategy.improves_set(candidate))
        strategy.record(candidate)
        self.assertEqual(strategy.pareto_set, [candidate])

    def test_local_maximum_when_no_neighbor_improves(self):
        strategy = SearchStrategy({}, known=[vec()])
        dominated = [vec(tps=25.0), vec(tps=20.0, agentic=0.4)]
        self.assertTrue(strategy.is_local_maximum(dominated))

    def test_not_local_maximum_when_neighbor_joins(self):
        strategy = SearchStrategy({}, known=[vec()])
        neighbors = [vec(tps=25.0), vec(ctx=65536, tps=45.0)]
        self.assertFalse(strategy.is_local_maximum(neighbors))

    def test_known_vectors_from_constructor_seed_the_front(self):
        strategy = SearchStrategy({}, known=[vec(), vec(tps=25.0)])
        self.assertEqual(len(strategy.pareto_set), 1)
        self.assertEqual(strategy.pareto_set[0], vec())


if __name__ == "__main__":
    unittest.main()
