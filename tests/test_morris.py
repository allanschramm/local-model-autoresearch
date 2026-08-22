"""Morris screen: pin unused factors; never emit UBATCH > BATCH."""

import random
import unittest

from autoresearch.core.morris import (
    elementary_effects,
    generate_trajectories,
    one_step,
    pins_from_effects,
    varying_space,
)


class TestMorris(unittest.TestCase):
    def test_varying_space_drops_singleton(self):
        space = {"FLASH_ATTN": ["on"], "A": [0, 1]}
        self.assertEqual(varying_space(space), {"A": [0, 1]})

    def test_one_step_never_ubatch_gt_batch(self):
        rng = random.Random(0)
        cfg = {"BATCH_SIZE": 256, "UBATCH_SIZE": 128}
        candidates = [64, 128, 256, 512]
        for _ in range(40):
            stepped = one_step(cfg, "UBATCH_SIZE", candidates, rng)
            if stepped is None:
                continue
            self.assertLessEqual(int(stepped["UBATCH_SIZE"]), int(stepped["BATCH_SIZE"]))

    def test_pins_unused_factor(self):
        # y = 10*A; B unused → pin B
        space = {"A": [0, 1, 2], "B": [10, 20]}
        samples = [
            ("A", 0.0, 10.0),
            ("A", 10.0, 20.0),
            ("B", 20.0, 20.0),
            ("B", 20.0, 20.5),
            ("A", 0.0, 10.0),
            ("A", 10.0, 20.0),
        ]
        effects = elementary_effects(samples)
        best = {"A": 2, "B": 10}
        pins = pins_from_effects(effects, space, best, frac=0.10)
        self.assertIn("B", pins)
        self.assertNotIn("A", pins)
        self.assertEqual(pins["B"], 10)

    def test_generate_trajectories_shape(self):
        rng = random.Random(1)
        space = {"A": [0, 1, 2], "B": [10, 20]}
        seed = {"A": 1, "B": 10, "BATCH_SIZE": 512, "UBATCH_SIZE": 128}
        trajs = generate_trajectories(space, 2, rng, seed)
        self.assertEqual(len(trajs), 2)
        self.assertEqual(trajs[0][0][1], "")


if __name__ == "__main__":
    unittest.main()
