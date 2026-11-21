"""Configuration and callback checks for the short optimizer comparison."""

import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.train import (OPTIMIZER_COMPARISON_CONFIG, build_comparison_optimizer,
                       comparison_callbacks, load_optimizer_comparison_config)


class OptimizerComparisonTests(unittest.TestCase):
    def test_selected_recipe_and_resolved_optimizer_settings(self):
        config = load_optimizer_comparison_config(OPTIMIZER_COMPARISON_CONFIG, Path("outputs"))
        self.assertEqual(config["max_epochs"], 5)
        self.assertEqual(config["l2"], 1e-4)
        self.assertFalse(config["augmentation"])
        for name, settings in config["optimizers"].items():
            resolved = build_comparison_optimizer(name, settings).get_config()
            for key, value in settings.items():
                if isinstance(value, float):
                    self.assertAlmostEqual(resolved[key], value, places=7)
                else:
                    self.assertEqual(resolved[key], value)
            self.assertFalse(resolved["use_ema"])
            self.assertIsNone(resolved["weight_decay"])
            if name != "sgd_momentum":
                self.assertIsNotNone(resolved["epsilon"])

    def test_callbacks_only_score_and_checkpoint(self):
        dataset = tf.data.Dataset.from_tensor_slices((
            np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int32),
        )).batch(1)
        callbacks, f1_callback = comparison_callbacks(dataset, np.zeros(1, dtype=int), Path("unused.keras"))
        self.assertIs(callbacks[0], f1_callback)
        self.assertEqual(len(callbacks), 2)
        self.assertIsInstance(callbacks[1], tf.keras.callbacks.ModelCheckpoint)


if __name__ == "__main__":
    unittest.main()
