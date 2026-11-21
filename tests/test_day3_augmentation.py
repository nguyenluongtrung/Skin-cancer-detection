"""Focused checks for the new augmentation and six-run comparison."""

import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from src.augmentation import build_moderate_augmentation
from src.compare import build_comparison


class Day3Tests(unittest.TestCase):
    def test_raw_augmented_pixels_stay_in_documented_range(self):
        images = tf.random.uniform((3, 224, 224, 3), 0, 255, seed=42)
        result = build_moderate_augmentation(42)(images, training=True).numpy()
        self.assertEqual(result.shape, (3, 224, 224, 3))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(), 0)
        self.assertLessEqual(result.max(), 255)

    def test_six_run_changes_are_augmented_minus_baseline(self):
        rows = []
        for architecture in ("efficientnetb0", "mobilenetv2", "resnet50"):
            for augmented in (False, True):
                rows.append({
                    "run_id": f"{architecture}_{augmented}", "architecture": architecture,
                    "augmentation": augmented,
                    "comparison_group": "day3_frozen_moderate_augmentation" if augmented else "day2_frozen_no_augmentation",
                    "best_epoch": 2, "actual_epochs": 4, "training_seconds": 12,
                    "validation_accuracy": 0.8 if augmented else 0.7,
                    "validation_macro_f1": 0.6 if augmented else 0.5,
                    "validation_balanced_accuracy": 0.7 if augmented else 0.6,
                })
        comparison, changes = build_comparison(pd.DataFrame(rows))
        self.assertEqual(len(comparison), 6)
        self.assertEqual(len(changes), 3)
        self.assertAlmostEqual(changes.iloc[0]["delta_validation_macro_f1"], 0.1)
        with self.assertRaisesRegex(ValueError, "Expected one"):
            build_comparison(pd.DataFrame(rows[:-1]))


if __name__ == "__main__":
    unittest.main()
