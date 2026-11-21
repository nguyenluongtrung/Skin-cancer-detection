"""Class weighting and regularized head checks."""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.train import training_class_weights
from src.models import build_frozen_classifier


class HeadExperimentTests(unittest.TestCase):
    def test_weights_follow_saved_integer_labels(self):
        mapping = {"first": 1, "second": 0}
        train = pd.DataFrame({"class_index": [0, 0, 0, 1]})
        self.assertEqual(training_class_weights(train, mapping), {0: 2 / 3, 1: 2.0})

    def test_dropout_and_l2_attach_only_to_head(self):
        model, backbone = build_frozen_classifier("efficientnetb0", num_classes=7,
                                                   weights=None, dropout=0.3, l2=1e-4)
        self.assertFalse(backbone.trainable)
        self.assertEqual(model.get_layer("head_dropout").rate, 0.3)
        self.assertIsNotNone(model.get_layer("dense_128").kernel_regularizer)
        self.assertEqual(len(model.losses), 1)
        images = tf.random.uniform((2, 224, 224, 3), maxval=255, seed=1)
        batch_norm = next(layer for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization))
        moving_mean = batch_norm.moving_mean.numpy().copy()
        inference = model(images, training=False).numpy()
        training = model(images, training=True).numpy()
        self.assertGreater(float(np.max(np.abs(training - inference))), 1e-6)
        self.assertTrue(np.array_equal(batch_norm.moving_mean.numpy(), moving_mean))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "head.keras"
            model.save(checkpoint)
            reloaded = tf.keras.models.load_model(checkpoint)
            self.assertTrue(np.allclose(reloaded(images, training=False).numpy(), inference, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
