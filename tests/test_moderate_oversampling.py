"""Checks for seeded moderate sampling and epoch batch boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.data_pipeline import (make_image_dataset, moderate_sampling_probabilities,
                               sample_training_indices)


class ModerateOversamplingTests(unittest.TestCase):
    def test_keras_steps_per_epoch_consumes_one_prebatched_epoch_at_a_time(self) -> None:
        """Check the finite concatenated schedule used by the Day 8 fit call."""
        class RecordingModel(tf.keras.Model):
            def __init__(self):
                super().__init__()
                self.seen: list[int] = []

            def train_step(self, data):
                self.seen.extend(int(value) for value in data[1].numpy())
                return {"loss": tf.constant(0.0)}

        first = tf.data.Dataset.from_tensor_slices((np.zeros((7, 1)), np.arange(7))).batch(3)
        second = tf.data.Dataset.from_tensor_slices((np.zeros((7, 1)), np.arange(7, 14))).batch(3)
        model = RecordingModel()
        model.compile(optimizer="sgd", run_eagerly=True)
        model.fit(first.concatenate(second), steps_per_epoch=3, epochs=2, verbose=0)
        self.assertEqual(model.seen, list(range(14)))

    def test_seeded_draws_and_epoch_batches_keep_image_label_pairs(self) -> None:
        labels = np.arange(7, dtype=np.int64)
        probabilities = moderate_sampling_probabilities(labels)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        first = sample_training_indices(labels, probabilities, 7, np.random.default_rng(42))
        repeated = sample_training_indices(labels, probabilities, 7, np.random.default_rng(42))
        np.testing.assert_array_equal(first, repeated)
        second = sample_training_indices(labels, probabilities, 7, np.random.default_rng(43))
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index in range(7):
                path = Path(temporary) / f"{index}.jpg"
                Image.fromarray(np.full((8, 8, 3), index * 30, dtype=np.uint8)).save(path)
                paths.append(str(path))
            frame = pd.DataFrame({"image_path": paths, "class_index": labels})
            dataset = make_image_dataset(frame, training=True, batch_size=3,
                                         sampled_indices_by_epoch=[first, second])
            batches = list(dataset.as_numpy_iterator())
        self.assertEqual([len(batch_labels) for _, batch_labels in batches], [3, 3, 1, 3, 3, 1])
        observed_labels = np.concatenate([batch_labels for _, batch_labels in batches])
        np.testing.assert_array_equal(observed_labels, labels[np.concatenate([first, second])])
        observed_pixels = np.array([image.mean() for images, _ in batches for image in images])
        np.testing.assert_allclose(observed_pixels, observed_labels * 30, atol=1.0)


if __name__ == "__main__":
    unittest.main()
