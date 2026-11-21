"""Training-only, raw RGB augmentation shared by the three backbones."""

from __future__ import annotations

import tensorflow as tf

from .utils import SEED


def build_moderate_augmentation(seed: int = SEED) -> tf.keras.Sequential:
    """Augment float RGB pixels in [0, 255] before model preprocessing.

    Keras rotation factors are fractions of 360 degrees. Zoom factors are
    fractions of image dimensions. Brightness 0.08 means up to 20.4 raw
    pixel levels of change; RandomBrightness clips to [0, 255].
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        tf.keras.layers.RandomRotation(20 / 360, fill_mode="reflect", seed=seed + 1),
        tf.keras.layers.RandomZoom(0.10, 0.10, fill_mode="reflect", seed=seed + 2),
        tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255), seed=seed + 3),
    ], name="moderate_raw_rgb_augmentation")
