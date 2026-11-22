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


class MultiplicativeBrightness(tf.keras.layers.Layer):
    """Multiply each float32 RGB image by an independent seeded factor."""

    def __init__(self, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.generator = tf.keras.random.SeedGenerator(seed)

    def call(self, images: tf.Tensor, training: bool = False) -> tf.Tensor:
        images = tf.cast(images, tf.float32)
        if not training:
            return images
        shape = (tf.shape(images)[0], 1, 1, 1) if images.shape.rank == 4 else (1, 1, 1)
        factor = tf.keras.random.uniform(shape, 0.95, 1.05, seed=self.generator)
        return tf.clip_by_value(images * factor, 0.0, 255.0)


def build_mild_augmentation(seed: int = SEED) -> tf.keras.Sequential:
    """Day 9 raw RGB policy; geometric layers use bilinear, reflect fill.

    Keras rotation factors are turns. RandomZoom's factor is the inverse
    coordinate scale minus one, so these bounds give image scales 0.95..1.05.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(10 / 360, interpolation="bilinear",
                                       fill_mode="reflect", seed=seed + 1),
        tf.keras.layers.RandomZoom((1 / 1.05 - 1, 1 / 0.95 - 1),
                                   interpolation="bilinear", fill_mode="reflect", seed=seed + 2),
        MultiplicativeBrightness(seed + 3),
    ], name="mild_raw_rgb_augmentation")
