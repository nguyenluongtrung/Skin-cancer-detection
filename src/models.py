"""Frozen ImageNet application backbones and the shared seven-class head."""

from __future__ import annotations

import tensorflow as tf

from .utils import SEED

ARCHITECTURES = ("efficientnetb0", "mobilenetv2", "resnet50")


def build_frozen_classifier(
    architecture: str, *, num_classes: int = 7,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet", seed: int = SEED,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Build backbone → GAP → Dense(128, ReLU) → Dense(classes, softmax).

    The backbone is called with training=False, keeping Batch Normalization in
    inference mode even while the head is trained.
    """
    name = architecture.lower()
    constructors = {
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "resnet50": tf.keras.applications.ResNet50,
    }
    if name not in constructors:
        raise ValueError(f"Unknown architecture {architecture!r}; choose from {ARCHITECTURES}.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    if weights not in ("imagenet", None):
        raise ValueError("Only ImageNet weights or None for a smoke check are supported.")

    inputs = tf.keras.Input(shape=(*image_size, 3), name="rgb_image")
    backbone = constructors[name](include_top=False, weights=weights, input_shape=(*image_size, 3))
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    hidden = tf.keras.layers.Dense(
        128, activation="relu", name="dense_128",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
    )(pooled)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="class_probabilities",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
    )(hidden)
    model = tf.keras.Model(inputs, outputs, name=f"{name}_frozen_classifier")
    if backbone.trainable or backbone.trainable_weights:
        raise AssertionError("Backbone unexpectedly has trainable weights.")
    return model, backbone


def parameter_counts(model: tf.keras.Model) -> dict[str, int]:
    trainable = sum(int(tf.size(weight)) for weight in model.trainable_weights)
    return {"total_parameters": int(model.count_params()), "trainable_parameters": trainable}
