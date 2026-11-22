"""Frozen ImageNet application backbones and the shared seven-class head."""

from __future__ import annotations

import tensorflow as tf

from .utils import SEED

ARCHITECTURES = ("efficientnetb0", "mobilenetv2", "resnet50")


@tf.keras.utils.register_keras_serializable(package="skin_lesion")
class FrozenDropoutClassifier(tf.keras.Model):
    """Keep backbone inference fixed while the head dropout follows training mode."""

    def __init__(
        self, architecture: str, num_classes: int = 7,
        image_size: tuple[int, int] = (224, 224), weights: str | None = "imagenet",
        seed: int = SEED, dropout: float = 0.3, l2: float = 0.0, **kwargs,
    ):
        super().__init__(**kwargs)
        self.architecture = architecture
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.pretrained_weights = weights
        self.seed = seed
        self.dropout_rate = dropout
        self.l2_strength = l2
        constructors = {
            "efficientnetb0": tf.keras.applications.EfficientNetB0,
            "mobilenetv2": tf.keras.applications.MobileNetV2,
            "resnet50": tf.keras.applications.ResNet50,
        }
        self.backbone = constructors[architecture](
            include_top=False, weights=weights, input_shape=(*self.image_size, 3),
        )
        self.backbone.trainable = False
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")
        self.hidden = tf.keras.layers.Dense(
            128, activation="relu", name="dense_128",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2) if l2 else None,
        )
        self.head_dropout = tf.keras.layers.Dropout(dropout, seed=seed + 2, name="head_dropout")
        self.class_probabilities = tf.keras.layers.Dense(
            num_classes, activation="softmax", name="class_probabilities",
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed + 1),
        )

    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        features = self.backbone(inputs, training=False)
        hidden = self.hidden(self.pool(features))
        hidden = self.head_dropout(hidden, training=training)
        return self.class_probabilities(hidden)

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.call(tf.zeros((1, *self.image_size, 3)), training=False)
        super().build(input_shape)

    def get_config(self) -> dict:
        return {
            **super().get_config(), "architecture": self.architecture,
            "num_classes": self.num_classes, "image_size": self.image_size,
            "weights": self.pretrained_weights, "seed": self.seed,
            "dropout": self.dropout_rate, "l2": self.l2_strength,
        }


def build_frozen_classifier(
    architecture: str, *, num_classes: int = 7,
    image_size: tuple[int, int] = (224, 224),
    weights: str | None = "imagenet", seed: int = SEED,
    dropout: float = 0.0, l2: float = 0.0,
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
    if not 0 <= dropout < 1 or l2 < 0:
        raise ValueError("Dropout must be in [0, 1) and L2 must be nonnegative.")

    if dropout:
        model = FrozenDropoutClassifier(
            architecture=name, num_classes=num_classes, image_size=image_size,
            weights=weights, seed=seed, dropout=dropout, l2=l2,
            name=f"{name}_frozen_classifier",
        )
        model.build((None, *image_size, 3))
        return model, model.backbone

    inputs = tf.keras.Input(shape=(*image_size, 3), name="rgb_image")
    backbone = constructors[name](include_top=False, weights=weights, input_shape=(*image_size, 3))
    backbone.trainable = False
    features = backbone(inputs, training=False)
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    hidden = tf.keras.layers.Dense(
        128, activation="relu", name="dense_128",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2) if l2 else None,
    )(pooled)
    if dropout:
        hidden = tf.keras.layers.Dropout(dropout, seed=seed + 2, name="head_dropout")(hidden)
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


def unfreeze_efficientnet_upper_stages(model: tf.keras.Model, *, first_block: int = 6) -> dict:
    """Train the selected upper stages and top conv, keeping BN frozen."""
    if first_block not in (0, 6, 7):
        raise ValueError("EfficientNet fine-tuning must use all non-BN layers or start at block6 or block7.")
    backbone = model.get_layer("efficientnetb0")
    names = [layer.name for layer in backbone.layers]
    for prefix in ("block6", "block7"):
        if not any(name.startswith(prefix) for name in names):
            raise ValueError(f"EfficientNet backbone has no {prefix} stage.")
    if "top_conv" not in names:
        raise ValueError("EfficientNet backbone has no top_conv layer.")
    backbone.trainable = True
    prefixes = tuple(f"block{number}" for number in range(first_block, 8))
    for layer in backbone.layers:
        layer.trainable = (
            (first_block == 0 or layer.name.startswith(prefixes) or layer.name == "top_conv")
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )
    for head_name in ("dense_128", "class_probabilities"):
        model.get_layer(head_name).trainable = True
    trainable = [layer.name for layer in backbone.layers if layer.trainable]
    frozen = [layer.name for layer in backbone.layers if not layer.trainable]
    if not trainable or any(isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
                            for layer in backbone.layers):
        raise AssertionError("Upper-stage trainability boundary is invalid.")
    if first_block == 0 and any(any(weight is variable for variable in model.trainable_weights)
                                for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
                                for weight in layer.weights):
        raise AssertionError("A backbone BatchNormalization variable is trainable.")
    return {
        "boundary": ("all non-BatchNormalization backbone layers; all backbone BatchNormalization frozen"
                     if first_block == 0 else ", ".join((*[f"block{number}*" for number in range(first_block, 8)],
                                                         "top_conv; all BatchNormalization frozen"))),
        "first_trainable_block": first_block,
        "backbone_trainable_layers": trainable,
        "backbone_frozen_layers": frozen,
        "backbone_trainable_variables": [weight.path for weight in backbone.trainable_weights],
        "backbone_frozen_variables": [weight.path for weight in backbone.non_trainable_weights],
        "model_trainable_variables": [weight.path for weight in model.trainable_weights],
        "model_non_trainable_variables": [weight.path for weight in model.non_trainable_weights],
        "backbone_trainable_parameters": sum(int(tf.size(weight)) for weight in backbone.trainable_weights),
        "backbone_frozen_parameters": sum(int(tf.size(weight)) for weight in backbone.non_trainable_weights),
        **parameter_counts(model),
        "non_trainable_parameters": sum(int(tf.size(weight)) for weight in model.non_trainable_weights),
    }
