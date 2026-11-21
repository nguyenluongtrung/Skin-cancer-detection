"""RGB image datasets, architecture-specific preprocessing, and Day 1 smoke check.

Run from the repository root: python -m src.data_pipeline --output-dir day1_artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

from .utils import SEED, runtime_info, save_json, set_seed
from .visualization import plot_first_batch

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 7


def decode_resize(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode original RGB JPEG to float32 pixels in [0, 255]."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    return tf.ensure_shape(image, (224, 224, 3)), tf.cast(label, tf.int32)


def preprocess_for(architecture: str) -> Callable[[tf.Tensor], tf.Tensor]:
    """Return the Keras application input transform; call exactly once after augmentation."""
    name = architecture.lower()
    if name == "efficientnetb0":
        # Keras EfficientNet includes input rescaling internally.
        return tf.keras.applications.efficientnet.preprocess_input
    if name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if name == "resnet50":
        return tf.keras.applications.resnet50.preprocess_input
    raise ValueError(f"Unknown architecture: {architecture}")


def make_image_dataset(
    frame: pd.DataFrame, *, training: bool = False, batch_size: int = 32,
    architecture: str | None = None,
    augment: Callable[[tf.Tensor], tf.Tensor] | None = None,
    seed: int = SEED,
) -> tf.data.Dataset:
    """Keep CSV row order for evaluation; shuffle and augment training rows only."""
    if batch_size < 1 or frame.empty:
        raise ValueError("Use a positive batch size and a nonempty split.")
    if augment is not None and not training:
        raise ValueError("Augmentation is permitted only when training=True.")
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["class_index"].astype("int64").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE)
    if augment is not None:
        dataset = dataset.map(lambda image, label: (augment(image), label), num_parallel_calls=tf.data.AUTOTUNE)
    if architecture is not None:
        preprocess = preprocess_for(architecture)
        dataset = dataset.map(lambda image, label: (preprocess(image), label), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def inspect_batch(train: pd.DataFrame, output_dir: Path, batch_size: int = 32) -> tf.data.Dataset:
    """Save the same raw-image batch preview and range checks as the original script."""
    dataset = make_image_dataset(train, training=False, batch_size=batch_size)
    images, labels = next(iter(dataset.take(1)))
    pixel_values, label_values = images.numpy(), labels.numpy()
    print("Batch image shape:", tuple(images.shape), "dtype:", images.dtype.name)
    print("Batch label shape:", tuple(labels.shape), "dtype:", labels.dtype.name)
    print("First labels:", label_values[:16].tolist())
    print("Pixel min/max:", float(pixel_values.min()), float(pixel_values.max()))
    if images.shape[1:] != (224, 224, 3) or not np.isfinite(pixel_values).all():
        raise AssertionError("Invalid image shape or nonfinite pixels.")
    if pixel_values.min() < 0 or pixel_values.max() > 255 or not np.isin(label_values, np.arange(NUM_CLASSES)).all():
        raise AssertionError("Invalid raw pixel range or class index.")
    plot_first_batch(pixel_values, label_values, output_dir)
    return dataset


def smoke_check(train_dataset: tf.data.Dataset, output_dir: Path) -> dict:
    """Train one small random-network batch; verify finite loss and checkpoint reload."""
    set_seed()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Rescaling(1 / 255),
        tf.keras.layers.Conv2D(8, 3, strides=4, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="day1_smoke_only")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    checkpoint = output_dir / "smoke_checkpoint.keras"
    start = time.monotonic()
    history = model.fit(
        train_dataset.take(1), epochs=1, verbose=0,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(str(checkpoint), save_best_only=False)],
    )
    loss = float(history.history["loss"][0])
    if not np.isfinite(loss) or not checkpoint.is_file():
        raise AssertionError("Smoke loss is nonfinite or checkpoint was not saved.")
    restored = tf.keras.models.load_model(checkpoint)
    images, _ = next(iter(train_dataset.take(1)))
    difference = float(np.max(np.abs(model(images, training=False).numpy() - restored(images, training=False).numpy())))
    if not np.isfinite(difference) or difference > 1e-5:
        raise AssertionError(f"Saved predictions changed after reload: {difference}")
    result = {
        "one_batch_loss": loss, "checkpoint": str(checkpoint),
        "reload_max_probability_difference": difference, "seconds": time.monotonic() - start,
    }
    save_json(result, output_dir / "smoke_result.json")
    print("Smoke check:", json.dumps(result, indent=2))
    return result


def run_smoke(output_dir: Path = Path("day1_artifacts"), train_csv: Path | None = None, batch_size: int = 32) -> dict:
    """Day 1 smoke entry point; requires existing preparation artifacts."""
    output_dir = Path(output_dir).expanduser().resolve()
    train_csv = Path(train_csv).expanduser().resolve() if train_csv else output_dir / "train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Prepare data first; missing {train_csv}")
    train = pd.read_csv(train_csv, dtype={"image_id": str, "lesion_id": str, "dx": str})
    set_seed()
    environment = runtime_info(batch_size)
    save_json(environment, output_dir / "environment.json")
    print("Runtime:", json.dumps(environment, indent=2))
    dataset = inspect_batch(train, output_dir, batch_size)
    return smoke_check(dataset, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("day1_artifacts"))
    parser.add_argument("--train-csv", type=Path, help="Override the prepared training CSV")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_smoke(args.output_dir, args.train_csv, args.batch_size)


if __name__ == "__main__":
    main()
