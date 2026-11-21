"""Frozen-backbone comparisons; explicit --smoke or --run required.

Examples (repository root):
    python -m src.train --architecture efficientnetb0 --smoke
    python -m src.train --architecture efficientnetb0 --run
    python -m src.train --all --run
    python -m src.train --optimizer-choice adam --config configs/frozen_optimizer_comparison.json --run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from .augmentation import build_moderate_augmentation
from .compare import (HEAD_EXPERIMENT_STAGES, OPTIMIZER_NAMES,
                      write_head_experiment_comparisons, write_optimizer_comparison)
from .data_pipeline import load_training_partitions, make_image_dataset, preprocess_for
from .evaluate import evaluate_validation
from .models import ARCHITECTURES, build_frozen_classifier, parameter_counts
from .utils import append_experiment_log, create_run_directory, runtime_info, save_json, set_seed
from .visualization import plot_augmentation_pairs, plot_learning_curves

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "day2_frozen_no_aug.json"
HEAD_EXPERIMENT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "frozen_head_experiments.json"
OPTIMIZER_COMPARISON_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "frozen_optimizer_comparison.json"


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "seed": 42, "image_size": [224, 224], "pretrained_weights": "imagenet",
        "backbone_trainable": False, "class_weights": False,
        "dropout": 0, "l2": 0, "head_units": 128, "optimizer": "adam",
        "learning_rate": 0.001, "loss": "sparse_categorical_crossentropy",
        "selection_metric": "val_macro_f1",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen comparison requires {key}={value!r}; got {config.get(key)!r}.")
    if config.get("augmentation") not in (True, False):
        raise ValueError("augmentation must be true or false.")
    if config["augmentation"]:
        policy = {
            "flips": "horizontal_and_vertical", "rotation_degrees": 20,
            "zoom_fraction": 0.10, "brightness_fraction_of_255": 0.08,
            "input_value_range": [0, 255],
        }
        if config.get("augmentation_policy") != policy:
            raise ValueError("Day 3 augmentation policy differs from the shared moderate policy.")
    if not 1 <= int(config.get("max_epochs", 0)) <= 12:
        raise ValueError("max_epochs must be between 1 and 12.")
    if int(config.get("early_stopping_patience", -1)) != 3:
        raise ValueError("Day 2 early stopping patience must be 3.")
    if int(config.get("batch_size", 0)) not in (16, 32):
        raise ValueError("Day 2 batch size must be 16 or 32.")
    if tuple(config.get("architectures", [])) != ARCHITECTURES:
        raise ValueError(f"Day 2 architecture list must be {ARCHITECTURES}.")
    return config


def baseline_comparator(outputs_dir: Path, architecture: str, batch_size: int) -> str:
    """Find the corresponding Day 2 run; augmented runs start from ImageNet."""
    log_path = Path(outputs_dir) / "experiment_log.csv"
    if not log_path.is_file():
        raise FileNotFoundError(f"Day 2 experiment log is missing: {log_path}")
    log = pd.read_csv(log_path)
    matches = log[
        (log["comparison_group"] == "day2_frozen_no_augmentation")
        & (log["architecture"] == architecture)
        & (log["batch_size"] == batch_size)
        & (log["seed"] == 42)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one matched Day 2 {architecture} baseline; found {len(matches)}.")
    return str(matches.iloc[0]["run_id"])


def load_head_experiment_config(path: Path) -> dict:
    """Keep the selected frozen recipe fixed across head experiments."""
    config = json.loads(path.read_text(encoding="utf-8"))
    baseline = load_config(DEFAULT_CONFIG)
    for key in ("seed", "image_size", "batch_size", "max_epochs", "early_stopping_patience",
                "pretrained_weights", "backbone_trainable", "augmentation", "head_units",
                "optimizer", "learning_rate", "loss", "selection_metric"):
        if config.get(key) != baseline[key]:
            raise ValueError(f"Head experiments must retain baseline {key}={baseline[key]!r}.")
    if config.get("architecture") != "efficientnetb0":
        raise ValueError("Selected architecture must be efficientnetb0.")
    return config


def load_optimizer_comparison_config(path: Path, outputs_dir: Path) -> dict:
    """Check the short-budget comparison against the selected frozen recipe."""
    config = json.loads(path.read_text(encoding="utf-8"))
    reference_id = config.get("reference_run_id")
    if not reference_id:
        raise ValueError("Optimizer comparison requires the selected Day 4 reference run ID.")
    row, reference, _ = logged_run(outputs_dir, reference_id)
    if (row.get("run_status") == "invalid" or reference.get("stage") != "T3_E0_L"
            or reference.get("architecture") != "efficientnetb0"):
        raise ValueError("Reference must be the valid selected L2 run.")
    decisions = json.loads((outputs_dir / "day4_decisions.json").read_text(encoding="utf-8"))
    if decisions[-1]["selected_run_id"] != reference_id:
        raise ValueError("Reference differs from the Day 4 validation decision.")
    for key in ("architecture", "seed", "image_size", "batch_size", "pretrained_weights",
                "backbone_trainable", "augmentation", "head_units", "class_weights",
                "dropout", "l2", "loss", "selection_metric"):
        if config.get(key) != reference.get(key):
            raise ValueError(f"Optimizer comparison differs from selected recipe on {key}.")
    if config.get("comparison_group") != "frozen_optimizer_comparison" or config.get("max_epochs") != 5:
        raise ValueError("Optimizer comparison requires its own group and exactly five epochs.")
    expected = {
        "adam": {"learning_rate": 0.001, "beta_1": 0.9, "beta_2": 0.999, "amsgrad": False},
        "rmsprop": {"learning_rate": 0.001, "rho": 0.9, "momentum": 0.0, "centered": False},
        "sgd_momentum": {"learning_rate": 0.01, "momentum": 0.9, "nesterov": False},
    }
    if config.get("optimizers") != expected:
        raise ValueError("Optimizer settings differ from the controlled comparison plan.")
    return config


def build_comparison_optimizer(name: str, settings: dict) -> tf.keras.optimizers.Optimizer:
    """Create an optimizer with fresh state and framework-default epsilon."""
    constructors = {
        "adam": tf.keras.optimizers.Adam,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "sgd_momentum": tf.keras.optimizers.SGD,
    }
    optimizer = constructors[name](**settings)
    resolved = optimizer.get_config()
    if resolved.get("use_ema") or resolved.get("weight_decay") not in (None, 0):
        raise ValueError("Optimizer EMA and optimizer weight decay must be disabled.")
    return optimizer


def initial_head_digest(model: tf.keras.Model) -> str:
    """Fingerprint head initialization for cross-run fairness checks."""
    digest = hashlib.sha256()
    for name in ("dense_128", "class_probabilities"):
        for weight in model.get_layer(name).get_weights():
            digest.update(np.asarray(weight).tobytes())
    return digest.hexdigest()


def training_class_weights(train: pd.DataFrame, mapping: dict[str, int]) -> dict[int, float]:
    """Calculate inverse-frequency weights from saved training integer labels."""
    counts = train["class_index"].value_counts()
    expected = set(mapping.values())
    if set(counts.index) != expected:
        raise ValueError("Every mapped class must occur in training labels.")
    return {int(index): len(train) / (len(mapping) * int(counts[index])) for index in sorted(expected)}


def logged_run(outputs_dir: Path, run_id: str) -> tuple[dict, dict, dict]:
    log = pd.read_csv(outputs_dir / "experiment_log.csv")
    rows = log.loc[log.run_id == run_id]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one logged run {run_id!r}.")
    directory = outputs_dir / "runs" / run_id
    return (rows.iloc[0].to_dict(),
            json.loads((directory / "config.json").read_text(encoding="utf-8")),
            json.loads((directory / "metrics.json").read_text(encoding="utf-8")))


def valid_stage_rows(log: pd.DataFrame, stage: str) -> pd.DataFrame:
    """Exclude runs invalidated by a verified implementation error."""
    rows = log.loc[log.comparison_group == stage]
    if "run_status" in rows:
        rows = rows.loc[rows.run_status.fillna("valid") != "invalid"]
    return rows


def select_head_experiment_parent(stage: str, outputs_dir: Path, requested: str | None, baseline_id: str) -> tuple[str, bool, float, float]:
    """Use the validation winner from the immediately preceding comparison."""
    if stage == HEAD_EXPERIMENT_STAGES[0]:
        if requested:
            raise ValueError("Class weighting uses the fixed baseline; omit --parent-run-id.")
        return baseline_id, True, 0.0, 0.0
    if not requested:
        raise ValueError("Later stages require --parent-run-id from the preceding comparison.")
    row, config, _ = logged_run(outputs_dir, requested)
    if row["architecture"] != "efficientnetb0" or str(row["augmentation"]).lower() != "false":
        raise ValueError("Parent must be an unaugmented EfficientNetB0 run.")
    log = pd.read_csv(outputs_dir / "experiment_log.csv")
    if stage == HEAD_EXPERIMENT_STAGES[1]:
        weighted = valid_stage_rows(log, HEAD_EXPERIMENT_STAGES[0])
        if len(weighted) != 1:
            raise ValueError("Run class weighting exactly once before dropout.")
        winner = max((baseline_id, str(weighted.iloc[0].run_id)),
                     key=lambda run: logged_run(outputs_dir, run)[2]["validation_macro_f1"])
        if requested != winner:
            raise ValueError(f"Dropout parent must be selected winner {winner}.")
        return requested, bool(config.get("class_weights", False)), 0.3, 0.0
    dropout = valid_stage_rows(log, HEAD_EXPERIMENT_STAGES[1])
    if len(dropout) != 1:
        raise ValueError("Run dropout exactly once before L2.")
    dropout_id = str(dropout.iloc[0].run_id)
    prior_id = str(dropout.iloc[0].parent_run_id)
    winner = max((prior_id, dropout_id),
                 key=lambda run: logged_run(outputs_dir, run)[2]["validation_macro_f1"])
    if requested != winner:
        raise ValueError(f"L2 parent must be selected winner {winner}.")
    return requested, bool(config.get("class_weights", False)), float(config.get("dropout", 0)), 1e-4


class ValidationMacroF1(tf.keras.callbacks.Callback):
    """Compute full-validation macro F1 before checkpoint and early stopping."""

    def __init__(self, validation_ds: tf.data.Dataset, labels: np.ndarray, num_classes: int):
        super().__init__()
        self.validation_ds = validation_ds
        self.labels = np.asarray(labels, dtype=int)
        self.num_classes = num_classes
        self.scores: list[float] = []
        self.learning_rates: list[float] = []

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if logs is None:
            raise AssertionError("Keras did not provide epoch logs.")
        probabilities = self.model.predict(self.validation_ds, verbose=0)
        if probabilities.shape != (len(self.labels), self.num_classes):
            raise AssertionError("Validation prediction count or class dimension changed.")
        score = float(f1_score(
            self.labels, probabilities.argmax(axis=1),
            labels=list(range(self.num_classes)), average="macro", zero_division=0,
        ))
        learning_rate = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logs["val_macro_f1"] = score
        logs["learning_rate"] = learning_rate
        self.scores.append(score)
        self.learning_rates.append(learning_rate)
        print(f"epoch {epoch + 1}: val_macro_f1={score:.4f}, lr={learning_rate:g}", flush=True)


def assert_validation_order(validation_ds: tf.data.Dataset, expected: np.ndarray) -> None:
    """Catch shuffled or reordered validation datasets before scoring."""
    observed = np.concatenate([labels.numpy() for _, labels in validation_ds])
    if not np.array_equal(observed, np.asarray(expected, dtype=int)):
        raise AssertionError("Validation dataset label order differs from validation.csv.")


def compile_baseline(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def baseline_callbacks(validation_ds: tf.data.Dataset, labels: np.ndarray, checkpoint: Path, patience: int):
    f1_callback = ValidationMacroF1(validation_ds, labels, num_classes=7)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint), monitor="val_macro_f1", mode="max", save_best_only=True,
    )
    stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_macro_f1", mode="max", patience=patience, restore_best_weights=False,
    )
    # Order matters: F1 must be in logs before the next two callbacks read it.
    return [f1_callback, checkpoint_callback, stop_callback], f1_callback


def comparison_callbacks(validation_ds: tf.data.Dataset, labels: np.ndarray, checkpoint: Path):
    """Checkpoint macro F1 without stopping or changing the learning rate."""
    f1_callback = ValidationMacroF1(validation_ds, labels, num_classes=7)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint), monitor="val_macro_f1", mode="max", save_best_only=True,
    )
    return [f1_callback, checkpoint_callback], f1_callback


def train_one(
    architecture: str, *, config: dict, splits_dir: Path, outputs_dir: Path, batch_size: int,
    stage: str | None = None, parent_run_id: str | None = None,
    optimizer_name: str | None = None,
) -> tuple[str, dict]:
    """Train one fresh ImageNet frozen model and score its best checkpoint."""
    train, validation, mapping = load_training_partitions(splits_dir)
    if optimizer_name:
        existing_log = pd.read_csv(outputs_dir / "experiment_log.csv")
        existing = existing_log.loc[
            (existing_log.comparison_group == config["comparison_group"])
            & (existing_log.optimizer == optimizer_name)
        ]
        if not existing.empty:
            raise ValueError(f"{optimizer_name} already has a comparison run; existing outputs will not be overwritten.")
        parent_id, use_weights, dropout, l2 = None, False, 0.0, config["l2"]
    elif stage:
        baseline_id = baseline_comparator(outputs_dir, architecture, batch_size)
        parent_id, use_weights, dropout, l2 = select_head_experiment_parent(stage, outputs_dir, parent_run_id, baseline_id)
        if not valid_stage_rows(pd.read_csv(outputs_dir / "experiment_log.csv"), stage).empty:
            raise ValueError(f"{stage} already completed; existing outputs will not be overwritten.")
        _, parent_config, _ = logged_run(outputs_dir, parent_id)
        for key in ("seed", "image_size", "batch_size", "max_epochs", "early_stopping_patience",
                    "pretrained_weights", "backbone_trainable", "augmentation", "head_units",
                    "optimizer", "learning_rate", "loss", "selection_metric"):
            if parent_config[key] != config[key]:
                raise ValueError(f"Parent differs from head experiment configuration on {key}.")
    else:
        parent_id, use_weights, dropout, l2 = None, False, 0.0, 0.0
    comparator_run_id = parent_id if stage else (baseline_comparator(outputs_dir, architecture, batch_size) if config["augmentation"] else None)
    class_weights = training_class_weights(train, mapping) if use_weights else None
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    model, backbone = build_frozen_classifier(
        architecture, num_classes=len(mapping), image_size=tuple(config["image_size"]),
        weights="imagenet", seed=config["seed"], dropout=dropout, l2=l2,
    )
    if backbone.trainable or backbone.trainable_weights:
        raise AssertionError("Backbone is not fully frozen.")
    head_digest = initial_head_digest(model) if optimizer_name else None
    optimizer = build_comparison_optimizer(optimizer_name, config["optimizers"][optimizer_name]) if optimizer_name else None
    if optimizer is None:
        compile_baseline(model)
    else:
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    counts = parameter_counts(model)
    augment = build_moderate_augmentation(config["seed"]) if config["augmentation"] else None
    train_ds = make_image_dataset(
        train, training=True, batch_size=batch_size, architecture=architecture,
        augment=augment, seed=config["seed"],
    )
    validation_ds = make_image_dataset(validation, training=False, batch_size=batch_size, architecture=architecture)
    assert_validation_order(validation_ds, validation["class_index"].to_numpy())

    run_label = f"{architecture}_{optimizer_name}" if optimizer_name else (stage.lower() if stage else architecture)
    run_id, run_dir = create_run_directory(outputs_dir, run_label)
    checkpoint = run_dir / "best.keras"
    run_config = {
        **config, "run_id": run_id, "architecture": architecture,
        "batch_size": batch_size, "parent_run_id": parent_id,
        "comparator": comparator_run_id or config["comparison_group"],
        "comparator_run_id": comparator_run_id,
        "splits_dir": str(Path(splits_dir).resolve()),
        "class_mapping": mapping,
        "preprocessing": "raw RGB 0-255 for EfficientNetB0; Keras preprocess_input once for MobileNetV2/ResNet50",
        "checkpoint_selection": "highest full-validation macro F1; first epoch wins ties",
        "stage": stage, "class_weights": use_weights, "class_weight_values": class_weights,
        "dropout": dropout, "l2": l2,
        "loss_note": "Logged loss includes L2 when enabled; class weights apply to training loss only. Validation metrics are unweighted. Losses across different objectives are not directly comparable.",
    }
    if optimizer_name:
        split_files = ("train.csv", "validation.csv", "class_mapping.json", "split_manifest.json")
        run_config.update({
            "optimizer": optimizer_name,
            "optimizer_settings": optimizer.get_config(),
            "learning_rate": config["optimizers"][optimizer_name]["learning_rate"],
            "initial_head_sha256": head_digest,
            "split_file_sha256": {
                name: hashlib.sha256((Path(splits_dir) / name).read_bytes()).hexdigest()
                for name in split_files
            },
            "training_policy": "five complete epochs; no early stopping or learning-rate callbacks",
            "reference_validation_macro_f1": logged_run(outputs_dir, config["reference_run_id"])[2]["validation_macro_f1"],
        })
    save_json(run_config, run_dir / "config.json")
    if augment is not None:
        examples = train.groupby("class_index", sort=True, group_keys=False).head(1)
        originals, example_labels = next(iter(make_image_dataset(examples, batch_size=7)))
        montage_augment = build_moderate_augmentation(config["seed"])
        changed = montage_augment(originals, training=True).numpy()
        plot_augmentation_pairs(originals.numpy(), changed, example_labels.numpy(), run_dir / "augmentation_examples.png")
    if optimizer_name:
        callbacks, f1_callback = comparison_callbacks(
            validation_ds, validation["class_index"].to_numpy(dtype=int), checkpoint,
        )
    else:
        callbacks, f1_callback = baseline_callbacks(
            validation_ds, validation["class_index"].to_numpy(dtype=int),
            checkpoint, config["early_stopping_patience"],
        )
    started = time.perf_counter()
    history = model.fit(
        train_ds, validation_data=validation_ds, epochs=config["max_epochs"],
        callbacks=callbacks, class_weight=class_weights, verbose=2,
    )
    duration = time.perf_counter() - started
    if optimizer_name:
        expected_updates = config["max_epochs"] * int(np.ceil(len(train) / batch_size))
        if len(f1_callback.scores) != 5 or int(optimizer.iterations.numpy()) != expected_updates:
            raise AssertionError("Short comparison did not complete five epochs with equal update counts.")
        if not np.allclose(f1_callback.learning_rates, config["optimizers"][optimizer_name]["learning_rate"]):
            raise AssertionError("Optimizer learning rate changed during the short comparison.")
    if not checkpoint.is_file():
        raise AssertionError("No best checkpoint was saved.")
    history_frame = pd.DataFrame(history.history)
    history_frame.insert(0, "epoch", np.arange(1, len(f1_callback.scores) + 1))
    history_frame["val_macro_f1"] = f1_callback.scores
    history_frame["learning_rate"] = f1_callback.learning_rates
    history_frame.to_csv(run_dir / "history.csv", index=False)
    plot_learning_curves(history_frame, run_dir / "learning_curves.png")
    best_epoch = int(np.argmax(f1_callback.scores) + 1)

    best_model = tf.keras.models.load_model(checkpoint)
    best_probabilities = best_model.predict(validation_ds, verbose=0)
    predictions, scores = evaluate_validation(validation, best_probabilities, mapping)
    if not np.isclose(scores["validation_macro_f1"], f1_callback.scores[best_epoch - 1], atol=1e-7):
        raise AssertionError("Best checkpoint score differs from the recorded best epoch.")
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)

    runtime = runtime_info(batch_size)
    metrics = {
        "run_id": run_id, "architecture": architecture,
        "augmentation": bool(config["augmentation"]), "comparator_run_id": comparator_run_id, **scores,
        "actual_epochs": len(f1_callback.scores), "best_epoch": best_epoch,
        "training_seconds": duration, **counts,
        "framework_versions": {key: runtime[key] for key in ("python", "tensorflow", "keras", "pandas", "numpy", "scikit_learn")},
        "detected_training_device": "GPU" if runtime["gpus"] else "CPU",
        "visible_gpus": runtime["gpus"],
        "best_checkpoint": str(checkpoint),
        "validation_images": len(validation),
        "stage": stage, "parent_run_id": parent_id, "class_weight_values": class_weights,
        "logged_loss_includes_l2": bool(l2),
    }
    if optimizer_name:
        metrics.update({"optimizer": optimizer_name, "optimizer_settings": optimizer.get_config(),
                        "optimizer_updates": expected_updates, "initial_head_sha256": head_digest,
                        "split_file_sha256": run_config["split_file_sha256"],
                        "reference_run_id": config["reference_run_id"]})
    save_json(metrics, run_dir / "metrics.json")
    append_experiment_log(outputs_dir, {
        "run_id": run_id, "comparison_group": stage or config["comparison_group"],
        "parent_run_id": parent_id or "", "comparator_run_id": comparator_run_id or "", "architecture": architecture,
        "augmentation": bool(config["augmentation"]), "class_weights": use_weights,
        "dropout": dropout, "l2": l2, "optimizer": optimizer_name or "adam",
        "learning_rate": config["optimizers"][optimizer_name]["learning_rate"] if optimizer_name else 1e-3,
        "batch_size": batch_size, "seed": config["seed"],
        "actual_epochs": metrics["actual_epochs"], "best_epoch": best_epoch,
        "validation_accuracy": scores["validation_accuracy"],
        "validation_macro_f1": scores["validation_macro_f1"],
        "validation_balanced_accuracy": scores["validation_balanced_accuracy"],
        "training_seconds": duration, **counts,
        "detected_training_device": metrics["detected_training_device"],
        "run_dir": str(run_dir),
        "logged_loss_includes_l2": bool(l2),
        "run_status": "valid",
        "optimizer_updates": expected_updates if optimizer_name else "",
        "initial_head_sha256": head_digest or "",
    })
    if stage:
        write_head_experiment_comparisons(outputs_dir, mapping)
    if optimizer_name:
        write_optimizer_comparison(outputs_dir, config["reference_run_id"])
    print(f"Saved run {run_id}: best epoch {best_epoch}, validation macro F1 {scores['validation_macro_f1']:.4f}")
    return run_id, metrics


def smoke_one(architecture: str, *, config: dict, splits_dir: Path) -> dict:
    """One batch and one epoch with random backbone weights; no experiment output."""
    train, validation, mapping = load_training_partitions(splits_dir)
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    train_small = train.groupby("class_index", sort=True, group_keys=False).head(1)
    validation_small = validation.groupby("class_index", sort=True, group_keys=False).head(1)
    model, backbone = build_frozen_classifier(
        architecture, num_classes=len(mapping), image_size=(224, 224),
        weights=None, seed=config["seed"],
    )
    if backbone.trainable or backbone.trainable_weights:
        raise AssertionError("Backbone is not fully frozen.")
    compile_baseline(model)
    augment = build_moderate_augmentation(config["seed"]) if config["augmentation"] else None
    train_ds = make_image_dataset(
        train_small, training=True, batch_size=7, architecture=architecture,
        augment=augment, seed=config["seed"],
    )
    validation_ds = make_image_dataset(validation_small, training=False, batch_size=7, architecture=architecture)
    assert_validation_order(validation_ds, validation_small["class_index"].to_numpy())
    first_val, first_labels = next(iter(validation_ds))
    second_val, second_labels = next(iter(validation_ds))
    if not np.array_equal(first_val.numpy(), second_val.numpy()) or not np.array_equal(first_labels.numpy(), second_labels.numpy()):
        raise AssertionError("Repeated validation loading is not deterministic.")
    raw_val, _ = next(iter(make_image_dataset(validation_small, batch_size=7)))
    expected_val = preprocess_for(architecture)(raw_val).numpy()
    if not np.allclose(first_val.numpy(), expected_val, atol=1e-5):
        raise AssertionError("Validation preprocessing differs from exactly one application transform.")
    train_images, train_labels = next(iter(train_ds))
    if train_images.shape != (7, 224, 224, 3) or set(train_labels.numpy()) != set(range(7)):
        raise AssertionError("Training image shape or labels are invalid.")
    if not np.isfinite(train_images.numpy()).all():
        raise AssertionError("Training pixels contain nonfinite values.")
    limits = {
        "efficientnetb0": (0, 255),
        "mobilenetv2": (-1, 1),
        "resnet50": (-123.68, 151.061),
    }
    lower, upper = limits[architecture]
    if float(tf.reduce_min(train_images)) < lower - 1e-4 or float(tf.reduce_max(train_images)) > upper + 1e-4:
        raise AssertionError(f"{architecture} preprocessed training pixels are outside [{lower}, {upper}].")
    augmented_raw_range = None
    if augment is not None:
        raw_augmented, _ = next(iter(make_image_dataset(
            train_small, training=True, batch_size=7,
            augment=build_moderate_augmentation(config["seed"]), seed=config["seed"],
        )))
        augmented_raw_range = [float(tf.reduce_min(raw_augmented)), float(tf.reduce_max(raw_augmented))]
        if augmented_raw_range[0] < 0 or augmented_raw_range[1] > 255:
            raise AssertionError("Augmented raw pixels left [0, 255].")
    with tempfile.TemporaryDirectory(prefix="ham_day2_smoke_") as temporary:
        checkpoint = Path(temporary) / "best.keras"
        callbacks, f1_callback = baseline_callbacks(
            validation_ds, validation_small["class_index"].to_numpy(dtype=int), checkpoint, patience=3,
        )
        history = model.fit(train_ds, validation_data=validation_ds, epochs=1, callbacks=callbacks, verbose=0)
        loss = float(history.history["loss"][0])
        if not np.isfinite(loss) or not checkpoint.is_file():
            raise AssertionError("Smoke loss is nonfinite or the best checkpoint is missing.")
        restored = tf.keras.models.load_model(checkpoint)
        current = model.predict(validation_ds, verbose=0)
        reloaded = restored.predict(validation_ds, verbose=0)
        difference = float(np.max(np.abs(current - reloaded)))
        if not np.isfinite(difference) or difference > 1e-5:
            raise AssertionError(f"Checkpoint reload difference too large: {difference}")
    result = {
        "architecture": architecture, "smoke_only": True,
        "augmentation": bool(config["augmentation"]),
        "pretrained_weights": None, "training_images": len(train_small),
        "validation_images": len(validation_small), "one_epoch_loss": loss,
        "val_macro_f1_pipeline_check": f1_callback.scores[0],
        "checkpoint_reload_max_probability_difference": difference,
        "train_batch_shape": list(train_images.shape),
        "train_preprocessed_pixel_range": [float(tf.reduce_min(train_images)), float(tf.reduce_max(train_images))],
        "augmented_raw_pixel_range": augmented_raw_range,
        "validation_repeated_identical": True,
        **parameter_counts(model), "backbone_frozen": True,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    architecture_group = parser.add_mutually_exclusive_group(required=True)
    architecture_group.add_argument("--architecture", choices=ARCHITECTURES)
    architecture_group.add_argument("--all", action="store_true", help="Run all three architectures sequentially")
    architecture_group.add_argument("--stage", choices=HEAD_EXPERIMENT_STAGES, help="Run one sequential head experiment")
    architecture_group.add_argument("--optimizer-choice", choices=OPTIMIZER_NAMES,
                                    help="Run one five-epoch frozen optimizer comparison")
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--smoke", action="store_true", help="Tiny random-weight pipeline and checkpoint check")
    action_group.add_argument("--run", action="store_true", help="Start full ImageNet frozen training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--splits-dir", type=Path, default=Path("day1_artifacts"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, choices=(16, 32), help="Override the shared configured batch size")
    parser.add_argument("--parent-run-id", help="Validation-selected parent for dropout or L2")
    args = parser.parse_args()
    if args.stage and (args.smoke or args.batch_size):
        parser.error("Head experiments require --run and the fixed configured batch size.")
    if args.optimizer_choice and (args.smoke or args.batch_size or args.parent_run_id):
        parser.error("Optimizer comparison requires --run and its fixed configured settings.")
    if args.parent_run_id and not args.stage:
        parser.error("--parent-run-id is only for head experiments.")
    config_path = (OPTIMIZER_COMPARISON_CONFIG if args.optimizer_choice and args.config == DEFAULT_CONFIG
                   else HEAD_EXPERIMENT_CONFIG if args.stage and args.config == DEFAULT_CONFIG else args.config)
    config = (load_optimizer_comparison_config(config_path, args.outputs_dir) if args.optimizer_choice
              else load_head_experiment_config(config_path) if args.stage else load_config(config_path))
    batch_size = args.batch_size or config["batch_size"]
    architectures = config["architectures"] if args.all else [config["architecture"] if args.stage or args.optimizer_choice else args.architecture]
    for architecture in architectures:
        if args.smoke:
            smoke_one(architecture, config=config, splits_dir=args.splits_dir)
        else:
            train_one(
                architecture, config=config, splits_dir=args.splits_dir,
                outputs_dir=args.outputs_dir, batch_size=batch_size,
                stage=args.stage, parent_run_id=args.parent_run_id,
                optimizer_name=args.optimizer_choice,
            )


if __name__ == "__main__":
    main()
