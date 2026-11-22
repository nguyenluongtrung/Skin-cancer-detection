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
from sklearn.metrics import confusion_matrix, f1_score

from .augmentation import build_moderate_augmentation
from .compare import (HEAD_EXPERIMENT_STAGES, OPTIMIZER_NAMES,
                      write_head_experiment_comparisons, write_optimizer_comparison)
from .data_pipeline import (load_training_partitions, make_image_dataset,
                            moderate_sampling_probabilities, preprocess_for, sample_training_indices)
from .evaluate import evaluate_validation
from .models import ARCHITECTURES, build_frozen_classifier, parameter_counts, unfreeze_efficientnet_upper_stages
from .utils import append_experiment_log, create_run_directory, runtime_info, save_json, set_seed
from .visualization import plot_augmentation_pairs, plot_confusion_matrix, plot_learning_curves

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "day2_frozen_no_aug.json"
HEAD_EXPERIMENT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "frozen_head_experiments.json"
OPTIMIZER_COMPARISON_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "frozen_optimizer_comparison.json"
FINE_TUNE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "efficientnetb0_partial_fine_tune.json"
DEPTH_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "efficientnetb0_shallow_fine_tune.json"
OVERSAMPLING_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "efficientnetb0_moderate_oversampling.json"


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
        self.learning_rates_used: list[float] = []

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        self.learning_rates_used.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))

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


def load_fine_tune_config(path: Path, outputs_dir: Path) -> dict:
    """Require the selected Day 4 recipe and fixed partial fine-tuning settings."""
    config = json.loads(path.read_text(encoding="utf-8"))
    reference_id = json.loads((outputs_dir / "day4_decisions.json").read_text(encoding="utf-8"))[-1]["selected_run_id"]
    if config.get("parent_run_id") != reference_id:
        raise ValueError("Fine-tuning parent must be the selected Day 4 checkpoint.")
    row, parent, _ = logged_run(outputs_dir, reference_id)
    if row.get("run_status") == "invalid" or parent.get("stage") != "T3_E0_L":
        raise ValueError("Selected Day 4 parent is invalid or is not the L2 experiment.")
    for key in ("architecture", "seed", "image_size", "batch_size", "pretrained_weights",
                "augmentation", "class_weights", "dropout", "l2", "head_units", "loss", "selection_metric"):
        if config.get(key) != parent.get(key):
            raise ValueError(f"Fine-tuning config differs from selected parent on {key}.")
    first_block = config.get("first_trainable_block", 6)
    if first_block not in (6, 7):
        raise ValueError("Fine-tuning boundary must start at block6 or block7.")
    group = ("efficientnetb0_partial_fine_tune" if first_block == 6
             else "efficientnetb0_shallow_fine_tune")
    required = {"comparison_group": group, "optimizer": "adam",
                "learning_rate": 1e-5, "beta_1": 0.9, "beta_2": 0.999,
                "amsgrad": False, "max_epochs": 15, "reduce_lr_factor": 0.5,
                "reduce_lr_patience": 2, "min_lr": 1e-7,
                "early_stopping_patience": 5, "min_delta": 0.0}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"Fine-tuning requires {key}={value!r}.")
    if first_block == 7:
        day6_id = config.get("day6_comparator_run_id")
        if not day6_id:
            raise ValueError("Day 7 requires an explicit Day 6 comparator run ID.")
        day6_row, day6_config, day6_metrics = logged_run(outputs_dir, day6_id)
        if (day6_row.get("run_status") == "invalid"
                or day6_config.get("comparison_group") != "efficientnetb0_partial_fine_tune"
                or day6_config.get("parent_run_id") != reference_id
                or day6_metrics.get("retained_candidate_run_id") != day6_id):
            raise ValueError("Day 7 comparator must be the retained Day 6 run from the same parent.")
        for key in required:
            if key != "comparison_group" and config.get(key) != day6_config.get(key):
                raise ValueError(f"Day 7 differs from Day 6 on {key}.")
        for key in ("architecture", "parent_run_id", "seed", "image_size", "batch_size",
                    "pretrained_weights", "augmentation", "class_weights", "dropout",
                    "l2", "head_units", "loss", "selection_metric"):
            if config.get(key) != day6_config.get(key):
                raise ValueError(f"Day 7 differs from Day 6 on {key}.")
    return config


def load_oversampling_config(path: Path, outputs_dir: Path) -> dict:
    """Require the Day 6 recipe with only the sampling policy changed."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if (config.get("comparison_group") != "efficientnetb0_moderate_oversampling"
            or config.get("sampling_policy") != "sqrt_training_class_count"
            or config.get("first_trainable_block") != 6):
        raise ValueError("Day 8 requires the moderate oversampling policy and Day 6 boundary.")
    day6_id = config.get("day6_comparator_run_id")
    if not day6_id:
        raise ValueError("Day 8 requires the selected Day 6 control run ID.")
    row, day6, metrics = logged_run(outputs_dir, day6_id)
    if (row.get("run_status") == "invalid"
            or day6.get("comparison_group") != "efficientnetb0_partial_fine_tune"
            or metrics.get("retained_candidate_run_id") != day6_id
            or metrics.get("validation_macro_f1") != 0.6266277351856852):
        raise ValueError("Day 8 control is not the selected valid Day 6 run.")
    for key in ("architecture", "parent_run_id", "seed", "image_size", "batch_size",
                "max_epochs", "pretrained_weights", "augmentation", "class_weights",
                "dropout", "l2", "head_units", "optimizer", "learning_rate", "beta_1",
                "beta_2", "amsgrad", "loss", "selection_metric", "reduce_lr_factor",
                "reduce_lr_patience", "min_lr", "early_stopping_patience", "min_delta"):
        if config.get(key) != day6.get(key):
            raise ValueError(f"Day 8 differs from Day 6 on {key}.")
    if config.get("parent_checkpoint_sha256") != day6.get("parent_checkpoint_sha256"):
        raise ValueError("Day 8 parent checkpoint hash differs from Day 6.")
    if (day6.get("backbone_call_training") is not False
            or day6.get("preprocessing") != "original RGB 0-255; EfficientNet internal rescaling"):
        raise ValueError("Day 6 backbone call or preprocessing differs from the expected control.")
    return config


def fine_tune_callbacks(validation_ds: tf.data.Dataset, labels: np.ndarray, checkpoint: Path, config: dict):
    """Calculate full-validation F1 before every callback that monitors it."""
    f1_callback = ValidationMacroF1(validation_ds, labels, num_classes=7)
    return [
        f1_callback,
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_macro_f1",
                                           mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max",
                                             factor=config["reduce_lr_factor"],
                                             patience=config["reduce_lr_patience"],
                                             min_lr=config["min_lr"], min_delta=0.0),
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                         patience=config["early_stopping_patience"],
                                         min_delta=0.0, restore_best_weights=True),
    ], f1_callback


def compile_fine_tune(model: tf.keras.Model, config: dict) -> None:
    """Create a fresh Adam state after changing layer trainability."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config["learning_rate"], beta_1=config["beta_1"],
        beta_2=config["beta_2"], amsgrad=config["amsgrad"],
    )
    settings = optimizer.get_config()
    if settings.get("use_ema") or settings.get("weight_decay") not in (None, 0):
        raise ValueError("Optimizer EMA or weight decay is enabled.")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])


def audit_fine_tune_parent(config: dict, splits_dir: Path, outputs_dir: Path):
    """Re-evaluate the untouched Day 4 checkpoint on ordered validation rows."""
    parent_id = config["parent_run_id"]
    _, parent_config, parent_metrics = logged_run(outputs_dir, parent_id)
    parent_dir = outputs_dir / "runs" / parent_id
    checkpoint = parent_dir / "best.keras"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    train, validation, mapping = load_training_partitions(splits_dir)
    if mapping != parent_config["class_mapping"]:
        raise ValueError("Current class mapping differs from the selected run.")
    if Path(parent_config["splits_dir"]).resolve() != splits_dir.resolve():
        raise ValueError("Selected parent used a different split directory.")
    validation_ds = make_image_dataset(validation, training=False, batch_size=config["batch_size"],
                                       architecture="efficientnetb0")
    assert_validation_order(validation_ds, validation["class_index"].to_numpy())
    sample, _ = next(iter(validation_ds.take(1)))
    repeated, _ = next(iter(validation_ds.take(1)))
    if not np.array_equal(sample.numpy(), repeated.numpy()):
        raise AssertionError("Validation images changed between deterministic passes.")
    if sample.shape[1:] != (224, 224, 3) or float(tf.reduce_min(sample)) < 0 or float(tf.reduce_max(sample)) > 255:
        raise AssertionError("EfficientNet input shape or 0-255 range changed.")
    raw, _ = next(iter(make_image_dataset(validation.head(config["batch_size"]),
                                          batch_size=config["batch_size"])))
    if not np.array_equal(sample.numpy(), preprocess_for("efficientnetb0")(raw).numpy()):
        raise AssertionError("EfficientNet validation preprocessing changed.")
    model = tf.keras.models.load_model(checkpoint, compile=False)
    backbone = model.get_layer("efficientnetb0")
    hidden = model.get_layer("dense_128")
    output = model.get_layer("class_probabilities")
    regularizer = tf.keras.regularizers.serialize(hidden.kernel_regularizer)
    if (backbone.trainable or hidden.units != 128 or hidden.activation.__name__ != "relu"
            or output.units != 7 or output.activation.__name__ != "softmax"
            or not regularizer or not np.isclose(regularizer["config"]["l2"], 1e-4)):
        raise AssertionError("Checkpoint architecture or hidden-layer L2 differs from Day 4.")
    if len(model.get_layer("global_average_pooling").weights) != 0 or not np.isfinite(hidden.kernel.numpy()).all():
        raise AssertionError("Checkpoint head has invalid weights.")
    probabilities = model.predict(validation_ds, verbose=0)
    predictions, metrics = evaluate_validation(validation, probabilities, mapping)
    saved = pd.read_csv(parent_dir / "validation_predictions.csv")
    if (not np.array_equal(predictions["image_id"].to_numpy(), saved["image_id"].to_numpy())
            or not np.array_equal(predictions["true_class_index"].to_numpy(), saved["true_class_index"].to_numpy())
            or not np.array_equal(predictions["predicted_class_index"].to_numpy(),
                                  saved["predicted_class_index"].to_numpy())):
        raise AssertionError("Saved validation prediction order or labels differ from checkpoint inference.")
    tolerance = 1e-6
    metric_differences = {key: abs(metrics[key] - parent_metrics[key]) for key in
                          ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy")}
    if any(value > tolerance for value in metric_differences.values()):
        raise AssertionError(f"Day 4 checkpoint validation metrics differ: {metric_differences}; tolerance={tolerance}.")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if config.get("parent_checkpoint_sha256") and checkpoint_hash != config["parent_checkpoint_sha256"]:
        raise AssertionError("Parent checkpoint SHA256 differs from the configured identity.")
    return model, train, validation, mapping, validation_ds, metrics, {
        "parent_run_id": parent_id, "parent_checkpoint": str(checkpoint.resolve()),
        "parent_checkpoint_sha256": checkpoint_hash,
        "parent_macro_f1_recomputed": metrics["validation_macro_f1"],
        "parent_macro_f1_saved": parent_metrics["validation_macro_f1"],
        "parent_validation_metric_differences": metric_differences,
        "macro_f1_difference": metric_differences["validation_macro_f1"],
        "macro_f1_tolerance": tolerance,
        "split_file_sha256": {name: hashlib.sha256((splits_dir / name).read_bytes()).hexdigest()
                              for name in ("train.csv", "validation.csv", "class_mapping.json", "split_manifest.json")},
    }


def audit_day6_boundary(model: tf.keras.Model, config: dict, outputs_dir: Path, audit: dict) -> dict:
    """Confirm the saved Day 6 boundary and provenance before Day 7."""
    if config.get("first_trainable_block", 6) != 7 and config.get("sampling_policy") is None:
        return {}
    day6_id = config["day6_comparator_run_id"]
    _, saved, metrics = logged_run(outputs_dir, day6_id)
    observed = unfreeze_efficientnet_upper_stages(model, first_block=6)
    for key in ("backbone_trainable_layers", "backbone_frozen_layers",
                "backbone_trainable_parameters", "trainable_parameters", "total_parameters"):
        if observed[key] != saved.get(key) or observed[key] != metrics.get(key):
            raise AssertionError(f"Day 6 recorded {key} differs from the reconstructed boundary.")
    if (saved.get("parent_checkpoint_sha256") != audit["parent_checkpoint_sha256"]
            or metrics.get("parent_checkpoint_sha256") != audit["parent_checkpoint_sha256"]
            or saved.get("split_file_sha256") != audit["split_file_sha256"]):
        raise AssertionError("Day 6 checkpoint or split provenance differs from Day 7.")
    return {"day6_comparator_run_id": day6_id,
            "day6_boundary_confirmed": observed["boundary"],
            "day6_trainable_parameters": observed["trainable_parameters"],
            "day6_backbone_trainable_parameters": observed["backbone_trainable_parameters"]}


def prepare_moderate_sampling(train: pd.DataFrame, mapping: dict[str, int],
                              config: dict) -> tuple[list[np.ndarray], pd.DataFrame, pd.DataFrame]:
    """Precompute seeded row draws and auditable exposure counts for each epoch."""
    labels = train["class_index"].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=7)
    probabilities = moderate_sampling_probabilities(labels)
    if not np.isclose(probabilities.sum(), 1, atol=1e-12):
        raise AssertionError("Moderate sampling probabilities do not sum to one.")
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    table = pd.DataFrame({"class": names, "class_index": range(7),
                          "training_image_count": counts,
                          "natural_probability": counts / len(train),
                          "target_probability": probabilities})
    rng = np.random.default_rng(config["seed"])
    schedules = [sample_training_indices(labels, probabilities, len(train), rng)
                 for _ in range(config["max_epochs"])]
    rows = []
    for epoch, indices in enumerate(schedules, 1):
        if indices.shape != (len(train),) or np.any(indices < 0) or np.any(indices >= len(train)):
            raise AssertionError("Sampling produced an index outside the training split.")
        sampled_counts = np.bincount(labels[indices], minlength=7)
        for class_index, name in enumerate(names):
            class_indices = indices[labels[indices] == class_index]
            rows.append({"epoch": epoch, "class": name, "class_index": class_index,
                         "sampled_image_exposures": int(sampled_counts[class_index]),
                         "unique_training_images_exposed": int(np.unique(class_indices).size)})
    return schedules, table, pd.DataFrame(rows)


def smoke_oversampling(config: dict, splits_dir: Path, outputs_dir: Path) -> dict:
    """Audit the reusable control, sampling budget, weights, and callback order."""
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    model, train, validation, mapping, validation_ds, parent_metrics, audit = audit_fine_tune_parent(
        config, splits_dir, outputs_dir)
    manifest = json.loads((splits_dir / "split_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("group_column") != "lesion_id"
            or manifest.get("zero_lesion_overlap") is not True
            or manifest.get("all_seven_classes_each_split") is not True):
        raise AssertionError("Saved split manifest does not confirm seven classes and zero lesion overlap.")
    control = audit_day6_boundary(model, config, outputs_dir, audit)
    _, control_config, control_metrics = logged_run(outputs_dir, config["day6_comparator_run_id"])
    control_dir = outputs_dir / "runs" / config["day6_comparator_run_id"]
    control_history = pd.read_csv(control_dir / "history.csv")
    control_predictions = pd.read_csv(control_dir / "validation_predictions.csv")
    control_f1 = float(f1_score(control_predictions["true_class_index"],
                                control_predictions["predicted_class_index"],
                                labels=list(range(7)), average="macro", zero_division=0))
    if (control_config.get("class_mapping") != mapping
            or len(control_history) != control_metrics["actual_epochs"]
            or len(control_predictions) != len(validation)
            or not np.array_equal(control_predictions["image_id"].to_numpy(), validation["image_id"].to_numpy())
            or not np.array_equal(control_predictions["true_class_index"].to_numpy(),
                                  validation["class_index"].to_numpy())
            or abs(control_f1 - control_metrics["validation_macro_f1"]) > 1e-12):
        raise AssertionError("Saved Day 6 history, mapping, or full-validation predictions disagree.")
    schedules, table, exposures = prepare_moderate_sampling(train, mapping, config)
    labels = train["class_index"].to_numpy(dtype=np.int64)
    probabilities = table["target_probability"].to_numpy()
    empirical_indices = sample_training_indices(labels, probabilities, 100000,
                                                 np.random.default_rng(config["seed"] + 1))
    empirical = np.bincount(labels[empirical_indices], minlength=7) / len(empirical_indices)
    if np.max(np.abs(empirical - probabilities)) > 0.01:
        raise AssertionError("Empirical sampling frequencies differ from targets by more than 0.01.")
    table["empirical_probability_100000_draws"] = empirical
    batch_size = config["batch_size"]
    ordinary_batch_sizes = [min(batch_size, len(train) - start) for start in range(0, len(train), batch_size)]
    sampled_batch_sizes = [min(batch_size, len(indices) - start)
                           for indices in schedules for start in range(0, len(indices), batch_size)]
    if sampled_batch_sizes != ordinary_batch_sizes * config["max_epochs"]:
        raise AssertionError("Oversampled epoch batch sizes differ from ordinary sampling.")
    sampled_ds = make_image_dataset(train, training=True, batch_size=batch_size,
                                    architecture="efficientnetb0", sampled_indices_by_epoch=schedules)
    steps = len(ordinary_batch_sizes)
    if int(tf.data.experimental.cardinality(sampled_ds)) != steps * config["max_epochs"]:
        raise AssertionError("Oversampled dataset does not have the matched optimizer-update budget.")
    checkpoint = outputs_dir / "day8_disposable_check.keras"
    callbacks, _ = fine_tune_callbacks(validation_ds, validation["class_index"].to_numpy(dtype=int),
                                       checkpoint, config)
    expected_types = (ValidationMacroF1, tf.keras.callbacks.ModelCheckpoint,
                      tf.keras.callbacks.ReduceLROnPlateau, tf.keras.callbacks.EarlyStopping)
    if tuple(type(callback) for callback in callbacks) != expected_types:
        raise AssertionError("Macro F1 callback order differs from Day 6.")
    parent_path = outputs_dir / "runs" / config["parent_run_id"] / "best.keras"
    other = tf.keras.models.load_model(parent_path, compile=False)
    boundary = unfreeze_efficientnet_upper_stages(model, first_block=6)
    other_boundary = unfreeze_efficientnet_upper_stages(other, first_block=6)
    if boundary != other_boundary or boundary["trainable_parameters"] != control["day6_trainable_parameters"]:
        raise AssertionError("Ordinary and sampled branches have different trainability.")
    if any(not np.array_equal(left.numpy(), right.numpy()) for left, right in zip(model.weights, other.weights)):
        raise AssertionError("The two branches do not start from identical parent weights.")
    result = {**audit, **control, **boundary, "smoke_only": True,
              "sampling_policy": config["sampling_policy"], "training_images": len(train),
              "validation_images": len(validation), "batch_size": batch_size,
              "steps_per_epoch": steps, "last_batch_size": ordinary_batch_sizes[-1],
              "max_epochs": config["max_epochs"], "maximum_optimizer_updates": steps * config["max_epochs"],
              "sampling_seed": config["seed"], "empirical_draws": len(empirical_indices),
              "max_empirical_target_difference": float(np.max(np.abs(empirical - probabilities))),
              "empirical_tolerance": 0.01, "same_initial_weights": True,
              "callback_order": [type(callback).__name__ for callback in callbacks],
              "day6_control_reused": True, "day6_history_epochs": len(control_history),
              "day6_macro_f1_recomputed_from_saved_predictions": control_f1,
              "day6_steps_per_epoch_inferred_from_finite_unbatched_training_rows": steps,
              "parent_validation_macro_f1": parent_metrics["validation_macro_f1"]}
    outputs_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(outputs_dir / "day8_sampling_probabilities.csv", index=False)
    save_json(result, outputs_dir / "day8_preflight.json")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("backbone_trainable_layers", "backbone_frozen_layers")}, indent=2))
    return result


def smoke_fine_tune(config: dict, splits_dir: Path, outputs_dir: Path) -> dict:
    """Verify gradients, frozen state, BN state, and reload on a disposable model."""
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    model, train, validation, mapping, validation_ds, parent_metrics, audit = audit_fine_tune_parent(
        config, splits_dir, outputs_dir)
    comparator_audit = audit_day6_boundary(model, config, outputs_dir, audit)
    images, _ = next(iter(validation_ds.take(1)))
    before_trainability = model(images, training=False).numpy()
    boundary = unfreeze_efficientnet_upper_stages(model, first_block=config.get("first_trainable_block", 6))
    trainability_difference = float(np.max(np.abs(before_trainability - model(images, training=False).numpy())))
    if trainability_difference > 1e-7:
        raise AssertionError(f"Changing trainability changed inference by {trainability_difference}.")
    compile_fine_tune(model, config)
    backbone = model.get_layer("efficientnetb0")
    upper = backbone.get_layer("block7a_expand_conv").kernel
    top = backbone.get_layer("top_conv").kernel
    earlier = backbone.get_layer("block6a_expand_conv" if config.get("first_trainable_block", 6) == 7
                                 else "block5a_expand_conv").kernel
    head = model.get_layer("dense_128").kernel
    bn_weights = [weight for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
                  for weight in (layer.moving_mean, layer.moving_variance)]
    before = [weight.numpy().copy() for weight in (upper, top, earlier, head, *bn_weights)]
    train_ds = make_image_dataset(train.head(2), training=True, batch_size=2,
                                  architecture="efficientnetb0", augment=None, seed=config["seed"])
    images, labels = next(iter(train_ds))
    loss = float(model.train_on_batch(images, labels)[0])
    after = [weight.numpy() for weight in (upper, top, earlier, head, *bn_weights)]
    differences = [float(np.max(np.abs(a - b))) for a, b in zip(after, before)]
    if not np.isfinite(loss) or differences[0] <= 0 or differences[1] <= 0 or differences[3] <= 0 or any(d != 0 for d in
                                                                                    (differences[2], *differences[4:])):
        raise AssertionError(f"Fine-tuning weight isolation failed: {differences}")
    with tempfile.TemporaryDirectory(prefix="ham_fine_tune_smoke_") as temporary:
        path = Path(temporary) / "disposable.keras"
        model.save(path)
        restored = tf.keras.models.load_model(path, compile=False)
        difference = float(np.max(np.abs(model(images, training=False).numpy()
                                         - restored(images, training=False).numpy())))
    if difference > 1e-5:
        raise AssertionError(f"Reload changed inference outputs by {difference}.")
    result = {**audit, **comparator_audit, **boundary, "smoke_only": True, "one_batch_loss": loss,
              "trainability_only_max_probability_difference": trainability_difference,
              "weight_max_changes_block7_top_frozen_head": differences[:4],
              "batch_norm_moving_stat_max_change": max(differences[4:]),
              "batch_norm_moving_stat_tensors_checked": len(bn_weights),
              "reload_max_probability_difference": difference,
              "augmentation": False, "class_weights": False,
              "validation_images": len(validation), "training_images": len(train),
              "parent_validation_macro_f1": parent_metrics["validation_macro_f1"]}
    save_json(result, outputs_dir / ("day7_preflight.json" if config.get("first_trainable_block", 6) == 7
                                      else "day6_preflight.json"))
    print(json.dumps(result, indent=2))
    return result


def train_fine_tune(config: dict, splits_dir: Path, outputs_dir: Path) -> tuple[str, dict]:
    """Fine-tune from a freshly reloaded Day 4 checkpoint and compare on validation."""
    tf.keras.backend.clear_session()
    set_seed(config["seed"])
    model, train, validation, mapping, validation_ds, parent_metrics, audit = audit_fine_tune_parent(
        config, splits_dir, outputs_dir)
    comparator_audit = audit_day6_boundary(model, config, outputs_dir, audit)
    sample, _ = next(iter(validation_ds.take(1)))
    original_probabilities = model(sample, training=False).numpy()
    boundary = unfreeze_efficientnet_upper_stages(model, first_block=config.get("first_trainable_block", 6))
    if not np.allclose(original_probabilities, model(sample, training=False).numpy(), atol=1e-7, rtol=0):
        raise AssertionError("Changing trainability changed parent inference.")
    compile_fine_tune(model, config)
    day8 = config.get("sampling_policy") == "sqrt_training_class_count"
    if day8:
        preflight_path = outputs_dir / "day8_preflight.json"
        if not preflight_path.is_file():
            raise ValueError("Run the Day 8 --smoke check before full training.")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if (preflight.get("parent_checkpoint_sha256") != audit["parent_checkpoint_sha256"]
                or preflight.get("split_file_sha256") != audit["split_file_sha256"]
                or preflight.get("day6_comparator_run_id") != config["day6_comparator_run_id"]):
            raise AssertionError("Day 8 preflight provenance differs from this training run.")
        schedules, sampling_table, exposure_table = prepare_moderate_sampling(train, mapping, config)
        train_ds = make_image_dataset(train, training=True, batch_size=config["batch_size"],
                                      architecture="efficientnetb0", sampled_indices_by_epoch=schedules)
    else:
        train_ds = make_image_dataset(train, training=True, batch_size=config["batch_size"],
                                      architecture="efficientnetb0", augment=None, seed=config["seed"])
    day7 = config.get("first_trainable_block", 6) == 7
    run_id, run_dir = create_run_directory(outputs_dir, "efficientnetb0_moderate_oversampling" if day8
                                            else "efficientnetb0_shallow_fine_tune" if day7
                                            else "efficientnetb0_fine_tune")
    checkpoint = run_dir / "best.keras"
    save_json({**config, **audit, **comparator_audit, **boundary, "run_id": run_id,
               "splits_dir": str(splits_dir.resolve()), "class_mapping": mapping,
               "preprocessing": "original RGB 0-255; EfficientNet internal rescaling",
               "backbone_call_training": False,
               "steps_per_epoch": int(np.ceil(len(train) / config["batch_size"])),
               "learning_rate_history_convention": "learning_rate_used is sampled at epoch start; learning_rate is sampled before end-of-epoch scheduling callbacks",
               "training_metric_distribution_note": "Training accuracy and loss are measured on this branch's sampled images and are not directly comparable to the natural-frequency control.",
               "loss_note": "Logged total loss includes hidden-layer L2 regularization."}, run_dir / "config.json")
    if day8:
        sampling_table.to_csv(run_dir / "sampling_probabilities.csv", index=False)
    callbacks, f1_callback = fine_tune_callbacks(
        validation_ds, validation["class_index"].to_numpy(dtype=int), checkpoint, config)
    started = time.perf_counter()
    history = model.fit(train_ds, validation_data=validation_ds, epochs=config["max_epochs"],
                        steps_per_epoch=int(np.ceil(len(train) / config["batch_size"])) if day8 else None,
                        callbacks=callbacks, verbose=2)
    duration = time.perf_counter() - started
    frame = pd.DataFrame(history.history)
    frame.insert(0, "epoch", np.arange(1, len(f1_callback.scores) + 1))
    frame["val_macro_f1"] = f1_callback.scores
    frame["learning_rate"] = f1_callback.learning_rates
    frame["learning_rate_used"] = f1_callback.learning_rates_used
    updates_per_epoch = int(np.ceil(len(train) / config["batch_size"]))
    expected_updates = len(frame) * updates_per_epoch
    actual_updates = int(tf.keras.backend.get_value(model.optimizer.iterations))
    if actual_updates != expected_updates:
        raise AssertionError(f"Expected {expected_updates} optimizer updates; observed {actual_updates}.")
    if day8:
        exposure_table.loc[exposure_table.epoch <= len(frame)].to_csv(
            run_dir / "sampled_class_exposures_by_epoch.csv", index=False)
        frame["optimizer_updates_this_epoch"] = updates_per_epoch
        frame["sampled_image_exposures_this_epoch"] = len(train)
    frame.to_csv(run_dir / "history.csv", index=False)
    plot_learning_curves(frame, run_dir / "learning_curves.png")
    best_epoch = int(np.argmax(f1_callback.scores) + 1)
    if not checkpoint.is_file():
        raise AssertionError("Fine-tuning saved no best checkpoint.")
    best = tf.keras.models.load_model(checkpoint, compile=False)
    predictions, scores = evaluate_validation(validation, best.predict(validation_ds, verbose=0), mapping)
    if not np.isclose(scores["validation_macro_f1"], f1_callback.scores[best_epoch - 1], atol=1e-7):
        raise AssertionError("Best fine-tuning checkpoint does not match epoch F1.")
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    matrix = confusion_matrix(predictions.true_class_index, predictions.predicted_class_index,
                              labels=list(range(7)))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(run_dir / "confusion_matrix.csv")
    plot_confusion_matrix(matrix, names, run_dir / "confusion_matrix.png")
    pd.DataFrame(matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1),
                 index=names, columns=names).to_csv(run_dir / "confusion_matrix_normalized.csv")
    parent_saved = logged_run(outputs_dir, config["parent_run_id"])[2]
    comparison = {key: {"frozen": parent_metrics[key], "fine_tuned": scores[key],
                        "change": scores[key] - parent_metrics[key]}
                  for key in ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy")}
    per_class = pd.DataFrame([{"class": name, "support": scores["per_class"][name]["support"],
                               **{f"frozen_{metric}": parent_saved["per_class"][name][metric]
                                  for metric in ("precision", "recall", "f1")},
                               **{f"fine_tuned_{metric}": scores["per_class"][name][metric]
                                  for metric in ("precision", "recall", "f1")},
                               **{f"change_{metric}": scores["per_class"][name][metric]
                                  - parent_saved["per_class"][name][metric]
                                  for metric in ("precision", "recall", "f1")}} for name in names])
    per_class.to_csv(run_dir / "per_class_comparison.csv", index=False)
    reductions = [int(i + 1) for i in range(1, len(f1_callback.learning_rates))
                  if f1_callback.learning_rates[i] < f1_callback.learning_rates[i - 1]]
    retained = run_id if scores["validation_macro_f1"] > parent_metrics["validation_macro_f1"] else config["parent_run_id"]
    if day8:
        control_id = config["day6_comparator_run_id"]
        _, control_config, control_metrics = logged_run(outputs_dir, control_id)
        retained = run_id if scores["validation_macro_f1"] > control_metrics["validation_macro_f1"] else control_id
        sampling_comparison = pd.DataFrame([
            {"branch": "ordinary_day6", "run_id": control_id,
             **{key: control_metrics[key] for key in ("validation_accuracy", "validation_macro_f1",
                                                        "validation_balanced_accuracy", "actual_epochs",
                                                        "best_epoch", "training_seconds")},
             "trainable_parameters": control_config["trainable_parameters"],
             "optimizer_updates": control_metrics["actual_epochs"] * updates_per_epoch},
            {"branch": "moderate_oversampling_day8", "run_id": run_id,
             **scores, "actual_epochs": len(frame), "best_epoch": int(np.argmax(f1_callback.scores) + 1),
             "training_seconds": duration, "trainable_parameters": boundary["trainable_parameters"],
             "optimizer_updates": actual_updates},
        ])
        sampling_comparison = sampling_comparison.drop(columns="per_class", errors="ignore")
        sampling_comparison.to_csv(run_dir / "sampling_validation_comparison.csv", index=False)
        pd.DataFrame([{"class": name, "support": scores["per_class"][name]["support"],
                       **{f"{branch}_{metric}": source["per_class"][name][metric]
                          for branch, source in (("day6", control_metrics), ("day8", scores))
                          for metric in ("precision", "recall", "f1")}}
                      for name in names]).to_csv(run_dir / "sampling_per_class_comparison.csv", index=False)
    if day7:
        day6_id = config["day6_comparator_run_id"]
        _, day6_config, day6_metrics = logged_run(outputs_dir, day6_id)
        retained = run_id if scores["validation_macro_f1"] > day6_metrics["validation_macro_f1"] else day6_id
        rows = []
        for label, identity, source, parameters in (
            ("Day 4 frozen parent", config["parent_run_id"], parent_saved,
             logged_run(outputs_dir, config["parent_run_id"])[2]["trainable_parameters"]),
            ("Day 6 block6+block7+top_conv", day6_id, day6_metrics,
             day6_config["trainable_parameters"]),
            ("Day 7 block7+top_conv", run_id, {**scores, "best_epoch": best_epoch,
                                            "actual_epochs": len(frame), "training_seconds": duration},
             boundary["trainable_parameters"]),
        ):
            rows.append({"experiment": label, "run_id": identity,
                         **{key: source[key] for key in ("validation_accuracy", "validation_macro_f1",
                                                          "validation_balanced_accuracy", "best_epoch",
                                                          "actual_epochs", "training_seconds")},
                         "trainable_parameters": parameters,
                         "macro_f1_change_vs_day6": source["validation_macro_f1"] - day6_metrics["validation_macro_f1"],
                         "macro_f1_change_vs_frozen": source["validation_macro_f1"] - parent_saved["validation_macro_f1"]})
        pd.DataFrame(rows).to_csv(run_dir / "three_way_validation_comparison.csv", index=False)
        day7_per_class = pd.DataFrame([
            {"class": name, "support": scores["per_class"][name]["support"],
             **{f"{stage}_{metric}": source["per_class"][name][metric]
                for stage, source in (("frozen", parent_saved), ("day6", day6_metrics), ("day7", scores))
                for metric in ("precision", "recall", "f1")}}
            for name in names])
        day7_per_class.to_csv(run_dir / "three_way_per_class_comparison.csv", index=False)
        noteworthy = {
            name: {
                "support": scores["per_class"][name]["support"],
                "day6_precision": day6_metrics["per_class"][name]["precision"],
                "day7_precision": scores["per_class"][name]["precision"],
                "day6_recall": day6_metrics["per_class"][name]["recall"],
                "day7_recall": scores["per_class"][name]["recall"],
            }
            for name in ("mel", "df", "bkl", "bcc", "vasc")
        }
    lowest_recall = sorted(names, key=lambda name: scores["per_class"][name]["recall"])[:2]
    interpretation = (
        f"Validation macro F1 changed by {comparison['validation_macro_f1']['change']:+.6f} "
        f"in this single run. Lowest fine-tuned recall: {', '.join(lowest_recall)}. "
        f"Training loss moved from {frame['loss'].iloc[0]:.4f} to {frame['loss'].iloc[-1]:.4f}; "
        f"validation loss moved from {frame['val_loss'].iloc[0]:.4f} to {frame['val_loss'].iloc[-1]:.4f}. "
        "These are observed validation results, not a causal estimate of fine-tuning benefit."
    )
    if day8:
        interpretation = (
            f"Moderate oversampling changed validation macro F1 by "
            f"{scores['validation_macro_f1'] - control_metrics['validation_macro_f1']:+.6f} "
            "relative to the saved ordinary-sampling Day 6 run. Training metrics use different "
            "sampling distributions and are not directly comparable. This single run does not "
            "establish a reliable causal benefit."
        )
    runtime = runtime_info(config["batch_size"])
    metrics = {"run_id": run_id, "parent_run_id": config["parent_run_id"], **scores,
               "actual_epochs": len(frame), "best_epoch": best_epoch,
               "learning_rate_reduction_epochs": reductions, "training_seconds": duration,
               "best_checkpoint": str(checkpoint.resolve()), "retained_candidate_run_id": retained,
               "comparison": comparison, "interpretation": interpretation,
               "logged_loss_includes_l2": True,
               "framework_versions": {key: runtime[key] for key in
                                      ("python", "tensorflow", "keras", "pandas", "numpy", "scikit_learn")},
               "detected_training_device": "GPU" if runtime["gpus"] else "CPU",
               "visible_gpus": runtime["gpus"], **boundary, **audit, **comparator_audit}
    if day8:
        metrics.update({"sampling_policy": config["sampling_policy"],
                        "day6_control_run_id": control_id,
                        "macro_f1_change_vs_day6": scores["validation_macro_f1"] - control_metrics["validation_macro_f1"],
                        "steps_per_epoch": updates_per_epoch, "optimizer_updates": actual_updates,
                        "sampled_image_exposures": len(frame) * len(train),
                        "training_metric_distribution_note": "Training loss and accuracy use the oversampled distribution; do not compare directly to ordinary sampling."})
    if day7:
        metrics["macro_f1_change_vs_day6"] = scores["validation_macro_f1"] - day6_metrics["validation_macro_f1"]
        metrics["macro_f1_change_vs_frozen"] = scores["validation_macro_f1"] - parent_saved["validation_macro_f1"]
        metrics["learning_rate_history_convention"] = "learning_rate_used is sampled at epoch start; learning_rate is sampled before scheduling callbacks"
        metrics["per_class_points_for_review"] = noteworthy
        metrics["vasc_caution"] = "Validation support is 20; interpret its change cautiously."
    save_json(metrics, run_dir / "metrics.json")
    save_json({"retained_candidate_run_id": retained, "frozen_parent_run_id": config["parent_run_id"],
               "fine_tuned_run_id": run_id, "comparison": comparison,
               "day6_comparator_run_id": config.get("day6_comparator_run_id"),
               "macro_f1_change_vs_day6": metrics.get("macro_f1_change_vs_day6"),
               "per_class_points_for_review": noteworthy if day7 else None,
               "best_epoch": best_epoch, "actual_epochs": len(frame),
               "learning_rate_reduction_epochs": reductions,
               "interpretation": interpretation}, run_dir / "decision.json")
    append_experiment_log(outputs_dir, {
        "run_id": run_id, "comparison_group": config["comparison_group"],
        "parent_run_id": config["parent_run_id"],
        "comparator_run_id": config["day6_comparator_run_id"] if day7 or day8 else config["parent_run_id"],
        "architecture": "efficientnetb0", "augmentation": False, "class_weights": False,
        "dropout": 0.0, "l2": config["l2"], "optimizer": "adam",
        "learning_rate": config["learning_rate"], "batch_size": config["batch_size"],
        "seed": config["seed"], "actual_epochs": len(frame), "best_epoch": best_epoch,
        **{key: scores[key] for key in comparison}, "training_seconds": duration,
        "total_parameters": boundary["total_parameters"],
        "trainable_parameters": boundary["trainable_parameters"],
        "optimizer_updates": actual_updates, "sampling_policy": config.get("sampling_policy", "natural"),
        "detected_training_device": metrics["detected_training_device"],
        "run_dir": str(run_dir), "logged_loss_includes_l2": True, "run_status": "valid"})
    print(json.dumps({"run_id": run_id, "decision": retained, "comparison": comparison,
                      "macro_f1_change_vs_day6": metrics.get("macro_f1_change_vs_day6")}, indent=2))
    return run_id, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    architecture_group = parser.add_mutually_exclusive_group(required=True)
    architecture_group.add_argument("--architecture", choices=ARCHITECTURES)
    architecture_group.add_argument("--all", action="store_true", help="Run all three architectures sequentially")
    architecture_group.add_argument("--stage", choices=HEAD_EXPERIMENT_STAGES, help="Run one sequential head experiment")
    architecture_group.add_argument("--optimizer-choice", choices=OPTIMIZER_NAMES,
                                    help="Run one five-epoch frozen optimizer comparison")
    architecture_group.add_argument("--fine-tune", action="store_true",
                                    help="Audit and partially fine-tune selected Day 4 EfficientNetB0")
    architecture_group.add_argument("--fine-tune-depth", action="store_true",
                                    help="Compare block7-only fine-tuning against the Day 6 candidate")
    architecture_group.add_argument("--moderate-oversampling", action="store_true",
                                    help="Compare sqrt-count training sampling against the Day 6 control")
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
    if args.fine_tune or args.fine_tune_depth or args.moderate_oversampling:
        if args.batch_size or args.parent_run_id:
            parser.error("Fine-tuning uses the fixed batch size and selected Day 4 parent.")
        config_path = ((OVERSAMPLING_CONFIG if args.moderate_oversampling else
                        DEPTH_CONFIG if args.fine_tune_depth else FINE_TUNE_CONFIG)
                       if args.config == DEFAULT_CONFIG else args.config)
        config = (load_oversampling_config(config_path, args.outputs_dir) if args.moderate_oversampling
                  else load_fine_tune_config(config_path, args.outputs_dir))
        if args.fine_tune_depth and config.get("first_trainable_block") != 7:
            parser.error("--fine-tune-depth requires the Day 7 block7 config.")
        if args.fine_tune and config.get("first_trainable_block", 6) != 6:
            parser.error("--fine-tune requires the Day 6 block6 config.")
        if args.smoke:
            if args.moderate_oversampling:
                smoke_oversampling(config, args.splits_dir, args.outputs_dir)
            else:
                smoke_fine_tune(config, args.splits_dir, args.outputs_dir)
        else:
            train_fine_tune(config, args.splits_dir, args.outputs_dir)
        return
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
