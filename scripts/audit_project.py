"""Read-only model audit; reads test split metadata but never test images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_pipeline import (decode_resize, make_image_dataset, moderate_sampling_probabilities,
                               preprocess_for)
from src.evaluate import evaluate_validation
from src.models import unfreeze_efficientnet_upper_stages
from src.prepare_data import verify_splits
from src.train import compile_fine_tune, fine_tune_callbacks
from src.utils import runtime_info, save_json, set_seed


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "day1_artifacts"
OUTPUTS = ROOT / "outputs"
RUN_IDS = {
    "day4": "20261121T134510640652Z_t3_e0_l_e8ee6a73",
    "day6": "20261121T154533949681Z_efficientnetb0_fine_tune_4f313742",
    "day7": "20261122T001239643036Z_efficientnetb0_shallow_fine_tune_05418693",
    "day8": "20261122T010137749622Z_efficientnetb0_moderate_oversampling_2e68db06",
    "day9": "20261122T013218161722Z_efficientnetb0_mild_augmentation_1a1dec41",
}
METRIC_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_saved_predictions(run_dir: Path, validation: pd.DataFrame,
                            mapping: dict[str, int], saved_metrics: dict) -> dict:
    """Recompute unweighted metrics from saved ordered full-precision probabilities."""
    saved = pd.read_csv(run_dir / "validation_predictions.csv")
    names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    columns = [f"prob_{name}" for name in names]
    require(all(column in saved for column in columns), "Saved class probability columns are incomplete.")
    require(len(saved) == len(validation), "Validation predictions dropped or duplicated rows.")
    for column, source in (("image_id", "image_id"), ("lesion_id", "lesion_id"),
                           ("true_dx", "dx"), ("true_class_index", "class_index")):
        require(np.array_equal(saved[column], validation[source]), f"Saved {column} is misaligned.")
    predictions, scores = evaluate_validation(validation, saved[columns].to_numpy(), mapping)
    require(np.array_equal(predictions.predicted_class_index, saved.predicted_class_index),
            "Saved predictions differ from argmax of probabilities.")
    for key in ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy"):
        require(abs(scores[key] - saved_metrics[key]) <= METRIC_TOLERANCE,
                f"Saved {key} differs from recomputed value.")
    for name in names:
        for key in ("precision", "recall", "f1", "support"):
            require(abs(scores["per_class"][name][key] - saved_metrics["per_class"][name][key])
                    <= METRIC_TOLERANCE, f"Saved {name} {key} differs.")
    return {"scores": scores, "saved_probabilities": saved[columns].to_numpy()}


def check_checkpoint(run_dir: Path, validation_ds: tf.data.Dataset, validation: pd.DataFrame,
                     mapping: dict[str, int], saved_probabilities: np.ndarray) -> dict:
    """Reload best checkpoint once and compare inference with saved predictions."""
    model = tf.keras.models.load_model(run_dir / "best.keras", compile=False)
    actual = model.predict(validation_ds, verbose=0)
    predictions, metrics = evaluate_validation(validation, actual, mapping)
    difference = float(np.max(np.abs(actual - saved_probabilities)))
    require(difference <= PROBABILITY_TOLERANCE, f"Reloaded probabilities differ: {difference}.")
    return {"model": model, "metrics": metrics, "max_saved_probability_difference": difference,
            "prediction_count": len(predictions)}


def check_gradient_update(parent_path: Path, config: dict, train: pd.DataFrame) -> dict:
    """Apply one update to a disposable loaded parent, then discard it."""
    model = tf.keras.models.load_model(parent_path, compile=False)
    boundary = unfreeze_efficientnet_upper_stages(model, first_block=6)
    compile_fine_tune(model, config)
    backbone = model.get_layer("efficientnetb0")
    upper = backbone.get_layer("block7a_expand_conv").kernel
    top = backbone.get_layer("top_conv").kernel
    frozen = backbone.get_layer("block5a_expand_conv").kernel
    head = model.get_layer("dense_128").kernel
    bn_stats = [weight for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)
                for weight in (layer.moving_mean, layer.moving_variance)]
    watched = [upper, top, frozen, head, *bn_stats]
    before = [weight.numpy().copy() for weight in watched]
    images, labels = next(iter(make_image_dataset(train.head(2), training=True, batch_size=2,
                                                   architecture="efficientnetb0", seed=config["seed"])))
    with tf.GradientTape() as tape:
        probabilities = model(images, training=True)
        cross_entropy = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities))
        regularization = tf.add_n(model.losses)
        loss = cross_entropy + regularization
    variables = model.trainable_weights
    gradients = tape.gradient(loss, variables)
    require(all(gradient is not None and bool(tf.reduce_all(tf.math.is_finite(gradient)))
                for gradient in gradients), "A trainable gradient is absent or nonfinite.")
    expected_l2 = tf.keras.regularizers.L2(1e-4)(head)
    require(abs(float(regularization) - float(expected_l2)) < 1e-5,
            "Regularization differs from head kernel L2.")
    require(all(any(variable is target for variable in variables) for target in (upper, top, head)),
            "Expected trainable convolution/head kernel missing from optimizer variables.")
    require(not any(variable is frozen for variable in variables), "Earlier convolution is trainable.")
    model.optimizer.apply_gradients(zip(gradients, variables))
    differences = [float(np.max(np.abs(weight.numpy() - old))) for weight, old in zip(watched, before)]
    require(all(value > 0 for value in (differences[0], differences[1], differences[3])),
            "Upper convolution or head kernel did not update.")
    require(all(value == 0 for value in (differences[2], *differences[4:])),
            "Frozen weight or BatchNorm moving statistic changed.")
    require(int(model.optimizer.iterations.numpy()) == 1, "Disposable optimizer did not make exactly one update.")
    return {"trainable_parameters": boundary["trainable_parameters"],
            "finite_gradient_count": len(gradients), "cross_entropy": float(cross_entropy),
            "l2_contribution": float(regularization), "total_loss": float(loss),
            "max_changes_block7_top_frozen_head": differences[:4],
            "batch_norm_stat_tensors_checked": len(bn_stats),
            "max_batch_norm_stat_change": max(differences[4:]), "optimizer_updates": 1}


def main() -> None:
    set_seed(42)
    mapping = json.loads((SPLITS / "class_mapping.json").read_text(encoding="utf-8"))
    parts = {name: pd.read_csv(SPLITS / f"{name}.csv") for name in
             ("train", "validation", "test")}
    discovery = json.loads((SPLITS / "discovered_paths.json").read_text(encoding="utf-8"))
    source = pd.read_csv(discovery["metadata"], usecols=["image_id", "dx"])
    counts = verify_splits(parts, source)
    require(counts.images.tolist() == [7153, 1431, 1431], "Saved image counts changed.")
    require(counts.lesions.tolist() == [5334, 1068, 1068], "Saved lesion counts changed.")
    require(sorted(mapping.values()) == list(range(7)), "Class mapping differs from 0..6.")
    manifest = json.loads((SPLITS / "split_manifest.json").read_text(encoding="utf-8"))
    require((manifest["seed"], manifest["validation_fold"], manifest["test_fold"]) == (42, 1, 3),
            "Saved fold selection changed.")
    hashes = {name: file_hash(SPLITS / name) for name in
              ("train.csv", "validation.csv", "class_mapping.json", "split_manifest.json")}
    train, validation = parts["train"], parts["validation"]
    train_image, train_label = next(iter(make_image_dataset(train.head(1), batch_size=1)))
    valid_image, valid_label = next(iter(make_image_dataset(validation.head(1), batch_size=1)))
    image_checks = {}
    for label, images, labels in (("train", train_image, train_label),
                                   ("validation", valid_image, valid_label)):
        values = images.numpy()
        require(values.shape == (1, 224, 224, 3) and values.dtype == np.float32
                and np.isfinite(values).all() and values.min() >= 0 and values.max() <= 255,
                f"{label} tensor shape, dtype or range is invalid.")
        image_checks[label] = {"shape": list(values.shape), "dtype": str(values.dtype),
                               "min": float(values.min()), "max": float(values.max()),
                               "class_index": int(labels[0])}
    require(np.array_equal(valid_image.numpy(), preprocess_for("efficientnetb0")(valid_image).numpy()),
            "EfficientNet preprocessing is not the expected identity transform.")
    validation_ds = make_image_dataset(validation, batch_size=32, architecture="efficientnetb0")
    runs = {}
    for day, run_id in RUN_IDS.items():
        run_dir = OUTPUTS / "runs" / run_id
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        predictions = check_saved_predictions(run_dir, validation, mapping, metrics)
        history = pd.read_csv(run_dir / "history.csv")
        require(len(history) == metrics["actual_epochs"] and
                int(history.loc[history.val_macro_f1.idxmax(), "epoch"]) == metrics["best_epoch"],
                f"{day} best epoch or history is inconsistent.")
        require(abs(float(history.val_macro_f1.max()) - metrics["validation_macro_f1"]) <= 1e-7,
                f"{day} best checkpoint F1 differs from history.")
        if day != "day4":
            require(config["parent_run_id"] == RUN_IDS["day4"] and config["seed"] == 42
                    and config["batch_size"] == 32 and config["class_mapping"] == mapping
                    and config["split_file_sha256"] == hashes,
                    f"{day} provenance differs from the selected parent/splits.")
        runs[day] = {"run_id": run_id, "checkpoint": str((run_dir / "best.keras").resolve()),
                     "checkpoint_sha256": file_hash(run_dir / "best.keras"),
                     "parent_checkpoint_sha256": config.get("parent_checkpoint_sha256"),
                     "validation_metrics_recomputed": predictions["scores"],
                     "actual_epochs": metrics["actual_epochs"], "best_epoch": metrics["best_epoch"],
                     "training_seconds": metrics["training_seconds"],
                     "trainable_parameters": metrics["trainable_parameters"],
                     "optimizer_updates_recorded": metrics.get("optimizer_updates"),
                     "optimizer_updates_inferred": metrics["actual_epochs"] * 224}
        if day in ("day4", "day6"):
            checkpoint = check_checkpoint(run_dir, validation_ds, validation, mapping,
                                          predictions["saved_probabilities"])
            runs[day]["reload_max_saved_probability_difference"] = checkpoint["max_saved_probability_difference"]
            for key in ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy"):
                require(abs(checkpoint["metrics"][key] - metrics[key]) <= 1e-7,
                        f"{day} reloaded checkpoint {key} differs.")
            del checkpoint
    require(all(runs[day]["parent_checkpoint_sha256"] == runs["day4"]["checkpoint_sha256"]
                for day in ("day6", "day7", "day8", "day9")),
            "Fine-tuning runs do not share the Day 4 checkpoint identity.")
    day6_config = json.loads((OUTPUTS / "runs" / RUN_IDS["day6"] / "config.json").read_text(encoding="utf-8"))
    gradient = check_gradient_update(Path(runs["day4"]["checkpoint"]), day6_config, train)
    require(gradient["trainable_parameters"] == runs["day6"]["trainable_parameters"],
            "Disposable Day 6 boundary differs from saved count.")
    model7 = tf.keras.models.load_model(runs["day4"]["checkpoint"], compile=False)
    boundary7 = unfreeze_efficientnet_upper_stages(model7, first_block=7)
    require(boundary7["trainable_parameters"] == runs["day7"]["trainable_parameters"],
            "Day 7 boundary count differs.")
    callbacks, _ = fine_tune_callbacks(validation_ds, validation.class_index.to_numpy(),
                                       OUTPUTS / "day10_unused.keras", day6_config)
    require([type(callback).__name__ for callback in callbacks] ==
            ["ValidationMacroF1", "ModelCheckpoint", "ReduceLROnPlateau", "EarlyStopping"],
            "Fine-tuning callback order differs.")
    sampling = pd.read_csv(OUTPUTS / "runs" / RUN_IDS["day8"] / "sampling_probabilities.csv")
    expected = moderate_sampling_probabilities(train.class_index.to_numpy())
    require(np.allclose(sampling.target_probability, expected, atol=1e-12),
            "Day 8 sampling table differs from training-only sqrt counts.")
    result = {"verdict": "READY", "metric_tolerance": METRIC_TOLERANCE,
              "reload_metric_tolerance": 1e-7, "probability_tolerance": PROBABILITY_TOLERANCE,
              "split_manifest": manifest, "split_counts": counts.to_dict(orient="records"),
              "split_file_sha256": hashes, "class_mapping": mapping,
              "representative_image_tensors": image_checks,
              "runs": runs, "disposable_gradient_update": gradient,
              "day7_trainable_parameters_reconstructed": boundary7["trainable_parameters"],
              "callback_order": [type(callback).__name__ for callback in callbacks],
              "day8_training_only_sampling_probabilities_verified": True,
              "runtime": runtime_info(32),
              "evidence_gaps": ["Day 6 and Day 7 historical optimizer iteration counters were not saved; updates are inferred from epochs and finite dataset cardinality.",
                                "Individual training and validation timing was not logged; only whole-fit runtime is available."]}
    save_json(result, OUTPUTS / "day10_audit.json")
    print(json.dumps({"verdict": result["verdict"],
                      "day4_sha256": runs["day4"]["checkpoint_sha256"],
                      "day6_sha256": runs["day6"]["checkpoint_sha256"],
                      "day4_f1": runs["day4"]["validation_metrics_recomputed"]["validation_macro_f1"],
                      "day6_f1": runs["day6"]["validation_metrics_recomputed"]["validation_macro_f1"],
                      "gradient_check": gradient, "audit_record": str(OUTPUTS / "day10_audit.json")}, indent=2))


if __name__ == "__main__":
    main()
