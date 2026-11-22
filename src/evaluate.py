"""Saved-split classification metrics and staged Day 13 final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support

from .utils import save_json


EVALUATION_RUNS = {
    "baseline": ("20261121T040611579432Z_efficientnetb0_edeb0f80", "fddc7d7888f23ec04aebfef32e516c88dd53e3435e4d8f059028a4fa00f721d8"),
    "day6": ("20261121T154533949681Z_efficientnetb0_fine_tune_4f313742", "daf555e7f501238c36a534682772f3741263ad794a5bf40893b704a9e012c176"),
    "day11": ("20261122T030007240028Z_efficientnetb0_broader_fine_tune_be751263", "997202ec9c292c86934958de5a744585367026a12143464ea0e475c249ea715a"),
}
PROVENANCE_SPLIT_FILES = ("train.csv", "validation.csv", "class_mapping.json", "split_manifest.json")
PROBABILITY_TOLERANCE = 1e-5  # CSV float32 softmax rounding; also require each value in [-atol, 1+atol].
METRIC_RECONCILIATION_TOLERANCE = 1e-12


def evaluate_validation(
    frame: pd.DataFrame, probabilities: np.ndarray, class_mapping: dict[str, int]
) -> tuple[pd.DataFrame, dict]:
    """Match one probability row to each validation CSV row, in saved order."""
    labels = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
    true = frame["class_index"].to_numpy(dtype=int)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(frame), len(labels)):
        raise ValueError(f"Prediction shape {probabilities.shape} does not match {(len(frame), len(labels))}.")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-3):
        raise ValueError("Validation probabilities are nonfinite or do not sum to one.")
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=list(range(len(labels))), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index, name in enumerate(labels)
    }
    metrics = {
        "validation_accuracy": float(accuracy_score(true, predicted)),
        "validation_macro_f1": float(f1.mean()),
        "validation_balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "per_class": per_class,
    }
    predictions = pd.DataFrame({
        "image_id": frame["image_id"].to_numpy(),
        "lesion_id": frame["lesion_id"].to_numpy(),
        "true_dx": frame["dx"].to_numpy(),
        "true_class_index": true,
        "predicted_dx": [labels[index] for index in predicted],
        "predicted_class_index": predicted,
    })
    for index, name in enumerate(labels):
        predictions[f"prob_{name}"] = probabilities[:, index]
    return predictions, metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mapping(splits_dir: Path) -> dict[str, int]:
    mapping = _read_json(splits_dir / "class_mapping.json")
    if sorted(mapping.values()) != list(range(7)) or len(mapping) != 7:
        raise ValueError("Saved class mapping must contain seven consecutive indices.")
    return mapping


def _split(splits_dir: Path, name: str, mapping: dict[str, int]) -> pd.DataFrame:
    frame = pd.read_csv(splits_dir / f"{name}.csv", dtype={"image_id": str, "lesion_id": str, "dx": str})
    required = {"image_id", "lesion_id", "dx", "image_path", "class_index"}
    if (not required.issubset(frame) or frame[list(required)].isna().any().any()
            or frame.image_id.duplicated().any() or len(frame) != 1431
            or frame.lesion_id.nunique() != 1068
            or set(frame.dx) != set(mapping)
            or not frame.class_index.eq(frame.dx.map(mapping)).all()):
        raise ValueError(f"Saved {name} split has invalid IDs, labels, class coverage, or row count.")
    return frame


def _verify_split_counts(splits_dir: Path) -> None:
    counts = pd.read_csv(splits_dir / "split_counts.csv").set_index("split")
    expected = {"train": (7153, 5334), "validation": (1431, 1068), "test": (1431, 1068)}
    if set(counts.index) != set(expected):
        raise ValueError("Saved split count rows differ from the fixed experiment protocol.")
    for name, (images, lesions) in expected.items():
        if int(counts.loc[name, "images"]) != images or int(counts.loc[name, "lesions"]) != lesions:
            raise ValueError(f"Saved {name} split counts differ from the fixed experiment protocol.")


def align_probabilities(frame: pd.DataFrame, predictions: pd.DataFrame,
                        mapping: dict[str, int]) -> np.ndarray:
    """Validate and order saved probabilities against one untouched split."""
    names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    columns = [f"prob_{name}" for name in names]
    metadata = {"image_id", "lesion_id", "true_dx", "true_class_index", "predicted_dx", "predicted_class_index"}
    if set(predictions.columns) != metadata | set(columns):
        raise ValueError("Prediction columns or class order differ from saved mapping.")
    if list(predictions.columns) != [*metadata_order(), *columns]:
        raise ValueError("Prediction class columns are out of saved integer order.")
    if (len(predictions) != len(frame) or predictions.image_id.isna().any()
            or predictions.image_id.duplicated().any()
            or set(predictions.image_id) != set(frame.image_id)):
        raise ValueError("Prediction image IDs are duplicate, missing, or unexpected.")
    aligned = predictions.set_index("image_id").loc[frame.image_id.to_numpy()].reset_index()
    for actual, expected in (("lesion_id", "lesion_id"), ("true_dx", "dx"),
                             ("true_class_index", "class_index")):
        if not aligned[actual].eq(frame[expected].to_numpy()).all():
            raise ValueError(f"Prediction {actual} disagrees with the saved split.")
    values = aligned[columns].to_numpy(dtype=np.float64)
    if (not np.isfinite(values).all() or np.any(values < -PROBABILITY_TOLERANCE)
            or np.any(values > 1 + PROBABILITY_TOLERANCE)
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=PROBABILITY_TOLERANCE)):
        raise ValueError("Probabilities are nonfinite, outside [0,1], or do not sum to one.")
    chosen = values.argmax(axis=1)
    if (not np.array_equal(chosen, aligned.predicted_class_index.to_numpy(dtype=int))
            or not np.array_equal(np.asarray(names)[chosen], aligned.predicted_dx.to_numpy())):
        raise ValueError("Saved hard predictions disagree with saved probabilities.")
    return values


def metadata_order() -> tuple[str, ...]:
    return ("image_id", "lesion_id", "true_dx", "true_class_index", "predicted_dx", "predicted_class_index")


def average_probabilities(day6: np.ndarray, day11: np.ndarray) -> np.ndarray:
    """Average one predefined pair of aligned softmax matrices."""
    if day6.shape != day11.shape or day6.ndim != 2 or day6.shape[1] != 7:
        raise ValueError("Ensemble components have incompatible shapes.")
    return 0.5 * day6 + 0.5 * day11


def select_final_system(day11_macro_f1: float, ensemble_macro_f1: float) -> str:
    """A strict full-precision improvement is required for the ensemble."""
    return "ensemble" if ensemble_macro_f1 > day11_macro_f1 else "day11"


def _score(frame: pd.DataFrame, probabilities: np.ndarray,
           mapping: dict[str, int]) -> tuple[pd.DataFrame, dict, np.ndarray]:
    if (not np.isfinite(probabilities).all() or np.any(probabilities < -PROBABILITY_TOLERANCE)
            or np.any(probabilities > 1 + PROBABILITY_TOLERANCE)
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=PROBABILITY_TOLERANCE)):
        raise ValueError("Inference probabilities are nonfinite, outside [0,1], or not normalized.")
    predictions, metrics = evaluate_validation(frame, probabilities, mapping)
    labels = list(range(len(mapping)))
    matrix = confusion_matrix(predictions.true_class_index, predictions.predicted_class_index, labels=labels)
    return predictions, metrics, matrix


def _write_score(directory: Path, name: str, frame: pd.DataFrame,
                 probabilities: np.ndarray, mapping: dict[str, int]) -> dict:
    predictions, metrics, matrix = _score(frame, probabilities, mapping)
    if name.startswith("test_"):
        metrics = {key.replace("validation_", "test_", 1) if key.startswith("validation_") else key: value
                   for key, value in metrics.items()}
    names = [name for name, _ in sorted(mapping.items(), key=lambda pair: pair[1])]
    predictions.to_csv(directory / f"{name}_predictions.csv", index=False)
    save_json(metrics, directory / f"{name}_metrics.json")
    pd.DataFrame(matrix, index=names, columns=names).to_csv(directory / f"{name}_confusion_matrix.csv")
    pd.DataFrame(matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1),
                 index=names, columns=names).to_csv(directory / f"{name}_confusion_matrix_normalized.csv")
    return metrics


def _comparison(directory: Path, phase: str, scores: dict[str, dict], mapping: dict[str, int],
                reference: str, candidate: str) -> None:
    keys = (f"{phase}_accuracy", f"{phase}_macro_f1", f"{phase}_balanced_accuracy")
    pd.DataFrame([{"system": name, **{key: metrics[key] for key in keys}}
                  for name, metrics in scores.items()]).to_csv(directory / f"{phase}_aggregate_comparison.csv", index=False)
    pd.DataFrame([{"metric": key, "reference": scores[reference][key],
                   "candidate": scores[candidate][key],
                   "candidate_minus_reference": scores[candidate][key] - scores[reference][key]}
                  for key in keys]).to_csv(directory / f"{phase}_aggregate_differences.csv", index=False)
    rows = []
    for class_name, index in sorted(mapping.items(), key=lambda pair: pair[1]):
        old = scores[reference]["per_class"][class_name]
        new = scores[candidate]["per_class"][class_name]
        rows.append({"class_index": index, "class_name": class_name, "support": old["support"],
                     **{f"{key}_{reference}": old[key] for key in ("precision", "recall", "f1")},
                     **{f"{key}_{candidate}": new[key] for key in ("precision", "recall", "f1")},
                     **{f"delta_{key}": new[key] - old[key] for key in ("precision", "recall", "f1")}})
    pd.DataFrame(rows).to_csv(directory / f"{phase}_per_class_comparison.csv", index=False)


def _verified_runs(outputs_dir: Path, splits_dir: Path, mapping: dict[str, int]) -> dict:
    split_hashes = {name: _sha256(splits_dir / name) for name in PROVENANCE_SPLIT_FILES}
    result = {}
    for role, (run_id, expected_hash) in EVALUATION_RUNS.items():
        directory = outputs_dir / "runs" / run_id
        checkpoint = directory / "best.keras"
        actual_hash = _sha256(checkpoint)
        config = _read_json(directory / "config.json")
        metrics = _read_json(directory / "metrics.json")
        if actual_hash != expected_hash or config.get("run_id") != run_id or metrics.get("run_id") != run_id:
            raise ValueError(f"Checkpoint hash or run identity mismatch: {role}.")
        if (config.get("architecture") != "efficientnetb0" or config.get("class_mapping") != mapping
                or Path(config.get("splits_dir", "")).resolve() != splits_dir.resolve()
                or config.get("augmentation") is not False):
            raise ValueError(f"Model or split provenance mismatch: {role}.")
        if role == "baseline":
            if (config.get("comparison_group") != "day2_frozen_no_augmentation"
                    or config.get("backbone_trainable") is not False
                    or config.get("dropout") != 0 or config.get("l2") != 0
                    or config.get("class_weights") is not False
                    or config.get("parent_run_id") is not None):
                raise ValueError("Original Day 2 baseline provenance is inconsistent.")
        elif (config.get("split_file_sha256") != split_hashes
              or config.get("backbone_call_training") is not False):
            raise ValueError(f"Split hashes or inference semantics disagree: {role}.")
        if role == "day6" and config.get("comparison_group") != "efficientnetb0_partial_fine_tune":
            raise ValueError("Day 6 recipe provenance is inconsistent.")
        if role == "day11" and (config.get("comparison_group") != "efficientnetb0_broader_fine_tune"
                                or config.get("day6_comparator_run_id") != EVALUATION_RUNS["day6"][0]
                                or config.get("day6_checkpoint_sha256") != EVALUATION_RUNS["day6"][1]
                                or metrics.get("best_checkpoint_sha256") != actual_hash
                                or _read_json(directory / "decision.json").get("retained_candidate_run_id") != run_id):
            raise ValueError("Day 11 selection or comparator provenance is inconsistent.")
        result[role] = {"run_id": run_id, "checkpoint": str(checkpoint.resolve()),
                        "sha256": actual_hash, "comparison_group": config["comparison_group"],
                        "validation_macro_f1_saved": metrics["validation_macro_f1"]}
    return {"runs": result, "split_sha256": split_hashes}


def _runtime() -> dict:
    import tensorflow as tf
    return {"python": platform.python_version(), "tensorflow": tf.__version__,
            "keras": getattr(tf.keras, "__version__", "bundled"),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}


def freeze_validation_selection(outputs_dir: Path, splits_dir: Path) -> Path:
    """Validate the fixed ensemble using saved validation predictions, then freeze selection."""
    directory = outputs_dir / "day13"
    selection_path = directory / "final_selection.json"
    if selection_path.exists():
        raise FileExistsError(f"Final selection already exists: {selection_path}")
    mapping = _mapping(splits_dir)
    _verify_split_counts(splits_dir)
    provenance = _verified_runs(outputs_dir, splits_dir, mapping)
    manifest = _read_json(splits_dir / "split_manifest.json")
    if (manifest.get("group_column") != "lesion_id" or manifest.get("stratify_column") != "dx"
            or not manifest.get("zero_lesion_overlap") or not manifest.get("all_seven_classes_each_split")):
        raise ValueError("Saved split manifest does not establish the required lesion-grouped protocol.")
    frame = _split(splits_dir, "validation", mapping)
    values = {}
    scores = {}
    for role in ("day6", "day11"):
        run_dir = outputs_dir / "runs" / provenance["runs"][role]["run_id"]
        saved = pd.read_csv(run_dir / "validation_predictions.csv", dtype={"image_id": str})
        values[role] = align_probabilities(frame, saved, mapping)
        _, metrics, _ = _score(frame, values[role], mapping)
        historical = _read_json(run_dir / "metrics.json")
        for key in ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy"):
            if abs(metrics[key] - historical[key]) > METRIC_RECONCILIATION_TOLERANCE:
                raise ValueError(f"{role} saved validation {key} disagrees with recomputed predictions.")
        scores[role] = metrics
    values["ensemble"] = average_probabilities(values["day6"], values["day11"])
    scores["ensemble"] = _score(frame, values["ensemble"], mapping)[1]
    selected = select_final_system(scores["day11"]["validation_macro_f1"],
                                   scores["ensemble"]["validation_macro_f1"])
    directory.mkdir(parents=True, exist_ok=True)
    for role in ("day6", "day11", "ensemble"):
        _write_score(directory, f"validation_{role}", frame, values[role], mapping)
    _comparison(directory, "validation", scores, mapping, "day11", "ensemble")
    save_json({"runs": provenance["runs"], "split_sha256": provenance["split_sha256"],
               "class_mapping": mapping, "split_manifest": manifest,
               "probability_tolerance": PROBABILITY_TOLERANCE,
               "saved_probability_source": "verified run validation_predictions.csv",
               "environment": _runtime()}, directory / "checkpoint_provenance.json")
    selection = {"selected_system": selected,
                 "components": ["day6", "day11"] if selected == "ensemble" else ["day11"],
                 "weights": {"day6": 0.5, "day11": 0.5} if selected == "ensemble" else {"day11": 1.0},
                 "checkpoints": {role: provenance["runs"][role] for role in
                                 (["day6", "day11"] if selected == "ensemble" else ["day11"])},
                 "class_mapping": mapping, "split_sha256": provenance["split_sha256"],
                 "test_sha256": _sha256(splits_dir / "test.csv"),
                 "split_manifest": manifest,
                 "preprocessing": "RGB float32 224x224 in [0,255]; EfficientNet preprocess_input once (internal rescaling); no augmentation or TTA",
                 "decision_rule": "argmax of softmax probabilities; ensemble uses exact 0.5/0.5 average",
                 "validation_metrics": scores,
                 "selection_reason": "ensemble strictly exceeds Day 11 validation macro F1" if selected == "ensemble" else "ensemble does not strictly exceed Day 11 validation macro F1",
                 "selection_used_validation_only": True,
                 "selected_at_utc": datetime.now(timezone.utc).isoformat(),
                 "command": "python -m src.evaluate --phase validation --splits-dir day1_artifacts --outputs-dir outputs"}
    save_json(selection, selection_path)
    return selection_path


def require_frozen_selection(selection_path: Path) -> dict:
    """Load the persisted decision before any test split or image access."""
    if not selection_path.is_file():
        raise FileNotFoundError(f"Run validation selection first; missing {selection_path}")
    selection = _read_json(selection_path)
    if selection.get("selection_used_validation_only") is not True:
        raise ValueError("Final selection was not frozen from validation only.")
    scores = selection["validation_metrics"]
    chosen = select_final_system(scores["day11"]["validation_macro_f1"],
                                 scores["ensemble"]["validation_macro_f1"])
    if selection.get("selected_system") != chosen:
        raise ValueError("Frozen selection conflicts with its validation scores.")
    expected_components = ["day6", "day11"] if chosen == "ensemble" else ["day11"]
    expected_weights = {"day6": 0.5, "day11": 0.5} if chosen == "ensemble" else {"day11": 1.0}
    if selection.get("components") != expected_components or selection.get("weights") != expected_weights:
        raise ValueError("Frozen component identity or weights are inconsistent.")
    return selection


def _predict_checkpoint(checkpoint: Path, frame: pd.DataFrame) -> np.ndarray:
    import tensorflow as tf
    from .data_pipeline import make_image_dataset
    from .models import FrozenDropoutClassifier  # Register saved custom class before load_model.
    del FrozenDropoutClassifier
    model = tf.keras.models.load_model(checkpoint, compile=False)
    dataset = make_image_dataset(frame, training=False, batch_size=32, architecture="efficientnetb0")
    return np.asarray(model.predict(dataset, verbose=0))


def evaluate_frozen_test(outputs_dir: Path, splits_dir: Path) -> Path:
    """Evaluate only the baseline and the already frozen final system on test."""
    directory = outputs_dir / "day13"
    selection_path = directory / "final_selection.json"
    selection = require_frozen_selection(selection_path)  # Must precede test.csv access.
    if (directory / "test_summary.json").exists():
        raise FileExistsError("Final test evaluation already exists; refusing to replace it.")
    mapping = _mapping(splits_dir)
    _verify_split_counts(splits_dir)
    provenance = _verified_runs(outputs_dir, splits_dir, mapping)
    if (selection["class_mapping"] != mapping or selection["split_sha256"] != provenance["split_sha256"]
            or selection["checkpoints"] != {role: provenance["runs"][role] for role in selection["components"]}):
        raise ValueError("Frozen selection provenance no longer matches checkpoints or splits.")
    # The saved decision is verified above; test partition and images are first accessed below.
    if _sha256(splits_dir / "test.csv") != selection["test_sha256"]:
        raise ValueError("Test split hash changed after selection.")
    frame = _split(splits_dir, "test", mapping)
    values = {"baseline": _predict_checkpoint(Path(provenance["runs"]["baseline"]["checkpoint"]), frame)}
    if selection["selected_system"] == "ensemble":
        day6 = _predict_checkpoint(Path(provenance["runs"]["day6"]["checkpoint"]), frame)
        day11 = _predict_checkpoint(Path(provenance["runs"]["day11"]["checkpoint"]), frame)
        values["final"] = average_probabilities(day6, day11)
    else:
        values["final"] = _predict_checkpoint(Path(provenance["runs"]["day11"]["checkpoint"]), frame)
    scores = {role: _write_score(directory, f"test_{role}", frame, probability, mapping)
              for role, probability in values.items()}
    _comparison(directory, "test", scores, mapping, "baseline", "final")
    deltas = {key: scores["final"][key] - scores["baseline"][key]
              for key in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy")}
    class_deltas = {name: {key: scores["final"]["per_class"][name][key] -
                                   scores["baseline"]["per_class"][name][key]
                           for key in ("precision", "recall", "f1")}
                    for name in mapping}
    summary_path = directory / "test_summary.json"
    save_json({"selected_system": selection["selected_system"], "validation_selection": selection["selection_reason"],
               "baseline": scores["baseline"], "final": scores["final"], "final_minus_baseline": deltas,
               "per_class_final_minus_baseline": class_deltas,
               "test_rows": len(frame), "test_sha256": _sha256(splits_dir / "test.csv"),
               "selection_artifact": str(selection_path.resolve()), "selection_artifact_sha256": _sha256(selection_path),
               "environment": _runtime(),
               "command": "python -m src.evaluate --phase test --splits-dir day1_artifacts --outputs-dir outputs"}, summary_path)
    class_changes = sorted(((name, change["f1"]) for name, change in class_deltas.items()),
                           key=lambda item: item[1])
    lines = ["# Day 13 final evaluation", "",
             f"Validation selected **{selection['selected_system']}**: {selection['selection_reason']}.",
             f"Day 11 validation macro F1: {selection['validation_metrics']['day11']['validation_macro_f1']:.12f}.",
             f"Ensemble validation macro F1: {selection['validation_metrics']['ensemble']['validation_macro_f1']:.12f}.",
             "", "The final system was frozen before test inference. Test results did not change the selection.", "",
             "| Test metric | Day 2 baseline | Final | Final minus baseline |", "|---|---:|---:|---:|"]
    for key in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy"):
        lines.append(f"| {key} | {scores['baseline'][key]:.6f} | {scores['final'][key]:.6f} | {deltas[key]:+.6f} |")
    lines.extend(["", "Per-class F1 changes (final minus baseline), from lowest to highest:", ""])
    lines.extend(f"- {name}: {delta:+.6f}" for name, delta in class_changes)
    lines.extend(["", "Lesion grouping is by lesion_id; no patient-level or clinical validation is claimed.", ""])
    (directory / "evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    parser.add_argument("--splits-dir", type=Path, default=Path("day1_artifacts"))
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    result = (freeze_validation_selection(args.outputs_dir, args.splits_dir) if args.phase == "validation"
              else evaluate_frozen_test(args.outputs_dir, args.splits_dir))
    print(result)


if __name__ == "__main__":
    main()
