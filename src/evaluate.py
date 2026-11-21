"""Validation-only predictions and classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support


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
