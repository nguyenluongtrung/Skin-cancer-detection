"""Compare the three Day 2 baselines with their Day 3 augmented runs.

Run after all three Day 3 jobs: python -m src.compare --outputs-dir outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .models import ARCHITECTURES
from .utils import save_json

METRICS = ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy")
GROUPS = {False: "day2_frozen_no_augmentation", True: "day3_frozen_moderate_augmentation"}
HEAD_EXPERIMENT_STAGES = ("T2_E0_CW", "T3_E0_D", "T3_E0_L")
OPTIMIZER_NAMES = ("adam", "rmsprop", "sgd_momentum")


def build_comparison(log: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require one completed run for each of the six planned configurations."""
    required = {"run_id", "comparison_group", "architecture", "augmentation", "best_epoch", "actual_epochs", "training_seconds", *METRICS}
    missing = required - set(log.columns)
    if missing:
        raise ValueError(f"Experiment log lacks columns: {sorted(missing)}")
    selected = []
    deltas = []
    for architecture in ARCHITECTURES:
        pair = {}
        for augmented, group in GROUPS.items():
            rows = log[(log["architecture"] == architecture) & (log["comparison_group"] == group)]
            if len(rows) != 1:
                raise ValueError(f"Expected one {architecture} run in {group}; found {len(rows)}. Complete the run or resolve duplicates.")
            row = rows.iloc[0]
            if str(row["augmentation"]).lower() != str(augmented).lower():
                raise ValueError(f"Augmentation flag conflicts with comparison group for {row['run_id']}.")
            pair[augmented] = row
            selected.append({
                "architecture": architecture, "augmentation": augmented,
                "best_epoch": int(row["best_epoch"]), "actual_epochs": int(row["actual_epochs"]),
                **{metric: float(row[metric]) for metric in METRICS},
                "training_seconds": float(row["training_seconds"]), "run_id": row["run_id"],
            })
        deltas.append({
            "architecture": architecture,
            **{f"delta_{metric}": float(pair[True][metric]) - float(pair[False][metric]) for metric in METRICS},
        })
    return pd.DataFrame(selected), pd.DataFrame(deltas)


def write_head_experiment_comparisons(outputs_dir: Path, mapping: dict[str, int]) -> None:
    """Refresh head experiment tables from saved best-checkpoint metrics."""
    outputs_dir = Path(outputs_dir).resolve()
    log = pd.read_csv(outputs_dir / "experiment_log.csv")
    rows = []
    decisions = []
    for stage in HEAD_EXPERIMENT_STAGES:
        matches = log.loc[log.comparison_group == stage]
        if "run_status" in matches:
            matches = matches.loc[matches.run_status.fillna("valid") != "invalid"]
        if matches.empty:
            break
        if len(matches) != 1:
            raise ValueError(f"Expected one completed {stage} run.")
        candidate_id = str(matches.iloc[0].run_id)
        parent_id = str(matches.iloc[0].parent_run_id)
        candidate = json.loads((outputs_dir / "runs" / candidate_id / "metrics.json").read_text(encoding="utf-8"))
        parent = json.loads((outputs_dir / "runs" / parent_id / "metrics.json").read_text(encoding="utf-8"))
        delta = candidate["validation_macro_f1"] - parent["validation_macro_f1"]
        decision = "keep" if delta > 0 else "reject"
        rows.append({"stage": stage, "run_id": candidate_id, "comparator_run_id": parent_id,
                     "decision": decision, **{f"{metric}_parent": parent[metric] for metric in METRICS},
                     **{f"{metric}_candidate": candidate[metric] for metric in METRICS},
                     **{f"delta_{metric}": candidate[metric] - parent[metric] for metric in METRICS}})
        decisions.append({"stage": stage, "run_id": candidate_id, "comparator_run_id": parent_id,
                          "decision": decision, "criterion": "strictly higher best-checkpoint validation macro F1",
                          "delta_validation_macro_f1": delta,
                          "selected_run_id": candidate_id if delta > 0 else parent_id})
        if stage == HEAD_EXPERIMENT_STAGES[0]:
            class_rows = []
            for name, index in sorted(mapping.items(), key=lambda pair: pair[1]):
                old = parent["per_class"][name]
                new = candidate["per_class"][name]
                class_rows.append({"class_index": index, "class_name": name, "support": old["support"],
                                   **{f"{key}_baseline": old[key] for key in ("precision", "recall", "f1")},
                                   **{f"{key}_weighted": new[key] for key in ("precision", "recall", "f1")},
                                   **{f"delta_{key}": new[key] - old[key] for key in ("precision", "recall", "f1")}})
            pd.DataFrame(class_rows).to_csv(outputs_dir / "day4_baseline_vs_weighted_per_class.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv(outputs_dir / "day4_comparison.csv", index=False)
        save_json(decisions, outputs_dir / "day4_decisions.json")


def write_optimizer_comparison(outputs_dir: Path, reference_run_id: str) -> None:
    """Compare saved five-epoch checkpoints only after all three runs finish."""
    outputs_dir = Path(outputs_dir).resolve()
    log = pd.read_csv(outputs_dir / "experiment_log.csv")
    group = log.loc[log.comparison_group == "frozen_optimizer_comparison"]
    if len(group) < len(OPTIMIZER_NAMES):
        return
    rows = []
    controls = []
    for name in OPTIMIZER_NAMES:
        matches = group.loc[group.optimizer == name]
        if len(matches) != 1 or str(matches.iloc[0].get("run_status", "valid")) == "invalid":
            raise ValueError(f"Expected one valid {name} optimizer run.")
        run_id = str(matches.iloc[0].run_id)
        run_dir = outputs_dir / "runs" / run_id
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        if metrics["actual_epochs"] != 5 or config["reference_run_id"] != reference_run_id:
            raise ValueError(f"Incomplete or mismatched optimizer run {run_id}.")
        controls.append((config["architecture"], config["seed"], tuple(config["image_size"]),
                         config["batch_size"], config["pretrained_weights"],
                         config["backbone_trainable"], config["augmentation"],
                         config["head_units"], config["class_weights"], config["dropout"],
                         config["l2"], config["loss"], config["selection_metric"],
                         config["initial_head_sha256"],
                         tuple(sorted(config["split_file_sha256"].items())), metrics["optimizer_updates"]))
        rows.append({
            "optimizer": name, "learning_rate": config["learning_rate"], "run_id": run_id,
            "best_epoch": metrics["best_epoch"],
            "validation_macro_f1": metrics["validation_macro_f1"],
            "validation_accuracy": metrics["validation_accuracy"],
            "validation_balanced_accuracy": metrics["validation_balanced_accuracy"],
            "training_seconds": metrics["training_seconds"],
        })
    if len(set(controls)) != 1:
        raise ValueError("Optimizer runs do not share seed, split, head initialization, or update count.")
    adam_score = rows[0]["validation_macro_f1"]
    for row in rows:
        row["delta_macro_f1_vs_fresh_adam"] = row["validation_macro_f1"] - adam_score
    pd.DataFrame(rows).to_csv(outputs_dir / "optimizer_comparison.csv", index=False)
    winner = max(rows, key=lambda row: row["validation_macro_f1"])
    reference = json.loads((outputs_dir / "runs" / reference_run_id / "metrics.json").read_text(encoding="utf-8"))
    save_json({
        "criterion": "highest best-checkpoint validation macro F1 among fresh five-epoch runs",
        "provisional_candidate_run_id": winner["run_id"],
        "provisional_candidate_optimizer": winner["optimizer"],
        "provisional_candidate_macro_f1": winner["validation_macro_f1"],
        "longer_budget_reference_run_id": reference_run_id,
        "longer_budget_reference_macro_f1": reference["validation_macro_f1"],
        "matched_longer_budget_confirmation_warranted": True,
        "note": "Five-epoch optimizer-plus-learning-rate results do not establish a universal optimizer ranking. Keep the longer-budget reference checkpoint unless a matched confirmation supports replacement.",
    }, outputs_dir / "optimizer_decision.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    log_path = args.outputs_dir / "experiment_log.csv"
    comparison, changes = build_comparison(pd.read_csv(log_path))
    comparison.to_csv(args.outputs_dir / "day3_six_run_comparison.csv", index=False)
    changes.to_csv(args.outputs_dir / "day3_augmentation_changes.csv", index=False)
    print("Six-run validation comparison:\n", comparison.to_string(index=False))
    print("\nAugmented minus baseline (multiply by 100 for percentage points):\n", changes.to_string(index=False))


if __name__ == "__main__":
    main()
