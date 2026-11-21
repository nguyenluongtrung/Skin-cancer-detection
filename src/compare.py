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
