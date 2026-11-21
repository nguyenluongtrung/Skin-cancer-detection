"""Compare the three Day 2 baselines with their Day 3 augmented runs.

Run after all three Day 3 jobs: python -m src.compare --outputs-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .models import ARCHITECTURES

METRICS = ("validation_accuracy", "validation_macro_f1", "validation_balanced_accuracy")
GROUPS = {False: "day2_frozen_no_augmentation", True: "day3_frozen_moderate_augmentation"}


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
