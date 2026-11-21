"""Shared reproducibility, runtime, and serialization helpers."""

from __future__ import annotations

import json
import platform
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from PIL import Image

SEED = 42


def save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def set_seed(seed: int = SEED) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def runtime_info(batch_size: int) -> dict:
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": getattr(tf.keras, "__version__", "bundled with TensorFlow"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "pillow": getattr(Image, "__version__", "unknown"),
        "gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "seed": SEED,
        "batch_size": batch_size,
    }


def create_run_directory(outputs_dir: Path, architecture: str) -> tuple[str, Path]:
    """Create a unique run folder without overwriting an existing experiment."""
    runs_dir = Path(outputs_dir).expanduser().resolve() / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{stamp}_{architecture.lower()}_{uuid.uuid4().hex[:8]}"
        run_dir = runs_dir / run_id
        try:
            run_dir.mkdir(exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            continue
    raise FileExistsError("Could not create a unique run directory.")


def append_experiment_log(outputs_dir: Path, row: dict) -> None:
    """Append one complete run, preserving the log's existing column order."""
    path = Path(outputs_dir).expanduser().resolve() / "experiment_log.csv"
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            columns = list(reader.fieldnames or [])
            existing = list(reader)
        if any(previous.get("run_id") == row["run_id"] for previous in existing):
            raise ValueError(f"Run ID already logged: {row['run_id']}")
        columns.extend(key for key in row if key not in columns)
    else:
        columns = list(row)
        existing = []
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
    temporary.replace(path)


def invalidate_experiment(outputs_dir: Path, run_id: str, reason: str) -> None:
    """Retain a flawed run while excluding it from staged decisions."""
    outputs_dir = Path(outputs_dir).expanduser().resolve()
    path = outputs_dir / "experiment_log.csv"
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    matches = [row for row in rows if row["run_id"] == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one logged run {run_id!r}.")
    if "run_status" not in columns:
        columns.append("run_status")
    matches[0]["run_status"] = "invalid"
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    save_json({"run_id": run_id, "status": "invalid", "reason": reason},
              outputs_dir / "runs" / run_id / "invalidated.json")
