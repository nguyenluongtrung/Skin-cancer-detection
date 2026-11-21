"""Shared reproducibility, runtime, and serialization helpers."""

from __future__ import annotations

import json
import platform
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
