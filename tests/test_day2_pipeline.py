"""Focused checks for fixed mapping, preprocessing, and validation outputs."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data_pipeline import load_training_partitions, make_image_dataset
from src.evaluate import evaluate_validation
from src.utils import append_experiment_log, create_run_directory


class Day2PipelineTests(unittest.TestCase):
    def test_saved_mapping_and_validation_order_without_test_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = {name: index for index, name in enumerate(("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))}
            (root / "class_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
            frames = {}
            for split in ("train", "validation"):
                rows = []
                for name, index in mapping.items():
                    image_id = f"{split}_{name}"
                    path = root / f"{image_id}.jpg"
                    Image.fromarray(np.full((10, 10, 3), index * 20, dtype=np.uint8)).save(path)
                    rows.append({
                        "image_id": image_id, "lesion_id": image_id, "dx": name,
                        "image_path": str(path), "class_index": index,
                    })
                frames[split] = pd.DataFrame(rows)
                frames[split].to_csv(root / f"{split}.csv", index=False)
            train, validation, loaded_mapping = load_training_partitions(root)
            self.assertEqual(loaded_mapping, mapping)
            self.assertEqual(len(train), 7)
            dataset = make_image_dataset(validation, training=False, batch_size=7, architecture="mobilenetv2")
            pixels, labels = next(iter(dataset))
            self.assertEqual(labels.numpy().tolist(), list(range(7)))
            self.assertEqual(tuple(pixels.shape), (7, 224, 224, 3))
            self.assertGreaterEqual(float(np.min(pixels.numpy())), -1.01)
            self.assertLessEqual(float(np.max(pixels.numpy())), 1.01)

            probabilities = np.eye(7, dtype=np.float32)
            predictions, metrics = evaluate_validation(validation, probabilities, mapping)
            self.assertEqual(predictions["image_id"].tolist(), validation["image_id"].tolist())
            self.assertEqual(predictions["prob_akiec"].tolist(), probabilities[:, 0].tolist())
            self.assertEqual(metrics["validation_macro_f1"], 1.0)
            self.assertEqual(metrics["validation_balanced_accuracy"], 1.0)

    def test_run_ids_and_log_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary)
            first, _ = create_run_directory(outputs, "efficientnetb0")
            second, _ = create_run_directory(outputs, "efficientnetb0")
            self.assertNotEqual(first, second)
            append_experiment_log(outputs, {"run_id": first, "validation_macro_f1": 0.1})
            with self.assertRaisesRegex(ValueError, "already logged"):
                append_experiment_log(outputs, {"run_id": first, "validation_macro_f1": 0.2})
            self.assertEqual(len(pd.read_csv(outputs / "experiment_log.csv")), 1)


if __name__ == "__main__":
    unittest.main()
