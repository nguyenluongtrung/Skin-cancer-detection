"""Small synthetic checks for split safety, image order, and checkpoint reload."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.prepare_data import CLASSES, verify_splits


class Day1PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parts = {}
        for split_number, split in enumerate(("train", "validation", "test")):
            rows = []
            for class_index, diagnosis in enumerate(CLASSES):
                image_id = f"{split}_{diagnosis}"
                path = self.root / f"{image_id}.jpg"
                Image.fromarray(np.full((16, 16, 3), 20 + class_index, dtype=np.uint8)).save(path)
                rows.append({
                    "image_id": image_id, "lesion_id": f"lesion_{split_number}_{class_index}",
                    "dx": diagnosis, "image_path": str(path), "class_index": class_index,
                })
            self.parts[split] = pd.DataFrame(rows)

    def test_split_verification_catches_lesion_leakage(self):
        counts = verify_splits(self.parts)
        self.assertEqual(counts["images"].tolist(), [7, 7, 7])
        corrupted = {name: frame.copy() for name, frame in self.parts.items()}
        corrupted["test"].loc[0, "lesion_id"] = corrupted["train"].loc[0, "lesion_id"]
        with self.assertRaisesRegex(ValueError, "lesion_id"):
            verify_splits(corrupted)

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is not installed")
    def test_loader_preserves_label_order_and_checkpoint_reloads(self):
        from src.data_pipeline import make_image_dataset, smoke_check

        train = self.parts["train"]
        dataset = make_image_dataset(train, batch_size=7)
        images, labels = next(iter(dataset))
        self.assertEqual(tuple(images.shape), (7, 224, 224, 3))
        self.assertEqual(labels.numpy().tolist(), list(range(7)))
        self.assertGreaterEqual(float(np.min(images.numpy())), 0)
        self.assertLessEqual(float(np.max(images.numpy())), 255)
        result = smoke_check(dataset, self.root)
        self.assertTrue(np.isfinite(result["one_batch_loss"]))
        self.assertTrue((self.root / "smoke_checkpoint.keras").is_file())


if __name__ == "__main__":
    unittest.main()
