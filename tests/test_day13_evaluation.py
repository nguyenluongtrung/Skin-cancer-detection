"""Focused checks for the staged validation and final test protocol."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (align_probabilities, average_probabilities,
                          evaluate_frozen_test, require_frozen_selection,
                          select_final_system)
from src.utils import save_json


class Day13EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = {name: index for index, name in enumerate(
            ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"))}
        self.frame = pd.DataFrame({"image_id": ["a", "b"], "lesion_id": ["l1", "l2"],
                                   "dx": ["akiec", "bcc"], "class_index": [0, 1]})
        probabilities = np.array([[0.7, 0.2, 0.1, 0, 0, 0, 0],
                                  [0.1, 0.8, 0.1, 0, 0, 0, 0]])
        self.saved = pd.DataFrame({"image_id": ["a", "b"], "lesion_id": ["l1", "l2"],
                                   "true_dx": ["akiec", "bcc"], "true_class_index": [0, 1],
                                   "predicted_dx": ["akiec", "bcc"],
                                   "predicted_class_index": [0, 1]})
        for index, name in enumerate(self.mapping):
            self.saved[f"prob_{name}"] = probabilities[:, index]
        self.probabilities = probabilities

    def test_shuffled_rows_align_by_image_id(self) -> None:
        shuffled = self.saved.iloc[[1, 0]].reset_index(drop=True)
        np.testing.assert_allclose(align_probabilities(self.frame, shuffled, self.mapping),
                                   self.probabilities)

    def test_duplicate_missing_unexpected_label_and_class_order_rejected(self) -> None:
        changes = []
        duplicate = self.saved.copy()
        duplicate.loc[1, "image_id"] = "a"
        changes.append(duplicate)
        missing = self.saved.iloc[:1].copy()
        changes.append(missing)
        unexpected = self.saved.copy()
        unexpected.loc[1, "image_id"] = "c"
        changes.append(unexpected)
        label = self.saved.copy()
        label.loc[1, "true_dx"] = "akiec"
        changes.append(label)
        order = self.saved.copy()
        columns = list(order.columns)
        columns[-1], columns[-2] = columns[-2], columns[-1]
        changes.append(order[columns])
        for changed in changes:
            with self.subTest(columns=list(changed.columns), ids=changed.image_id.tolist()):
                with self.assertRaises(ValueError):
                    align_probabilities(self.frame, changed, self.mapping)

    def test_average_and_argmax(self) -> None:
        first = np.zeros((1, 7))
        second = np.zeros((1, 7))
        first[0, 0], first[0, 1] = 0.8, 0.2
        second[0, 0], second[0, 1] = 0.1, 0.9
        combined = average_probabilities(first, second)
        self.assertAlmostEqual(combined[0, 1], 0.55)
        self.assertEqual(int(combined.argmax(axis=1)[0]), 1)

    def test_strict_selection(self) -> None:
        self.assertEqual(select_final_system(0.6, 0.61), "ensemble")
        self.assertEqual(select_final_system(0.6, 0.59), "day11")
        self.assertEqual(select_final_system(0.6, 0.6), "day11")

    def test_test_phase_requires_selection_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                require_frozen_selection(root / "final_selection.json")
            with self.assertRaises(FileNotFoundError):
                evaluate_frozen_test(root, root / "missing_splits")
            frozen = root / "day13" / "final_selection.json"
            save_json({"selection_used_validation_only": False}, frozen)
            with self.assertRaises(ValueError):
                evaluate_frozen_test(root, root / "missing_splits")


if __name__ == "__main__":
    unittest.main()
