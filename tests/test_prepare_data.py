"""End-to-end preparation checks with small synthetic JPEG folders."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.prepare_data import CLASSES, load_fixed_splits, run_preparation, verify_splits


class PreparationTests(unittest.TestCase):
    def test_two_image_folders_audit_split_and_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for class_index, diagnosis in enumerate(CLASSES):
                for lesion_number in range(14):
                    image_id = f"img_{class_index}_{lesion_number}"
                    folder = root / f"HAM10000_images_part_{1 + lesion_number % 2}"
                    folder.mkdir(exist_ok=True)
                    Image.fromarray(np.full((12, 12, 3), 20 + class_index, dtype=np.uint8)).save(folder / f"{image_id}.jpg")
                    rows.append({"image_id": image_id, "lesion_id": f"lesion_{class_index}_{lesion_number}", "dx": diagnosis})
            metadata = root / "HAM10000_metadata.csv"
            pd.DataFrame(rows).to_csv(metadata, index=False)
            output = root / "artifacts"

            parts = run_preparation(input_root=root, metadata_path=metadata, output_dir=output)
            counts = verify_splits(parts)
            self.assertEqual(int(counts["images"].sum()), len(rows))
            self.assertEqual(len(pd.read_csv(output / "audit_issues.csv")), 0)
            discovered = json.loads((output / "discovered_paths.json").read_text(encoding="utf-8"))
            self.assertEqual(len(discovered["jpeg_folders"]), 2)
            self.assertTrue((output / "class_distribution.png").is_file())
            self.assertTrue((output / "sample_images.png").is_file())

            command = [
                sys.executable, "-m", "src.prepare_data", "--input-root", str(root),
                "--metadata", str(metadata), "--output-dir", str(output),
            ]
            rerun = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertIn("Reusing fixed split counts", rerun.stdout)

            reused = load_fixed_splits(output, pd.DataFrame(rows).assign(
                image_path=lambda f: f["image_id"].map({
                    path.stem: str(path.resolve()) for path in root.rglob("*.jpg")
                })
            ))
            for name in ("train", "validation", "test"):
                self.assertEqual(set(parts[name]["image_id"]), set(reused[name]["image_id"]))


if __name__ == "__main__":
    unittest.main()
