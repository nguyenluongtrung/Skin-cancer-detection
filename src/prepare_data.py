"""Discover and audit HAM10000, then create or reuse fixed lesion-group splits.

Run from the repository root: python -m src.prepare_data --input-root DATA_DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedGroupKFold

from .utils import SEED, save_json
from .visualization import plot_dataset_overview

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}
SPLIT_NAMES = ("train", "validation", "test")


def search_roots(input_root: Path | None = None) -> list[Path]:
    if input_root is not None:
        chosen = Path(input_root).expanduser().resolve()
        if not chosen.is_dir():
            raise NotADirectoryError(
                f"Dataset folder does not exist: {chosen}. "
                "Pass the real extracted HAM10000 folder to --input-root; "
                "example paths such as C:\\path\\to\\extracted\\HAM10000 are placeholders."
            )
        return [chosen]
    configured = os.environ.get("HAM10000_DATA_ROOT")
    candidates = [configured] if configured else ["/kaggle/input", Path.cwd(), Path(__file__).resolve().parents[1]]
    roots = []
    for value in candidates:
        if value:
            path = Path(value).expanduser().resolve()
            if path.is_dir() and path not in roots:
                roots.append(path)
    return roots


def discover_data(
    input_root: Path | None = None, metadata_path: Path | None = None
) -> tuple[pd.DataFrame, Path, dict[str, Path], list[Path], dict[str, list[str]]]:
    """Locate metadata by required columns and JPEGs by extension, regardless of folder names."""
    roots = search_roots(input_root)
    if metadata_path is not None:
        chosen_path = Path(metadata_path).expanduser().resolve()
        if not chosen_path.is_file():
            raise FileNotFoundError(chosen_path)
        metadata = pd.read_csv(chosen_path, low_memory=False)
        if not REQUIRED_COLUMNS.issubset(metadata.columns):
            raise ValueError(f"Metadata lacks {sorted(REQUIRED_COLUMNS - set(metadata.columns))}")
    else:
        csv_paths = {p.resolve() for root in roots for p in root.rglob("*.csv")}
        candidates = []
        for path in sorted(csv_paths, key=lambda p: ("metadata" not in p.name.lower(), str(p))):
            try:
                frame = pd.read_csv(path, low_memory=False)
            except (OSError, UnicodeError, pd.errors.ParserError):
                continue
            if REQUIRED_COLUMNS.issubset(frame.columns):
                candidates.append((path, frame))
        if not candidates:
            raise FileNotFoundError("No metadata CSV with image_id, lesion_id, dx found. Attach the dataset or set --input-root.")
        if len(candidates) > 1:
            print("Metadata candidates:", [str(path) for path, _ in candidates])
        chosen_path, metadata = candidates[0]

    image_paths = set()
    for root in [*roots, chosen_path.parent]:
        for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            image_paths.update(p.resolve() for p in root.rglob(pattern))
    folders = sorted({p.parent for p in image_paths})
    image_index: dict[str, Path] = {}
    duplicate_files: dict[str, list[str]] = {}
    for path in sorted(image_paths):
        if path.stem in image_index:
            duplicate_files.setdefault(path.stem, [str(image_index[path.stem])]).append(str(path))
        else:
            image_index[path.stem] = path
    print("Metadata:", chosen_path)
    print("JPEG folders:", [str(path) for path in folders])
    print("JPEG file count:", len(image_paths))
    return metadata, chosen_path, image_index, folders, duplicate_files


def audit_data(
    metadata: pd.DataFrame, image_index: dict[str, Path], duplicate_files: dict[str, list[str]], output_dir: Path
) -> pd.DataFrame:
    """Write audit artifacts and reject missing, inconsistent, or unreadable training rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(duplicate_files, output_dir / "duplicate_image_files.json")
    frame = metadata.copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = frame[column].astype("string").str.strip()
        frame.loc[frame[column].eq(""), column] = pd.NA

    issues = []

    def add_issue(kind: str, rows: pd.DataFrame) -> None:
        for _, row in rows.iterrows():
            issues.append({"issue": kind, "image_id": row.get("image_id"), "lesion_id": row.get("lesion_id"), "dx": row.get("dx")})

    for column in ("image_id", "lesion_id", "dx"):
        add_issue(f"missing_{column}", frame[frame[column].isna()])
    add_issue("unexpected_dx", frame[frame["dx"].notna() & ~frame["dx"].isin(CLASSES)])
    add_issue("duplicate_metadata_image_id", frame[frame["image_id"].notna() & frame["image_id"].duplicated(keep=False)])
    diagnosis_counts = frame.dropna(subset=["lesion_id", "dx"]).groupby("lesion_id")["dx"].nunique()
    conflicting_lesions = set(diagnosis_counts[diagnosis_counts > 1].index)
    add_issue("conflicting_dx_within_lesion", frame[frame["lesion_id"].isin(conflicting_lesions)])
    frame["image_path"] = frame["image_id"].map(
        lambda x: str(image_index[x]) if pd.notna(x) and x in image_index else pd.NA
    )
    add_issue("missing_image_file", frame[frame["image_id"].notna() & frame["image_path"].isna()])
    add_issue("duplicate_image_file_id", frame[frame["image_id"].isin(duplicate_files)])

    unreadable = []
    for path in sorted(set(frame["image_path"].dropna())):
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError, UnidentifiedImageError):
            unreadable.append(path)
    add_issue("unreadable_image", frame[frame["image_path"].isin(unreadable)])

    issue_frame = pd.DataFrame(issues, columns=["issue", "image_id", "lesion_id", "dx"])
    issue_frame.to_csv(output_dir / "audit_issues.csv", index=False)
    summary = {
        "metadata_rows": len(frame),
        "unique_image_ids": int(frame["image_id"].nunique()),
        "unique_lesions": int(frame["lesion_id"].nunique()),
        "indexed_jpeg_files": len(image_index),
        "class_images": {str(k): int(v) for k, v in frame["dx"].value_counts(dropna=False).items()},
        "class_lesions": {str(k): int(v) for k, v in frame.drop_duplicates("lesion_id")["dx"].value_counts(dropna=False).items()},
        "issue_counts": {str(k): int(v) for k, v in issue_frame["issue"].value_counts().items()},
    }
    save_json(summary, output_dir / "audit_summary.json")
    print("Audit:", json.dumps(summary, indent=2))
    plot_dataset_overview(frame, output_dir)
    if not issue_frame.empty:
        raise ValueError(f"Dataset audit failed. Inspect {output_dir / 'audit_issues.csv'} before splitting.")
    if set(frame["dx"]) != set(CLASSES):
        raise ValueError(f"Expected seven HAM10000 classes; found {sorted(frame['dx'].unique())}.")
    return frame


def verify_splits(parts: dict[str, pd.DataFrame], source: pd.DataFrame | None = None) -> pd.DataFrame:
    """Check complete membership, class coverage, unique image IDs, and zero lesion overlap."""
    if set(parts) != set(SPLIT_NAMES):
        raise ValueError(f"Expected splits {SPLIT_NAMES}; found {sorted(parts)}")
    lesion_sets = {name: set(parts[name]["lesion_id"]) for name in SPLIT_NAMES}
    if any(not lesion_sets[a].isdisjoint(lesion_sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("A lesion_id appears in multiple splits.")
    combined = pd.concat([parts[name] for name in SPLIT_NAMES], ignore_index=True)
    if combined["image_id"].duplicated().any():
        raise ValueError("An image_id appears in multiple split rows.")
    if any(set(part["dx"]) != set(CLASSES) for part in parts.values()):
        raise ValueError("Every split must contain all seven classes.")
    mapping = {label: index for index, label in enumerate(CLASSES)}
    if any(not part["class_index"].astype(int).eq(part["dx"].map(mapping)).all() for part in parts.values()):
        raise ValueError("Split class indices do not match the fixed class mapping.")
    if source is not None:
        if len(combined) != len(source) or set(combined["image_id"]) != set(source["image_id"]):
            raise ValueError("Saved splits do not cover the audited metadata exactly.")
        source_labels = source.set_index("image_id")["dx"]
        if not combined["dx"].eq(combined["image_id"].map(source_labels)).all():
            raise ValueError("Saved split diagnoses differ from current metadata.")
    rows = []
    for name in SPLIT_NAMES:
        part = parts[name]
        rows.append({
            "split": name, "images": len(part), "lesions": part["lesion_id"].nunique(),
            "fraction": len(part) / len(combined),
            **{f"images_{label}": int((part["dx"] == label).sum()) for label in CLASSES},
            **{f"lesions_{label}": int(part.loc[part["dx"] == label, "lesion_id"].nunique()) for label in CLASSES},
        })
    return pd.DataFrame(rows)


def create_fixed_splits(frame: pd.DataFrame, output_dir: Path, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Select 5/1/1 stratified group folds, approximately 70/15/15 by image count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {name: index for index, name in enumerate(CLASSES)}
    frame = frame.copy()
    frame["class_index"] = frame["dx"].map(mapping).astype("int64")
    splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    fold_for_row = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, held) in enumerate(splitter.split(frame, frame["class_index"], groups=frame["lesion_id"])):
        fold_for_row[held] = fold
    if (fold_for_row < 0).any():
        raise AssertionError("Some rows received no fold.")
    overall = frame["dx"].value_counts(normalize=True).reindex(CLASSES).to_numpy()
    best = None
    for validation_fold in range(7):
        for test_fold in range(7):
            if validation_fold == test_fold:
                continue
            masks = {
                "train": (fold_for_row != validation_fold) & (fold_for_row != test_fold),
                "validation": fold_for_row == validation_fold,
                "test": fold_for_row == test_fold,
            }
            parts = {name: frame.loc[mask] for name, mask in masks.items()}
            if any(set(part["dx"]) != set(CLASSES) for part in parts.values()):
                continue
            targets = {"train": 5 / 7, "validation": 1 / 7, "test": 1 / 7}
            size_error = sum(abs(len(part) / len(frame) - targets[name]) for name, part in parts.items())
            balance_error = sum(
                np.abs(part["dx"].value_counts(normalize=True).reindex(CLASSES, fill_value=0).to_numpy() - overall).mean()
                for part in parts.values()
            )
            score = size_error + balance_error
            if best is None or score < best[0]:
                best = (score, validation_fold, test_fold, parts)
    if best is None:
        raise ValueError("No seven-fold assignment preserved all classes in all splits.")
    _, validation_fold, test_fold, parts = best
    counts = verify_splits(parts, frame)
    save_json(mapping, output_dir / "class_mapping.json")
    for name, part in parts.items():
        part.to_csv(output_dir / f"{name}.csv", index=False)
    counts.to_csv(output_dir / "split_counts.csv", index=False)
    save_json({
        "seed": seed,
        "method": "StratifiedGroupKFold, seven folds, five train/one validation/one test",
        "group_column": "lesion_id", "stratify_column": "dx",
        "validation_fold": validation_fold, "test_fold": test_fold,
        "zero_lesion_overlap": True, "all_seven_classes_each_split": True,
        "note": "Lesion-level grouping only; no claim of patient-level separation.",
    }, output_dir / "split_manifest.json")
    print("Fixed split counts:\n", counts.to_string(index=False))
    return parts


def load_fixed_splits(output_dir: Path, source: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Reuse assignments; reject incomplete or inconsistent saved artifacts."""
    mapping_path = output_dir / "class_mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(mapping_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping != {name: index for index, name in enumerate(CLASSES)}:
        raise ValueError("Saved class mapping differs from the fixed HAM10000 mapping.")
    parts = {name: pd.read_csv(output_dir / f"{name}.csv", dtype={"image_id": str, "lesion_id": str, "dx": str}) for name in SPLIT_NAMES}
    counts = verify_splits(parts, source)
    if source is not None:
        current_paths = source.set_index("image_id")["image_path"]
        for name, part in parts.items():
            resolved_paths = part["image_id"].map(current_paths)
            if not part["image_path"].eq(resolved_paths).all():
                part = part.copy()
                part["image_path"] = resolved_paths
                part.to_csv(output_dir / f"{name}.csv", index=False)
                parts[name] = part
                print(f"Updated {name} image paths; split membership is unchanged.")
    print("Reusing fixed split counts:\n", counts.to_string(index=False))
    return parts


def run_preparation(
    input_root: Path | None = None, output_dir: Path = Path("day1_artifacts"),
    metadata_path: Path | None = None, force_resplit: bool = False,
) -> dict[str, pd.DataFrame]:
    """Day 1 preparation entry point; existing valid splits are reused by default."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, chosen_path, image_index, folders, duplicate_files = discover_data(input_root, metadata_path)
    save_json({"metadata": str(chosen_path), "jpeg_folders": [str(p) for p in folders]}, output_dir / "discovered_paths.json")
    clean = audit_data(metadata, image_index, duplicate_files, output_dir)
    existing = [output_dir / f"{name}.csv" for name in SPLIT_NAMES] + [output_dir / "class_mapping.json"]
    if not force_resplit and all(path.is_file() for path in existing):
        return load_fixed_splits(output_dir, clean)
    if not force_resplit and any(path.exists() for path in existing):
        raise FileExistsError("Some fixed split files exist. Restore the full set or pass --force-resplit deliberately.")
    return create_fixed_splits(clean, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, help="Extracted dataset root; auto-discovered if omitted")
    parser.add_argument("--metadata", type=Path, help="Explicit metadata CSV when discovery is ambiguous")
    parser.add_argument("--output-dir", type=Path, default=Path("day1_artifacts"))
    parser.add_argument("--force-resplit", action="store_true", help="Replace existing fixed split assignments")
    args = parser.parse_args()
    run_preparation(args.input_root, args.output_dir, args.metadata, args.force_resplit)


if __name__ == "__main__":
    main()
