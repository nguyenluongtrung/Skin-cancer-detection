"""Download official ISIC 2018 Task 3 HAM10000 training data into data/.

The ISIC archive provides the same 10,015 HAM10000 training images as a single
ZIP. Lesion groupings and one-hot diagnoses are joined into a CSV with the
image_id, lesion_id, and dx columns required by this project.

Run: python -m scripts.download_ham10000_isic --output-dir data
Interrupted downloads resume using a per-range state file.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE = "https://isic-archive.s3.amazonaws.com/challenges/2018/"
IMAGES = "ISIC2018_Task3_Training_Input.zip"
GROUPINGS = "ISIC2018_Task3_Training_LesionGroupings.csv"
GROUND_TRUTH = "ISIC2018_Task3_Training_GroundTruth.zip"
CLASSES = ("AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC")
RANGE_BYTES = 32 * 1024 * 1024


def download_small(name: str, output_dir: Path) -> Path:
    path = output_dir / name
    response = requests.head(BASE + name, timeout=30)
    response.raise_for_status()
    expected = int(response.headers["Content-Length"])
    if path.is_file() and path.stat().st_size == expected:
        return path
    with requests.get(BASE + name, timeout=(20, 120)) as response:
        response.raise_for_status()
        path.write_bytes(response.content)
    if path.stat().st_size != expected:
        raise IOError(f"Incomplete download: {path}")
    print(f"Downloaded {name} ({expected:,} bytes)", flush=True)
    return path


def download_images(output_dir: Path, workers: int = 6) -> Path:
    """Download disjoint byte ranges in parallel and resume completed ranges."""
    path = output_dir / IMAGES
    state_path = output_dir / (IMAGES + ".state.json")
    response = requests.head(BASE + IMAGES, timeout=30)
    response.raise_for_status()
    total = int(response.headers["Content-Length"])
    etag = response.headers.get("ETag", "")
    segments = [(start, min(start + RANGE_BYTES, total) - 1) for start in range(0, total, RANGE_BYTES)]
    if path.is_file() and path.stat().st_size == total and not state_path.exists():
        print(f"Archive already downloaded: {path}", flush=True)
        return path
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("size") != total or state.get("etag") != etag:
            raise ValueError("Remote archive changed; move the partial archive and state file before retrying.")
    else:
        state = {"size": total, "etag": etag, "completed": []}
        state_path.write_text(json.dumps(state), encoding="utf-8")
    if not path.exists():
        with path.open("wb") as output:
            output.truncate(total)
    elif path.stat().st_size != total:
        raise ValueError("Partial archive size differs from download state; inspect the archive before retrying.")

    lock = threading.Lock()
    completed = set(state["completed"])
    remaining = [(index, start, end) for index, (start, end) in enumerate(segments) if index not in completed]
    print(f"Image archive: {total / 1024**3:.2f} GiB; {len(remaining)}/{len(segments)} ranges remaining", flush=True)

    def fetch(index: int, start: int, end: int) -> None:
        for attempt in range(4):
            try:
                with requests.get(BASE + IMAGES, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 120)) as result:
                    result.raise_for_status()
                    if result.status_code != 206 or result.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                        raise IOError(f"Server did not honor range {start}-{end}: {result.status_code}")
                    received = 0
                    with path.open("r+b") as output:
                        output.seek(start)
                        for chunk in result.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                                received += len(chunk)
                    if received != end - start + 1:
                        raise IOError(f"Incomplete range {index}: {received} bytes")
                with lock:
                    completed.add(index)
                    state["completed"] = sorted(completed)
                    temporary = state_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state), encoding="utf-8")
                    temporary.replace(state_path)
                    if len(completed) % 4 == 0 or len(completed) == len(segments):
                        print(f"Downloaded {len(completed)}/{len(segments)} ranges ({len(completed) / len(segments):.0%})", flush=True)
                return
            except (requests.RequestException, IOError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, *item) for item in remaining]
        for future in as_completed(futures):
            future.result()
    if len(completed) != len(segments):
        raise IOError("Image archive download incomplete.")
    state_path.unlink()
    return path


def build_metadata(groupings_path: Path, ground_truth_path: Path, output_dir: Path) -> Path:
    groupings = pd.read_csv(groupings_path)
    with zipfile.ZipFile(ground_truth_path) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        truth = pd.read_csv(archive.open(csv_name))
        archive.extract("ISIC2018_Task3_Training_GroundTruth/ATTRIBUTION.txt", output_dir)
        archive.extract("ISIC2018_Task3_Training_GroundTruth/LICENSE.txt", output_dir)
    if len(groupings) != 10015 or len(truth) != 10015:
        raise ValueError("Official grouping and diagnosis files must each contain 10,015 rows.")
    if not truth[list(CLASSES)].isin((0, 1)).all().all() or not truth[list(CLASSES)].sum(axis=1).eq(1).all():
        raise ValueError("Ground-truth rows must have exactly one of the seven diagnoses.")
    merged = groupings.merge(truth, on="image", how="inner", validate="one_to_one")
    if len(merged) != 10015:
        raise ValueError("Image IDs do not match between grouping and diagnosis files.")
    metadata = pd.DataFrame({
        "image_id": merged["image"],
        "lesion_id": merged["lesion_id"],
        "dx": merged[list(CLASSES)].idxmax(axis=1).str.lower(),
        "diagnosis_confirm_type": merged["diagnosis_confirm_type"],
    })
    output = output_dir / "HAM10000_metadata.csv"
    metadata.to_csv(output, index=False)
    print(f"Created {output}: {len(metadata):,} rows, {metadata['lesion_id'].nunique():,} lesions", flush=True)
    return output


def extract_images(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        image_names = [name for name in archive.namelist() if name.lower().endswith(".jpg")]
        if len(image_names) != 10015:
            raise ValueError(f"Expected 10,015 JPEGs in official archive; found {len(image_names)}")
        corrupt = archive.testzip()
        if corrupt:
            raise zipfile.BadZipFile(f"CRC check failed: {corrupt}")
        root = output_dir.resolve()
        for name in archive.namelist():
            target = (output_dir / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {name}")
        for index, name in enumerate(image_names, 1):
            info = archive.getinfo(name)
            target = output_dir / name
            if not target.is_file() or target.stat().st_size != info.file_size:
                archive.extract(info, output_dir)
            if index % 2000 == 0 or index == len(image_names):
                print(f"Extracted/verified {index:,}/{len(image_names):,} JPEGs", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    groupings_path = download_small(GROUPINGS, output_dir)
    ground_truth_path = download_small(GROUND_TRUTH, output_dir)
    build_metadata(groupings_path, ground_truth_path, output_dir)
    archive_path = download_images(output_dir, args.workers)
    extract_images(archive_path, output_dir)
    (output_dir / "SOURCE.txt").write_text(
        "HAM10000 / ISIC 2018 Task 3 training data, official ISIC archive.\n"
        f"Source: {BASE}\nFiles: {IMAGES}, {GROUPINGS}, {GROUND_TRUTH}\n"
        "HAM10000_metadata.csv was derived by joining image IDs, lesion groupings, "
        "and one-hot diagnoses; diagnosis_confirm_type is not a model predictor.\n"
        "Cite Tschandl et al., Scientific Data 5, 180161 (2018), and ISIC 2018.\n",
        encoding="utf-8",
    )
    print("Download complete. Run: python -m src.prepare_data --input-root data --output-dir day1_artifacts", flush=True)


if __name__ == "__main__":
    main()
