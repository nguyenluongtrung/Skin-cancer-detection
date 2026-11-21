# Repository Guidelines

## Project Structure & Module Organization

`src/prepare_data.py` discovers HAM10000 metadata and JPEGs, audits records, and creates or verifies fixed lesion-grouped splits. `src/data_pipeline.py` handles RGB decoding, resizing, model-specific preprocessing, dataset construction, and the Day 1 smoke check. Shared seed and JSON helpers live in `src/utils.py`; plotting lives in `src/visualization.py`. `notebooks/01_eda.ipynb` imports these modules for exploration. `scripts/download_ham10000_isic.py` downloads the official ISIC release and builds the required metadata CSV. Generated data and artifacts are ignored by Git. The README describes later work, but experiment modules do not exist yet.

## Build, Test, and Development Commands

Install dependencies with `python -m pip install -r requirements.txt`. If needed, run `python -m scripts.download_ham10000_isic --output-dir data`; this downloads about 2.58 GiB and records provenance. Then run `python -m src.prepare_data --input-root data --output-dir day1_artifacts` to audit and split, followed by `python -m src.data_pipeline --output-dir day1_artifacts --batch-size 32` for the one-batch save/reload check. On Kaggle, set `--input-root /kaggle/input` and output under `/kaggle/working/`. Run synthetic checks with `python -m unittest discover -s tests -v`. There is no build step.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, uppercase constants, type hints, and short docstrings for reusable functions. Keep computation in functions or guarded CLI entry points, not module imports. Do not apply a universal pixel rescale: training-only augmentation precedes each architecture's preprocessing. No formatter or linter is configured yet.

## Testing & Evaluation Guidelines

Tests use `unittest` and `tests/test_*.py`. Synthetic checks cover lesion overlap detection, image order and pixel range, finite one-batch loss, and checkpoint reload. Before training, inspect the real audit, class plots, split counts, and smoke artifacts. Require all seven classes per split and zero lesion overlap. Select experiments with validation macro F1; keep test predictions and scores closed until decisions are final. Do not claim patient-level separation or invent results.

## Commits & Pull Requests

No Git history exists here, so no established commit format can be inferred. Use short imperative subjects such as `Refactor Day 1 data preparation`. Describe commands run, output changes, and limitations in pull requests. Include plots when visual outputs change. Do not commit dataset images, credentials, or model checkpoints.
