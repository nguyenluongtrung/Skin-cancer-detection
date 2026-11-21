# Repository Guidelines

## Project Structure & Module Organization

`src/prepare_data.py` audits HAM10000 and manages fixed lesion-grouped splits. `src/data_pipeline.py` handles RGB decoding, resizing, model-specific preprocessing, and datasets. `src/models.py` builds frozen ImageNet classifiers; `src/train.py` runs Day 2 baselines; `src/evaluate.py` scores validation predictions. Shared helpers and plots live in `src/utils.py` and `src/visualization.py`. `notebooks/01_eda.ipynb` imports these modules. `scripts/download_ham10000_isic.py` downloads the official ISIC release. Generated `data/`, `day1_artifacts/`, and `outputs/` are ignored by Git.

## Build, Test, and Development Commands

Install dependencies with `python -m pip install -r requirements.txt`. If needed, download data with `python -m scripts.download_ham10000_isic --output-dir data`. Run the Day 2 pipeline check with `python -m src.train --architecture efficientnetb0 --smoke`; explicitly start one baseline with `python -m src.train --architecture efficientnetb0 --run`. Run all three sequentially with `python -m src.train --all --run`. Commands read saved splits in `day1_artifacts/` and write completed runs to `outputs/`. Run tests with `python -m unittest discover -s tests -v`. There is no build step.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, uppercase constants, type hints, and short docstrings for reusable functions. Keep computation in functions or guarded CLI entry points, not module imports. Do not apply a universal pixel rescale: training-only augmentation precedes each architecture's preprocessing. No formatter or linter is configured yet.

## Testing & Evaluation Guidelines

Tests use `unittest` and `tests/test_*.py`. Synthetic checks cover lesion overlap detection, image order and pixel range, finite one-batch loss, and checkpoint reload. Before training, inspect the real audit, class plots, split counts, and smoke artifacts. Require all seven classes per split and zero lesion overlap. Select experiments with validation macro F1; keep test predictions and scores closed until decisions are final. Do not claim patient-level separation or invent results.

## Commits & Pull Requests

No Git history exists here, so no established commit format can be inferred. Use short imperative subjects such as `Refactor Day 1 data preparation`. Describe commands run, output changes, and limitations in pull requests. Include plots when visual outputs change. Do not commit dataset images, credentials, or model checkpoints.
