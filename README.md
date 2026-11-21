# HAM10000 Skin Lesion Classification

An ML/DL assignment investigating how pretrained architectures, data augmentation, class weighting, regularization, optimizers, and fine-tuning affect seven-class skin lesion classification.

## Project Objectives

- Compare MobileNetV2, EfficientNetB0, and ResNet50.
- Measure augmentation’s effect on each architecture.
- Investigate techniques for improving class-balanced performance.
- Prevent data leakage through lesion-grouped splitting.
- Build a reproducible Python training and evaluation pipeline.
- Explain experimental results, including changes that do not improve performance.

## Dataset

**Skin Cancer MNIST: HAM10000**

Dataset: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-HAM10000/data

The repository also provides a resumable downloader for the [official ISIC 2018 Task 3 training release](https://challenge.isic-archive.com/data/). It supplies the 10,015 HAM10000 training JPEGs, lesion groupings, and diagnosis ground truth. `scripts/download_ham10000_isic.py` joins the official grouping and one-hot diagnosis files into `data/HAM10000_metadata.csv`. This derived CSV has the required `image_id`, `lesion_id`, and `dx` columns; it is not a byte-for-byte copy of Kaggle's metadata CSV. Source URLs and attribution are recorded in `data/SOURCE.txt` after download.

The dataset contains 10,015 dermatoscopic images across seven diagnostic categories. These include both benign and malignant lesions.

| Code    | Diagnostic category                             |
| ------- | ----------------------------------------------- |
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc`   | Basal cell carcinoma                            |
| `bkl`   | Benign keratosis-like lesions                   |
| `df`    | Dermatofibroma                                  |
| `mel`   | Melanoma                                        |
| `nv`    | Melanocytic nevi                                |
| `vasc`  | Vascular lesions                                |

The project uses:

- Original RGB JPEG images from both Kaggle image folders, or the equivalent official ISIC training archive (one image folder).
- `HAM10000_metadata.csv` from Kaggle or the derived three-column-compatible ISIC CSV.
- `dx` as the classification target.
- `lesion_id` for grouped splitting.
- `image_id` for image identification.

The initial model is image-only. Other metadata may be explored but is not used as a predictor. In particular, `dx_type` is excluded because it describes how the diagnosis was established.

Follow the dataset’s published license and attribution requirements.

## Implementation Reference

Main reference notebook:

[Skin Cancer Detection – EfficientNetB3 (HAM10000), by Forough Ghayyem](https://www.kaggle.com/code/foroughgh95/skin-cancer-detection-efficientnetb3-ham10000)

The reference provides implementation ideas for exploration, image loading, transfer learning, and evaluation.

Planned adaptations include:

- Lesion-grouped splitting.
- Comparison of three pretrained architectures.
- Paired experiments with and without augmentation.
- Controlled optimization experiments.
- Expanded class-level evaluation.
- Reusable Python functions and experiment configurations.

Published reference scores are not treated as directly comparable results. This project establishes its own baseline using its fixed evaluation protocol.

## Technology Stack

- Python
- TensorFlow / Keras
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Pillow

Record the actual Python, framework, and hardware versions used for experiments.

## Project Structure

| Path | Current purpose |
| --- | --- |
| `src/prepare_data.py` | Discover metadata and JPEGs; audit, create, reuse, and verify lesion-grouped splits |
| `src/data_pipeline.py` | Decode and resize RGB images; apply model-specific preprocessing; construct datasets; run the Day 1 batch and checkpoint smoke check |
| `src/models.py` | Frozen ImageNet backbones with the shared seven-class head |
| `src/train.py` | Day 2 callbacks, checkpoint selection, training entry point, and experiment log |
| `src/evaluate.py` | Validation predictions and classification metrics |
| `src/visualization.py` | Reusable class, image, and batch figures |
| `src/utils.py` | Seed, runtime information, and JSON serialization |
| `scripts/download_ham10000_isic.py` | Resumable official-data download and metadata conversion |
| `notebooks/01_eda.ipynb` | Interactive Day 1 exploration using functions from `src/` |
| `configs/day2_frozen_no_aug.json` | Shared Day 2 settings for all three architectures |
| `day1_artifacts/` | Generated audit, plots, fixed splits, class mapping, and smoke checkpoint; ignored by Git |
| `outputs/` | Run-specific Day 2 artifacts and `experiment_log.csv`; ignored by Git |
| `requirements.txt`, `.gitignore` | Dependencies and local artifact exclusions |

Later experiment modules and outputs will be added by responsibility when their stages begin. They are not implemented yet.

## Getting Started

### 1. Prepare the environment

Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

GPU support depends on the operating system and installed framework build. Confirm that TensorFlow detects the intended GPU before starting long experiments.

Alternatively, use a Kaggle notebook with the dataset attached. Day 1 works on CPU. Keep the repository under `/kaggle/working/` so `src/` is importable and outputs are writable.

### 2. Run Day 1 preparation

If Kaggle download is unavailable, download the official ISIC release first. The archive is about 2.58 GiB; the script resumes interrupted downloads and verifies all 10,015 JPEGs:

```bash
python -m scripts.download_ham10000_isic --output-dir data
```

From the repository root, with an extracted local dataset:

```bash
python -m src.prepare_data --input-root data --output-dir day1_artifacts
```

`data` must be an existing folder containing the extracted dataset (possibly in nested folders). In PowerShell, check it first with `Test-Path .\data` and locate the CSV with `Get-ChildItem .\data -Recurse -Filter HAM10000_metadata.csv`. Replace example paths with the location of your own downloaded files.

In Kaggle, point at the attached input directory:

```bash
python -m src.prepare_data --input-root /kaggle/input --output-dir /kaggle/working/day1_artifacts
```

The command finds a CSV with `image_id`, `lesion_id`, and `dx`, then searches all JPEG folders. If several metadata files match, use `--metadata /path/to/HAM10000_metadata.csv`. It audits missing labels and files, unreadable JPEGs, duplicate IDs, and conflicting diagnoses within a lesion. Errors are saved in `audit_issues.csv`; the command stops rather than excluding rows silently. It writes `class_distribution.png`, `sample_images.png`, and fixed split files. On rerun, complete saved splits are verified and reused; `--force-resplit` deliberately replaces them. Do not commit dataset images or credentials.

### 3. Run the Day 1 smoke check

```bash
python -m src.data_pipeline --output-dir day1_artifacts --batch-size 32
```

For Kaggle, use `--output-dir /kaggle/working/day1_artifacts`. Use `--batch-size 16` if memory is tight. The command records package versions and GPU visibility, saves `first_batch.png`, checks `(batch, 224, 224, 3)` RGB images and `[0, 255]` pixels, fits a tiny model for **one batch**, and verifies a finite loss and checkpoint reload. This is not a full training experiment. `--train-csv` can select a prepared training CSV at another path. You can also run the same steps interactively in `notebooks/01_eda.ipynb`.

Inspect `audit_summary.json`, `audit_issues.csv`, `split_counts.csv`, `split_manifest.json`, `class_mapping.json`, `environment.json`, and `smoke_result.json` before Day 2. Save the entire `day1_artifacts/` folder so later experiments use the same partitions. If the dataset moves, rerun preparation: it verifies existing split membership and refreshes image paths without changing partition assignments.

Run the small synthetic checks with `python -m unittest discover -s tests -v`. They check split overlap detection, raw image order and range, finite one-batch loss, and checkpoint reload. They do not replace the audit of the real HAM10000 files.

### 4. Run Day 2 frozen baselines

Use an activated PowerShell virtual environment from `C:\Users\OS\OneDrive\Desktop\Skin cancer detection`. The following tiny check uses the saved train/validation rows, random backbone weights, and one batch. It verifies preprocessing, frozen weights, callback order, and checkpoint reload without starting an experiment:

```powershell
python -m src.train --architecture efficientnetb0 --smoke --splits-dir .\day1_artifacts
```

`--all --smoke` checks all three architectures sequentially. After that, explicitly start **one ImageNet-pretrained baseline**:

```powershell
python -m src.train --architecture efficientnetb0 --run --splits-dir .\day1_artifacts --outputs-dir .\outputs
```

Run the other baselines one at a time by replacing `efficientnetb0` with `mobilenetv2` or `resnet50`. To run all three sequentially with identical settings:

```powershell
python -m src.train --all --run --splits-dir .\day1_artifacts --outputs-dir .\outputs
```

The runner never starts full training without `--run`. It reads the existing `train.csv`, `validation.csv`, and `class_mapping.json`; it does not regenerate splits or open `test.csv`. It trains a frozen ImageNet backbone plus Global Average Pooling, Dense(128, ReLU), and Dense(7, softmax), with Adam at `1e-3`, batch size 32, at most 12 epochs, and early stopping after three epochs without improved full-validation macro F1. No augmentation, class weighting, dropout, or L2 is used. If memory requires batch size 16, add `--batch-size 16` to **every** Day 2 run for a fair comparison.

Each `outputs/runs/<run_id>/` contains `config.json`, `best.keras`, `history.csv`, `validation_predictions.csv`, `metrics.json`, and `learning_curves.png`. The validation metrics and probabilities come from `best.keras`. `outputs/experiment_log.csv` has one row per completed run. Run IDs are unique and existing run directories are never replaced. Training requires ImageNet weight downloads on first use. This Windows environment currently detects no TensorFlow GPU, so full runs may be slow locally; Kaggle GPU is an option. Keep test predictions and scores closed until model choices are final.

## Data Splitting and Leakage Prevention

Target partition sizes:

| Partition  | Approximate proportion |
| ---------- | ---------------------: |
| Training   |                    70% |
| Validation |                    15% |
| Test       |                    15% |

Multiple images can belong to the same lesion. Every image with the same `lesion_id` must remain in one partition.

Procedure implemented for Day 1:

1. Verify that each lesion has one consistent diagnosis.
2. Assign images to seven stratified folds while keeping each `lesion_id` together.
3. Choose five folds for training, one for validation, and one for test, balancing image counts and class proportions.
4. Verify zero lesion overlap and all seven classes in each partition.
5. Save assignments and class indices; reuse them for every experiment.

Image proportions may differ from the target because lesions have different numbers of images. Report actual image and lesion counts.

This protocol provides lesion-separated evaluation. It does not establish patient-separated evaluation.

Additional rules:

- Split before augmentation or resampling.
- Fit learned transformations using training data only.
- Calculate class weights using training labels only.
- Preserve natural class distributions in validation and test sets.
- Keep test predictions and performance closed until all decisions are finalized.

## Preprocessing

All images are decoded as RGB and resized to `224 × 224`.

Preprocessing depends on the pretrained architecture:

| Model          | Input preparation                                                                      |
| -------------- | -------------------------------------------------------------------------------------- |
| EfficientNetB0 | Standard Keras implementation expects pixels in the 0–255 range and includes rescaling |
| MobileNetV2    | Use its architecture-specific `preprocess_input`                                       |
| ResNet50       | Use its architecture-specific `preprocess_input`                                       |

Do not apply a universal division by 255 before every architecture.

Maintain a documented preprocessing contract so inference after loading a model matches training. Avoid applying preprocessing twice.

## Data Augmentation

Use the same moderate augmentation policy for all architecture comparisons:

- Random horizontal and vertical flips.
- Rotation up to approximately 20 degrees.
- Zoom up to approximately 10%.
- Mild brightness variation.

Apply random augmentation:

- To training images only.
- Before architecture-specific preprocessing.
- Within a clearly defined pixel range.

Inspect original and augmented examples to confirm that lesions remain visible and colors are not heavily distorted.

Validation and test preprocessing must be deterministic.

Online augmentation creates training variations, not additional independent lesions.

## Initial Model Configuration

All three architectures begin with:

```text
Frozen ImageNet-pretrained backbone
→ Global Average Pooling
→ Dense(128, ReLU)
→ Dense(7, softmax)
```

| Setting                 | Initial value                                 |
| ----------------------- | --------------------------------------------- |
| Seed                    | `42`                                          |
| Image size              | `224 × 224`                                   |
| Batch size              | `32`, or consistently `16` if memory requires |
| Optimizer               | Adam                                          |
| Learning rate           | `1e-3`                                        |
| Loss                    | Sparse categorical cross-entropy              |
| Maximum epochs          | `12`                                          |
| Early-stopping patience | `3`                                           |
| Selection metric        | Validation macro F1                           |

Use integer labels with sparse categorical cross-entropy.

## Experiment Plan

The core plan contains 13 training runs, including three short optimizer comparisons.

### B0 and T1: Architecture and Augmentation

Compare three architectures with and without augmentation:

| Run ID  | Architecture   | Augmentation |
| ------- | -------------- | ------------ |
| `B0_E0` | EfficientNetB0 | Off          |
| `T1_M0` | MobileNetV2    | Off          |
| `T1_R0` | ResNet50       | Off          |
| `T1_EA` | EfficientNetB0 | On           |
| `T1_MA` | MobileNetV2    | On           |
| `T1_RA` | ResNet50       | On           |

Keep the split, resolution, classifier-head design, optimizer, loss, batch size, epoch limit, and selection policy consistent.

Select a candidate using validation macro F1, supporting metrics, and computational cost.

This identifies the strongest frozen-backbone recipe under the tested settings. It does not establish a universally best architecture.

### T2: Class Weighting

Add training-derived inverse-frequency class weights to the selected recipe.

Compare against the corresponding unweighted configuration.

Inspect:

- Macro F1.
- Accuracy.
- Per-class recall and F1.
- Majority-class versus minority-class trade-offs.

Retain weighting only if validation evidence supports it.

### T3: Regularization

| Run    | Change                                         | Comparator                            |
| ------ | ---------------------------------------------- | ------------------------------------- |
| `T3_D` | Add dropout `0.3` after the hidden Dense layer | Best preceding recipe                 |
| `T3_L` | Add hidden-layer L2 coefficient `1e-4`         | Selected dropout or no-dropout recipe |

Keep the hidden layer at 128 units.

Inspect validation metrics alongside learning curves. A reduced training–validation gap alone does not prove improvement.

### T4: Optimizer Comparison

Use a matched short budget of up to five epochs per optimizer:

| Optimizer | Initial learning rate | Additional setting      |
| --------- | --------------------: | ----------------------- |
| Adam      |                `1e-3` | Default moment settings |
| RMSprop   |                `1e-3` | Default settings        |
| SGD       |                `1e-2` | Momentum `0.9`          |

Keep architecture, head, starting weights, data protocol, and budget consistent.

Interpret this as a limited comparison at selected settings. Learning rate and training duration can affect the ranking.

If results are inconclusive, retain Adam and document the limitation.

### T5: Fine-Tuning

1. Load the selected frozen-model checkpoint.
2. Unfreeze the final backbone block or blocks.
3. Keep Batch Normalization layers in inference mode.
4. Recompile with a new optimizer and learning rate around `1e-5`.
5. Train for up to 15 additional epochs.
6. Apply one learning-rate policy.
7. Compare against the frozen checkpoint using validation macro F1.

Suggested callbacks:

| Callback          | Setting                     |
| ----------------- | --------------------------- |
| ReduceLROnPlateau | Factor `0.5`, patience `2`  |
| Early stopping    | Patience `5`                |
| Model checkpoint  | Highest validation macro F1 |

Keep the frozen model if fine-tuning does not improve validation performance.

Fine-tuning and learning-rate decay are tested together in this stage. Their individual contributions are not isolated.

## Fair Comparison Rules

- Reuse fixed partitions and class order.
- Restart recipe comparisons from the same pretrained weights and reproducible head initialization.
- Keep initialization consistent within each architecture.
- Change one factor at a time where practical.
- Record each run’s parent or comparator.
- Keep unsuccessful experiments in the log.
- Fine-tuning deliberately continues from a checkpoint.
- Include preceding training phases when reporting fine-tuning cost.

If validation macro F1 is calculated through a custom callback, execute it before callbacks that monitor `val_macro_f1`.

Ensure validation predictions and labels remain aligned.

## Evaluation

### Primary Metric

**Macro F1** is used for checkpoint and configuration selection because it gives each class equal importance.

### Supporting Metrics

- Top-1 accuracy.
- Balanced accuracy.
- Per-class precision.
- Per-class recall.
- Per-class F1.
- Per-class support.
- Training duration.
- Total and trainable parameter counts.

### Final Test Evaluation

After all decisions are fixed:

1. Evaluate the saved baseline and final model on the same test partition.
2. Generate classification reports.
3. Generate count-based confusion matrices.
4. Generate row-normalized confusion matrices.
5. Inspect representative correct and incorrect predictions.
6. Discuss remaining weaknesses.

Do not select another configuration or tune settings based on test results.

## Experiment Outputs

Save the following for every run:

| File                         | Contents                                                       |
| ---------------------------- | -------------------------------------------------------------- |
| `config.json`                | Settings, seed, architecture, comparator, split identifier     |
| `history.csv`                | Epoch-by-epoch loss, metrics, and learning rate                |
| `best.keras`                 | Selected checkpoint                                            |
| `validation_predictions.csv` | Image IDs, lesion IDs, true labels, predictions, probabilities |
| `metrics.json`               | Validation metrics, timing, and parameter counts               |
| `learning_curves.png`        | Training and validation curves                                 |

Maintain one row per actual run in `experiment_log.csv`.

Suggested columns:

```text
run_id
parent_run_id
architecture
augmentation
class_weights
dense_units
activation
dropout
l2
optimizer
learning_rate
seed
actual_epochs
val_accuracy
val_macro_f1
val_balanced_accuracy
training_seconds
decision
notes
```

## Seven-Day Schedule

| Day | Main work                                                                     | Required outputs                                              |
| --- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1   | Environment setup, data audit, EDA, grouped splits, basic loader, smoke check | Audit summary, EDA figures, splits, class mapping             |
| 2   | Shared training pipeline and three models without augmentation                | Three checkpoints, histories, validation results              |
| 3   | Three models with augmentation and candidate selection                        | Augmentation examples, six-run comparison                     |
| 4   | Class weighting, dropout, and L2                                              | Per-class comparisons and regularization results              |
| 5   | Short optimizer comparisons and recipe selection                              | Optimizer curves and selected frozen configuration            |
| 6   | Fine-tuning, locked decisions, final test evaluation                          | Final model, test metrics, confusion matrices, error analysis |
| 7   | Report, slides, code cleanup, loading/inference verification                  | Complete submission package                                   |

Measure actual epoch time early and adjust the budget consistently.

Prioritize the core experiments and final analysis over optional variations.

## Course Techniques Demonstrated

| Technique                       | Evidence                                                        |
| ------------------------------- | --------------------------------------------------------------- |
| Preprocessing                   | Data audit, image loading, resizing, model-specific preparation |
| Augmentation                    | Paired comparisons across three architectures                   |
| Transfer learning               | ImageNet-pretrained backbones                                   |
| Mini-batch gradient descent     | Batched, shuffled training                                      |
| Dropout                         | T3                                                              |
| L2 regularization               | T3                                                              |
| Adam                            | Baseline and T4                                                 |
| Gradient descent with momentum  | T4                                                              |
| RMSprop                         | T4                                                              |
| Learning-rate decay             | T5                                                              |
| Early stopping                  | Shared training protocol                                        |
| Architecture changes            | Three-backbone comparison                                       |
| Exponentially weighted averages | Explain Adam’s gradient-moment estimates                        |
| Bias correction                 | Explain Adam’s correction of initially biased moment estimates  |

Adam’s internal mechanisms are explained rather than presented as separately tested improvements.

## Results

Experiments have not been reported yet. Replace the placeholders below with measured results.

| Architecture   | Val macro F1: no augmentation | Val macro F1: augmentation |
| -------------- | ----------------------------: | -------------------------: |
| MobileNetV2    |                       Pending |                    Pending |
| EfficientNetB0 |                       Pending |                    Pending |
| ResNet50       |                       Pending |                    Pending |

| Model                | Test accuracy | Test macro F1 | Test balanced accuracy |
| -------------------- | ------------: | ------------: | ---------------------: |
| Baseline             |       Pending |       Pending |                Pending |
| Final selected model |       Pending |       Pending |                Pending |

## Limitations

- Strong class imbalance and small minority classes.
- A limited experiment and computing budget.
- One main split and initial training seed.
- Lesion grouping does not establish patient separation.
- No external-dataset validation.
- Model explanations require supporting evidence.
- The model is an educational prototype and is not clinically validated.

## Final Submission Checklist

- [ ] Runnable source code or notebook.
- [ ] Environment requirements and recorded versions.
- [ ] Fixed split assignments and class mapping.
- [ ] Experiment configurations and log.
- [ ] Selected saved model.
- [ ] Training curves and comparison tables.
- [ ] Baseline-versus-final test evaluation.
- [ ] Per-class metrics and confusion matrices.
- [ ] Representative error analysis.
- [ ] Report and presentation slides.
- [ ] Reference attribution.
- [ ] Verified model loading and inference.
- [ ] No credentials or dataset images committed.

## References

- [HAM10000 dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-HAM10000/data)
- [HAM10000 original paper](https://arxiv.org/abs/1803.10417)
- [Kaggle implementation reference](https://www.kaggle.com/code/foroughgh95/skin-cancer-detection-efficientnetb3-ham10000)
- [Keras Applications](https://keras.io/api/applications/)
- [Keras transfer learning guide](https://keras.io/guides/transfer_learning/)
- [Adam paper](https://arxiv.org/abs/1412.6980)
