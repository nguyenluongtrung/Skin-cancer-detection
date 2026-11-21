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

## Day 3: Moderate augmentation

Day 3 reuses the saved train/validation splits and class mapping. It starts each frozen backbone from ImageNet weights with the same seeded head initialization as Day 2. Training-only augmentation applies horizontal and vertical flips, rotations within ±20 degrees, zoom within approximately ±10%, and brightness within ±20.4 raw pixel levels. Augmentation acts on RGB pixels in `[0, 255]` before each model's preprocessing. Validation stays deterministic. Inspect each run's `augmentation_examples.png` for lesion visibility.

First run a tiny pipeline and checkpoint check (replace the architecture to check the others):

```powershell
python -m src.train --architecture efficientnetb0 --smoke --config .\configs\day3_frozen_aug.json --splits-dir .\day1_artifacts
```

Run the three full jobs **sequentially** on the local CPU:

```powershell
python -m src.train --architecture efficientnetb0 --run --config .\configs\day3_frozen_aug.json --splits-dir .\day1_artifacts --outputs-dir .\outputs
python -m src.train --architecture mobilenetv2 --run --config .\configs\day3_frozen_aug.json --splits-dir .\day1_artifacts --outputs-dir .\outputs
python -m src.train --architecture resnet50 --run --config .\configs\day3_frozen_aug.json --splits-dir .\day1_artifacts --outputs-dir .\outputs
```

Each run writes a new folder under `outputs/runs/` with configuration, montage, best checkpoint, history, validation probabilities, metrics, and learning curves. Day 2 outputs are preserved. When all three augmented runs are complete, generate the six-run validation table and score changes:

```powershell
python -m src.compare --outputs-dir .\outputs
```

The command writes `day3_six_run_comparison.csv` and `day3_augmentation_changes.csv` under `outputs/`. A change is augmented minus baseline; multiply by 100 for percentage points. The command requires one completed run per configuration and never opens test data.

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

The six Day 2 and Day 3 frozen runs are measured. EfficientNetB0 without augmentation is selected for Day 4; test results remain closed.

| Architecture   | Val macro F1: no augmentation | Val macro F1: augmentation |
| -------------- | ----------------------------: | -------------------------: |
| MobileNetV2    |                        0.5557 |                   0.5087 |
| EfficientNetB0 |                        0.6201 |                   0.5640 |
| ResNet50       |                        0.5443 |                   0.5565 |

### Day 4: staged head experiments

From the repository root, run only the class-weighting stage first:

```powershell
.venv\Scripts\python.exe -m src.train --stage T2_E0_CW --config configs/frozen_head_experiments.json --run
```

Inspect `outputs/day4_comparison.csv`, `outputs/day4_baseline_vs_weighted_per_class.csv`, and `outputs/day4_decisions.json`. The decision file names the validation-selected run. Supply that ID to the dropout stage only after reviewing the first result:

```powershell
.venv\Scripts\python.exe -m src.train --stage T3_E0_D --config configs/frozen_head_experiments.json --parent-run-id SELECTED_RUN_ID --run
```

After reviewing dropout, use its selected run ID for L2:

```powershell
.venv\Scripts\python.exe -m src.train --stage T3_E0_L --config configs/frozen_head_experiments.json --parent-run-id SELECTED_RUN_ID --run
```

Each stage starts from fresh ImageNet weights with the same seeded head initialization and the fixed train/validation split. Class weights come only from training labels and affect only training loss. Validation macro F1 is computed over all validation images in fixed class order and selects the saved best checkpoint. The runner permits one logged run per stage and rejects an unselected parent. Run folders contain configuration, checkpoint, history, validation probabilities and predictions, metrics, runtime, and curves. The comparison CSV records changes from each stage's exact parent; strictly higher validation macro F1 means keep, and ties mean reject. Keras logged loss includes the L2 term when enabled, and weighted and regularized losses are not directly comparable with baseline loss.

Class weighting (`T2_E0_CW`) completed with best-checkpoint validation macro F1 **0.567180** at epoch 8, compared with **0.620086** for the unweighted baseline (change **-0.052907**). Decision: **reject class weights**. The selected parent for dropout was `20261121T040611579432Z_efficientnetb0_edeb0f80`. Its command was:

```powershell
.venv\Scripts\python.exe -m src.train --stage T3_E0_D --config configs/frozen_head_experiments.json --parent-run-id 20261121T040611579432Z_efficientnetb0_edeb0f80 --run
```

The first dropout run (`20261121T124732150308Z_t3_e0_d_6357e54a`) was invalidated: its Functional graph held head dropout in inference mode, giving identical histories and validation probabilities to the baseline. Its files are retained with `invalidated.json`, and its experiment-log row has `run_status=invalid`. After fixing the training-mode call and verifying checkpoint reload, the valid dropout run (`20261121T132354369564Z_t3_e0_d_05eea9d1`) reached best-checkpoint validation macro F1 **0.572482** at epoch 4, a change of **-0.047604** from the baseline. Decision: **reject dropout**. The selected parent for L2 was the unweighted, no-dropout baseline. Its command was:

```powershell
.venv\Scripts\python.exe -m src.train --stage T3_E0_L --config configs/frozen_head_experiments.json --parent-run-id 20261121T040611579432Z_efficientnetb0_edeb0f80 --run
```

L2 (`T3_E0_L`) completed with best-checkpoint validation macro F1 **0.621915** at epoch 12, compared with **0.620086** for the identical recipe without L2 (change **+0.001829**). Decision: **keep L2** under the predefined strict macro-F1 rule. The selected run is `20261121T134510640652Z_t3_e0_l_e8ee6a73`: EfficientNetB0 frozen, no augmentation, no class weighting, no dropout, and L2 coefficient `0.0001` on the hidden Dense layer. The gain is small, and validation accuracy (0.798043 versus 0.799441) and balanced accuracy (0.604547 versus 0.613682) decreased. Logged L2 loss includes the regularization penalty, so compare validation metrics rather than loss values across these runs.

| Modification | Best validation macro F1 | Change from matching parent | Decision |
| --- | ---: | ---: | --- |
| Baseline | 0.620086 | — | Parent |
| Class weights | 0.567180 | -0.052907 | Reject |
| Dropout 0.3 | 0.572482 | -0.047604 | Reject |
| L2 0.0001 | 0.621915 | +0.001829 | Keep |

### Controlled optimizer comparison

The selected longer-budget reference is the Day 4 L2 checkpoint (validation macro F1 **0.621915** after up to 12 epochs). The following commands create fresh ImageNet models; they never resume that checkpoint. Run them from the repository root, one at a time or sequentially:

```powershell
.venv\Scripts\python.exe -m src.train --optimizer-choice adam --config configs/frozen_optimizer_comparison.json --run
.venv\Scripts\python.exe -m src.train --optimizer-choice rmsprop --config configs/frozen_optimizer_comparison.json --run
.venv\Scripts\python.exe -m src.train --optimizer-choice sgd_momentum --config configs/frozen_optimizer_comparison.json --run
```

Each run uses the same saved train/validation splits, integer class mapping, seed 42, training shuffle seed, 224 × 224 RGB input, batch size 32, frozen EfficientNetB0 in inference mode, GAP, Dense(128, ReLU) with kernel L2 `0.0001`, and seven-class softmax. There is no augmentation, class weighting, or dropout. Each optimizer is newly constructed with empty state. The runs complete exactly five epochs with 224 updates per epoch and a constant specified learning rate; only full-validation macro F1 scoring and best-checkpoint saving run as callbacks. The recorded head-weight hash and split-file hashes must match across all three runs before comparison output is written. Logged total loss includes the L2 term.

Adam uses learning rate `0.001`, first-moment coefficient `0.9`, second-moment coefficient `0.999`, and no AMSGrad. RMSprop uses learning rate `0.001`, squared-gradient averaging coefficient `0.9`, zero momentum, and no centering. SGD with momentum uses learning rate `0.01`, momentum `0.9`, and no Nesterov acceleration. Installed Keras defaults supply epsilon for Adam and RMSprop; its resolved value is saved in each run's configuration. Optimizer EMA and optimizer weight decay stay disabled.

SGD momentum maintains an exponentially weighted average of gradients. RMSprop maintains an exponentially weighted average of squared gradients. Adam maintains first and second moments with bias correction. This comparison varies complete optimizer-plus-learning-rate configurations; it does not separately ablate these mechanisms or establish a universal ranking.

Each command creates a unique `outputs/runs/<run_id>/` containing `config.json`, `best.keras`, `history.csv`, `validation_predictions.csv`, `metrics.json`, and `learning_curves.png`, and appends `outputs/experiment_log.csv`. After the third run, `outputs/optimizer_comparison.csv` compares the three fresh best checkpoints, and `outputs/optimizer_decision.json` records a provisional winner and the separate Day 4 reference. The five-epoch winner needs a matched longer-budget confirmation before choosing the next starting point, especially if it scores below the Day 4 checkpoint. No longer run or test evaluation starts automatically.

All three five-epoch runs completed. They used the same split-file hashes and initial head-weight hash, and each completed 1,120 optimizer updates. The table reports each run's best checkpoint, which can precede epoch 5.

| Optimizer | Learning rate | Best epoch | Validation macro F1 | Accuracy | Balanced accuracy | Training seconds | Macro F1 vs fresh Adam |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam | 0.001 | 5 | **0.586173** | 0.778477 | 0.574334 | 675.1 | 0 |
| RMSprop | 0.001 | 2 | 0.540248 | 0.751223 | 0.545301 | 647.7 | -0.045925 |
| SGD with momentum | 0.01 | 5 | 0.541468 | 0.757512 | 0.557551 | 545.7 | -0.044705 |

Adam is the provisional winner for this optimizer-plus-learning-rate comparison. The Day 4 L2 checkpoint scored **0.621915** with a longer, up-to-12-epoch budget; it is a separate reference and remains the selected checkpoint. A matched longer-budget comparison is warranted before using the short-run ranking to choose a different optimizer for the next stage. These results do not show that one optimizer is universally better. Test data and metrics remain closed.

### Day 6 partial fine-tuning

The starting checkpoint is `outputs/runs/20261121T134510640652Z_t3_e0_l_e8ee6a73/best.keras`, selected on Day 4. Its SHA-256 is `930b48be4988dd4e867a763189e71eba5ebb4b52527c5abf2aa87d93395bf4d8`. The validation audit reloaded it without optimizer state and reproduced macro F1 `0.6219153840498015`, exactly matching the saved full-precision value; the acceptance tolerance is `1e-6`. The displayed `0.6219` is rounded. Saved train and validation contain 7,153 and 1,431 images respectively, all seven classes, and no overlapping lesion IDs. The class order is `akiec, bcc, bkl, df, mel, nv, vasc`. Validation stays unshuffled and unaugmented. JPEGs decode to RGB, resize to 224 by 224, and keep 0–255 values for EfficientNet's internal rescaling.

The Day 4 checkpoint has GAP, a 128-unit ReLU layer with kernel L2 `1e-4`, and seven-class softmax. Its weakest validation recall was `df` (0.2500) and `mel` (0.3522); `nv` recall was 0.9322. Across Day 4 epochs, training loss fell from 0.8161 to 0.2630, while validation loss reached 0.7378 at epoch 12 after a lower value of 0.6389 at epoch 3. These observations do not identify a cause for the class errors.

The fine-tuning boundary is EfficientNet layers starting with `block6` or `block7`, plus `top_conv`; every earlier stage and all backbone BatchNormalization layers stay frozen. The exact layer lists and counts are recorded in each new run's `config.json`. The backbone is still called with `training=False`: this controls BatchNormalization and other training-mode behavior, while gradients still flow through trainable convolution kernels. The classification head remains trainable. Training starts from the Day 4 learned weights, recompiles with fresh Adam at `1e-5`, and runs at most 15 additional epochs. Full-validation macro F1 drives checkpointing, learning-rate reduction, and early stopping. No augmentation or class weighting is used.

Run the audit and disposable update check, then explicitly start the full experiment from the repository root:

```powershell
.venv\Scripts\python.exe -m src.train --fine-tune --smoke --config configs/efficientnetb0_partial_fine_tune.json
.venv\Scripts\python.exe -m src.train --fine-tune --run --config configs/efficientnetb0_partial_fine_tune.json
```

The smoke command writes `outputs/day6_preflight.json` with the audit and exact layer lists and parameter counts. The full command creates a unique `outputs/runs/<run_id>/` with the resolved config, parent and split hashes, trainability lists, best checkpoint, epoch history, validation predictions, per-class comparison, confusion matrices, learning curves, metrics, and a keep/reject decision. It appends `outputs/experiment_log.csv`. The frozen checkpoint remains available regardless of the decision. Logged total loss includes the hidden-layer L2 term. Only validation is evaluated; test remains closed.

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
