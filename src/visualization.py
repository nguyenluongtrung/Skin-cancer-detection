"""Small, reusable Day 1 plots; no plotting runs during import."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save figures without a display server.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")


def plot_dataset_overview(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = frame["dx"].value_counts().reindex(CLASSES, fill_value=0)
    lesions = frame.drop_duplicates("lesion_id")["dx"].value_counts().reindex(CLASSES, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=axes[0], color="steelblue")
    axes[0].set(title="Images by diagnosis", xlabel="dx", ylabel="Images")
    sns.barplot(x=lesions.index, y=lesions.values, ax=axes[1], color="darkseagreen")
    axes[1].set(title="Lesions by diagnosis", xlabel="dx", ylabel="Lesions")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    valid = frame.dropna(subset=["image_path", "dx"])
    selected = valid.groupby("dx", group_keys=False).head(2)
    fig, axes = plt.subplots(2, 7, figsize=(15, 5))
    for ax in axes.flat:
        ax.axis("off")
    for col, diagnosis in enumerate(CLASSES):
        for row_index, (_, row) in enumerate(selected[selected["dx"] == diagnosis].head(2).iterrows()):
            try:
                with Image.open(row["image_path"]) as image:
                    axes[row_index, col].imshow(image.convert("RGB"))
                axes[row_index, col].set_title(f"{diagnosis}: {row['image_id']}", fontsize=8)
            except (OSError, ValueError, UnidentifiedImageError):
                axes[row_index, col].set_title(f"Unreadable: {diagnosis}", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_images.png", dpi=150)
    plt.close(fig)


def plot_first_batch(images: np.ndarray, labels: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].astype(np.uint8))
            ax.set_title(f"class index {int(labels[i])}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "first_batch.png", dpi=150)
    plt.close(fig)


def plot_augmentation_pairs(original: np.ndarray, augmented: np.ndarray, labels: np.ndarray, path: Path) -> None:
    """Show one original and one augmented training image for each class."""
    if original.shape != augmented.shape or len(original) != len(labels):
        raise ValueError("Original, augmented, and label counts must match.")
    fig, axes = plt.subplots(2, len(labels), figsize=(2.2 * len(labels), 5))
    for column, label in enumerate(labels):
        for row, batch in enumerate((original, augmented)):
            axes[row, column].imshow(np.clip(batch[column], 0, 255).astype(np.uint8))
            axes[row, column].set_title(f"{CLASSES[int(label)]}: {'original' if row == 0 else 'augmented'}", fontsize=9)
            axes[row, column].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curves(history: pd.DataFrame, path: Path) -> None:
    """Save training loss and full-validation macro F1 across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["epoch"], history["loss"], marker="o", label="train loss")
    if "val_loss" in history:
        axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Sparse cross-entropy", title="Learning loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_f1"], marker="o", color="teal")
    axes[1].set(xlabel="Epoch", ylabel="Macro F1", title="Full validation set", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
