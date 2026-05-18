from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from src.visualization.matplotlib_style import setup_plot_style


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_plot_style()
    counts = df["label"].value_counts().reindex(["acne", "eczema", "dermatitis", "pigmentation", "others"])
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=ax, color="#3b82f6")
    ax.set_title("Class Distribution")
    ax.set_ylabel("Images")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_source_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_plot_style()
    counts = df["source_dataset"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax, color="#10b981")
    ax.set_title("Source Dataset Distribution")
    ax.set_ylabel("Images")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_class_source_heatmap(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_plot_style()
    pivot = pd.pivot_table(
        df,
        index="label",
        columns="source_dataset",
        values="image_path",
        aggfunc="count",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("Class by Source Dataset")
    _save(fig, output_path)


def plot_split_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_plot_style()
    pivot = pd.pivot_table(
        df[df["split"] != "external_test"],
        index="label",
        columns="split",
        values="image_path",
        aggfunc="count",
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Train / Validation / Test Distribution")
    ax.set_ylabel("Images")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_image_size_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df["width"], bins=40, alpha=0.75, label="width")
    axes[0].hist(df["height"], bins=40, alpha=0.75, label="height")
    axes[0].set_title("Image Width / Height")
    axes[0].legend()
    area = df["width"] * df["height"]
    axes[1].hist(area, bins=40, color="#f59e0b")
    axes[1].set_title("Image Area")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, output_path)


def plot_sample_grid(df: pd.DataFrame, output_path: str | Path, n_per_class: int = 9) -> None:
    setup_plot_style()
    labels = ["acne", "eczema", "dermatitis", "pigmentation", "others"]
    fig, axes = plt.subplots(len(labels), n_per_class, figsize=(n_per_class * 1.5, len(labels) * 1.6))
    for row_idx, label in enumerate(labels):
        sample = df[df["label"] == label].sample(
            n=min(n_per_class, (df["label"] == label).sum()),
            random_state=42,
        )
        for col_idx in range(n_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(sample):
                img = Image.open(sample.iloc[col_idx]["image_path"]).convert("RGB")
                ax.imshow(img)
            if col_idx == 0:
                ax.set_ylabel(label)
    _save(fig, output_path)


def plot_source_sample_grid(df: pd.DataFrame, output_path: str | Path, n_per_source: int = 6) -> None:
    setup_plot_style()
    sources = sorted(df["source_dataset"].dropna().unique().tolist())
    if not sources:
        return
    fig, axes = plt.subplots(len(sources), n_per_source, figsize=(n_per_source * 1.6, len(sources) * 1.8))
    axes = pd.Series(axes.flatten() if hasattr(axes, "flatten") else [axes])
    for ax in axes:
        ax.axis("off")
    pos = 0
    for source in sources:
        sample = df[df["source_dataset"] == source].sample(
            n=min(n_per_source, (df["source_dataset"] == source).sum()),
            random_state=42,
        )
        for col_idx in range(n_per_source):
            ax = axes.iloc[pos]
            pos += 1
            if col_idx < len(sample):
                img = Image.open(sample.iloc[col_idx]["image_path"]).convert("RGB")
                ax.imshow(img)
                ax.set_title(sample.iloc[col_idx]["label"], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(source)
    _save(fig, output_path)


def plot_split_sample_grid(df: pd.DataFrame, output_path: str | Path, n_per_split: int = 6) -> None:
    setup_plot_style()
    splits = [s for s in ["train", "val", "test", "external_test"] if (df["split"] == s).any()]
    if not splits:
        return
    fig, axes = plt.subplots(len(splits), n_per_split, figsize=(n_per_split * 1.6, len(splits) * 1.8))
    axes = pd.Series(axes.flatten() if hasattr(axes, "flatten") else [axes])
    for ax in axes:
        ax.axis("off")
    pos = 0
    for split in splits:
        sample = df[df["split"] == split].sample(n=min(n_per_split, (df["split"] == split).sum()), random_state=43)
        for col_idx in range(n_per_split):
            ax = axes.iloc[pos]
            pos += 1
            if col_idx < len(sample):
                img = Image.open(sample.iloc[col_idx]["image_path"]).convert("RGB")
                ax.imshow(img)
                ax.set_title(sample.iloc[col_idx]["label"], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(split)
    _save(fig, output_path)


def plot_cleaning_summary(interim_dir: str | Path, output_path: str | Path) -> None:
    setup_plot_style()
    path = Path(interim_dir) / "cleaning_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    labels = {
        "kept": "Kept",
        "duplicate": "Duplicate",
        "invalid_image": "Invalid",
        "too_small": "Too small",
    }
    df["status"] = df["status"].map(labels).fillna(df["status"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(df["status"], df["count"], color=["#10b981", "#f97316", "#ef4444", "#6366f1"][: len(df)])
    ax.set_title("Data Cleaning Summary")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def generate_dataset_figures(manifest_csv: str | Path, figures_dir: str | Path) -> None:
    df = pd.read_csv(manifest_csv)
    figures_dir = Path(figures_dir)
    plot_class_distribution(df, figures_dir / "01_class_distribution_bar.png")
    plot_source_distribution(df[df["split"] != "external_test"], figures_dir / "02_source_dataset_distribution.png")
    plot_class_source_heatmap(df, figures_dir / "03_class_source_heatmap.png")
    plot_split_distribution(df, figures_dir / "04_split_distribution.png")
    plot_image_size_distribution(df, figures_dir / "05_image_size_distribution.png")
    plot_cleaning_summary(Path(manifest_csv).parent, figures_dir / "06_data_cleaning_summary.png")
    plot_sample_grid(df[df["split"] == "train"], figures_dir / "07_class_sample_grid.png")
    plot_source_sample_grid(df, figures_dir / "43_source_sample_grid.png")
    plot_split_sample_grid(df, figures_dir / "44_split_sample_grid.png")
