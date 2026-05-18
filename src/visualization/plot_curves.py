from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.matplotlib_style import setup_plot_style


def plot_training_curves(history_csv: str | Path, output_path: str | Path) -> None:
    setup_plot_style()
    df = pd.read_csv(history_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(df["epoch"], df["val_accuracy"], marker="o")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[2].plot(df["epoch"], df["val_macro_f1"], marker="o")
    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

