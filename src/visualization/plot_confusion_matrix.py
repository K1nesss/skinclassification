from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.visualization.matplotlib_style import setup_plot_style


def plot_confusion_matrix(cm, class_names: list[str], output_path: str | Path, normalize: bool = False) -> None:
    setup_plot_style()
    matrix = np.asarray(cm, dtype=float)
    if normalize:
        denom = matrix.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        matrix = matrix / denom
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

