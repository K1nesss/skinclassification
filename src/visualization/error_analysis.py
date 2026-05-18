from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.matplotlib_style import setup_plot_style


def plot_confidence_distribution(predictions_csv: str | Path, output_path: str | Path) -> None:
    setup_plot_style()
    df = pd.read_csv(predictions_csv)
    if "confidence" not in df or "correct" not in df:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in df.groupby("correct"):
        ax.hist(group["confidence"], bins=20, alpha=0.6, label=f"correct={label}")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Samples")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

