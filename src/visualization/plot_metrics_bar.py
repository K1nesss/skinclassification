from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.visualization.matplotlib_style import setup_plot_style


def plot_per_class_metrics(per_class: dict, output_path: str | Path) -> None:
    setup_plot_style()
    rows = []
    for label, metrics in per_class.items():
        for key in ["precision", "recall", "f1"]:
            rows.append({"class": label, "metric": key, "value": metrics[key]})
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, group in df.groupby("metric"):
        ax.plot(group["class"], group["value"], marker="o", label=metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

