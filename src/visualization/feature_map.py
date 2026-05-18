from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.visualization.matplotlib_style import setup_plot_style


@torch.no_grad()
def save_feature_map_grid(feature_tensor: torch.Tensor, output_path: str | Path, max_channels: int = 16) -> None:
    setup_plot_style()
    feat = feature_tensor.detach().cpu()[0]
    if feat.ndim != 3:
        return
    channels = min(max_channels, feat.shape[0])
    cols = 4
    rows = (channels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis("off")
        if i < channels:
            axes[i].imshow(feat[i], cmap="viridis")
            axes[i].set_title(f"ch {i}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
