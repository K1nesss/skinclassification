from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.datasets.transforms import BackgroundPerturbation
from src.visualization.matplotlib_style import setup_plot_style


def plot_augmentation_examples(manifest_csv: str | Path, output_path: str | Path) -> None:
    setup_plot_style()
    df = pd.read_csv(manifest_csv)
    row = df[df["split"] == "train"].sample(n=1, random_state=7).iloc[0]
    image = Image.open(row["image_path"]).convert("RGB")
    ops = [
        ("Original", lambda x: x),
        ("Random Crop", transforms.RandomResizedCrop(224, scale=(0.75, 1.0))),
        ("Horizontal Flip", transforms.RandomHorizontalFlip(p=1.0)),
        ("Color Jitter", transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04)),
        ("Random Rotation", transforms.RandomRotation(12)),
        ("Background Perturb", BackgroundPerturbation(p=1.0)),
    ]
    fig, axes = plt.subplots(1, len(ops), figsize=(15, 3))
    for ax, (title, op) in zip(axes, ops):
        ax.imshow(op(image))
        ax.set_title(title)
        ax.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

