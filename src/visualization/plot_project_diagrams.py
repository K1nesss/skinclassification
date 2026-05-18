from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.visualization.matplotlib_style import setup_plot_style


def _box(ax, xy, text: str, width: float = 2.3, height: float = 0.55, color: str = "#dbeafe") -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.0,
        edgecolor="#334155",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, start, end) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#334155"})


def _save(fig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_technical_route(output_path: str | Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    items = [
        ("Raw datasets", 0.3, 2.1),
        ("Cleaning and labels", 2.2, 2.1),
        ("Augmentation\nClass balance", 4.1, 2.1),
        ("Multi-model\ntraining", 6.0, 2.1),
        ("Evaluation and\nexplainability", 7.9, 2.1),
    ]
    colors = ["#e0f2fe", "#dcfce7", "#fef3c7", "#ede9fe", "#fee2e2"]
    for (text, x, y), color in zip(items, colors):
        _box(ax, (x, y), text, width=1.65, height=0.8, color=color)
    for x in [1.95, 3.85, 5.75, 7.65]:
        _arrow(ax, (x, 2.5), (x + 0.18, 2.5))
    ax.set_title("Technical Route")
    _save(fig, output_path)


def plot_system_architecture(output_path: str | Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    _box(ax, (0.5, 3.4), "Data layer\nraw / processed / manifest", width=2.5, height=0.9, color="#e0f2fe")
    _box(ax, (3.8, 3.4), "Training layer\nPyTorch models", width=2.5, height=0.9, color="#dcfce7")
    _box(ax, (7.1, 3.4), "Evaluation layer\nmetrics / figures", width=2.5, height=0.9, color="#fef3c7")
    _box(ax, (3.8, 1.4), "Explainability\nGrad-CAM / features", width=2.5, height=0.9, color="#ede9fe")
    _box(ax, (7.1, 1.4), "Streamlit demo\nprediction / Top-3 / CAM", width=2.5, height=0.9, color="#fee2e2")
    _arrow(ax, (3.0, 3.85), (3.8, 3.85))
    _arrow(ax, (6.3, 3.85), (7.1, 3.85))
    _arrow(ax, (5.05, 3.4), (5.05, 2.3))
    _arrow(ax, (6.3, 1.85), (7.1, 1.85))
    ax.set_title("System Architecture")
    _save(fig, output_path)


def plot_data_pipeline(output_path: str | Path) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    steps = [
        ("Read archives\nand folders", 0.4),
        ("Validate image\nand min size", 2.2),
        ("SHA-256\ndeduplication", 4.0),
        ("Map labels\nto 5 classes", 5.8),
        ("Stratified\ntrain/val/test", 7.6),
    ]
    for text, x in steps:
        _box(ax, (x, 2.2), text, width=1.55, height=0.85, color="#e0f2fe")
    for x in [1.95, 3.75, 5.55, 7.35]:
        _arrow(ax, (x, 2.62), (x + 0.22, 2.62))
    _box(ax, (3.0, 0.8), "cleaned_samples.csv\nsplit_samples.csv\nsummary statistics", width=4.0, height=0.9, color="#dcfce7")
    _arrow(ax, (5.0, 2.2), (5.0, 1.7))
    ax.set_title("Data Processing Pipeline")
    _save(fig, output_path)


def generate_project_diagrams(figures_dir: str | Path) -> None:
    figures_dir = Path(figures_dir)
    plot_technical_route(figures_dir / "45_technical_route.png")
    plot_system_architecture(figures_dir / "46_system_architecture.png")
    plot_data_pipeline(figures_dir / "47_data_processing_pipeline.png")
