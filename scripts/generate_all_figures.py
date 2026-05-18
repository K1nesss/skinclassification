from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_config
from src.visualization.plot_aug_examples import plot_augmentation_examples
from src.visualization.plot_dataset_stats import generate_dataset_figures
from src.visualization.plot_project_diagrams import generate_project_diagrams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset and augmentation figures.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest = Path(cfg["paths"]["manifest"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    print("\n" + "=" * 80)
    print("生成数据集与增强可视化图表")
    print("=" * 80)
    print(f"配置文件：{args.config}")
    print(f"样本清单：{manifest.resolve()}")
    print(f"图表输出目录：{figures_dir.resolve()}")
    generate_dataset_figures(manifest, figures_dir)
    generate_project_diagrams(figures_dir)
    print("已生成：类别分布、数据来源分布、类别-来源热力图、划分分布、图片尺寸分布、数据清洗统计、类别样本九宫格、数据源样本图、划分样本图")
    plot_augmentation_examples(manifest, figures_dir / "08_augmentation_examples.png")
    print(f"已生成：{figures_dir / '08_augmentation_examples.png'}")
    plot_augmentation_examples(manifest, figures_dir / "09_background_perturbation_examples.png")
    print(f"已生成：{figures_dir / '09_background_perturbation_examples.png'}")
    print("新增 image 可视化：")
    print(f"  每类样本图：{figures_dir / '07_class_sample_grid.png'}")
    print(f"  每个数据源样本图：{figures_dir / '43_source_sample_grid.png'}")
    print(f"  每个划分样本图：{figures_dir / '44_split_sample_grid.png'}")
    print("新增报告流程图：")
    print(f"  技术路线图：{figures_dir / '45_technical_route.png'}")
    print(f"  系统架构图：{figures_dir / '46_system_architecture.png'}")
    print(f"  数据处理流程图：{figures_dir / '47_data_processing_pipeline.png'}")
    print(f"图表生成完成：{figures_dir.resolve()}")


if __name__ == "__main__":
    main()
