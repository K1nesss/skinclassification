from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成简明数据集统计报告。")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    cfg = load_config(parse_args().config)
    df = pd.read_csv(cfg["paths"]["manifest"])
    out = Path(cfg["paths"]["reports_dir"]) / "dataset_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Report",
        "",
        "## Split by Class",
        "",
        "```text",
        df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False),
        "```",
        "",
        "## Source by Class",
        "",
        "```text",
        df.groupby(["source_dataset", "label"]).size().rename("count").reset_index().to_string(index=False),
        "```",
        "",
        "## Image Size",
        "",
        "```text",
        df[["width", "height"]].describe().to_string(),
        "```",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 80)
    print("数据集报告生成完成")
    print("=" * 80)
    print(f"报告路径：{out}")


if __name__ == "__main__":
    main()
