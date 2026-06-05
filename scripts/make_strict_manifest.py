from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stricter manifest by removing known ambiguous rows.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Input manifest. Defaults to config paths.manifest.")
    parser.add_argument(
        "--output",
        default="data/interim/split_samples_strict_no_scin_ed.csv",
        help="Output manifest path.",
    )
    parser.add_argument("--remove-source", default="scin")
    parser.add_argument("--remove-labels", nargs="+", default=["eczema", "dermatitis"])
    return parser.parse_args()


def print_distribution(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(
        df.groupby(["split", "source_dataset", "label"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "source_dataset", "label"])
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    input_path = Path(args.input or cfg["paths"]["manifest"])
    output_path = Path(args.output)
    df = pd.read_csv(input_path)

    remove_mask = (df["source_dataset"] == args.remove_source) & df["label"].isin(args.remove_labels)
    removed = df[remove_mask].copy()
    kept = df[~remove_mask].copy()
    if removed.empty:
        raise ValueError("No rows matched the requested strict-manifest removal rule.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_path, index=False)

    print("\n严格 manifest 已生成")
    print("=" * 80)
    print(f"输入：{input_path.resolve()}")
    print(f"输出：{output_path.resolve()}")
    print(f"原始样本数：{len(df)}")
    print(f"移除样本数：{len(removed)}")
    print(f"保留样本数：{len(kept)}")
    print("\n移除规则：")
    print(f"  source_dataset = {args.remove_source}")
    print(f"  label in {', '.join(args.remove_labels)}")
    print("\n移除分布：")
    print(
        removed.groupby(["split", "source_dataset", "label"])
        .size()
        .rename("count")
        .reset_index()
        .to_string(index=False)
    )
    print_distribution("保留后类别分布", kept)


if __name__ == "__main__":
    main()
