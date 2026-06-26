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
    parser = argparse.ArgumentParser(
        description="Build train/val/test manifests from the manually curated data/new_1 folder."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--input", default=None, help="Base split manifest. Defaults to config paths.manifest.")
    parser.add_argument("--curated-dir", default="data/new_1")
    parser.add_argument("--output", default="data/interim/split_samples_manual_new1.csv")
    parser.add_argument(
        "--strict-output",
        default="data/interim/split_samples_manual_new1_strict_no_scin_ed.csv",
    )
    parser.add_argument("--removed-output", default="data/interim/manual_new1_removed_samples.csv")
    parser.add_argument("--remove-source", default="scin")
    parser.add_argument("--remove-labels", nargs="+", default=["eczema", "dermatitis"])
    return parser.parse_args()


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_count_by_label(curated_dir: Path, class_names: list[str]) -> pd.Series:
    counts = {}
    for label in class_names:
        label_dir = curated_dir / label
        counts[label] = sum(1 for path in label_dir.rglob("*") if path.is_file()) if label_dir.exists() else 0
    return pd.Series(counts, name="file_count")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = cfg["project"]["class_names"]

    input_path = Path(args.input or cfg["paths"]["manifest"])
    curated_dir = Path(args.curated_dir)
    output_path = Path(args.output)
    strict_output_path = Path(args.strict_output)
    removed_output_path = Path(args.removed_output)

    df = pd.read_csv(input_path)
    kept_rows: list[pd.Series] = []
    removed_rows: list[dict] = []

    normal_locations: dict[str, set[str]] = {}
    deleted_locations: dict[str, set[str]] = {}
    for directory in sorted(path for path in curated_dir.iterdir() if path.is_dir()):
        target = deleted_locations if directory.name.startswith("del") else normal_locations
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                target.setdefault(file_path.name, set()).add(directory.name)

    for _, row in df.iterrows():
        label = str(row["label"])
        basename = Path(str(row["new_image_path"])).name
        curated_path = curated_dir / label / basename
        if curated_path.exists():
            item = row.copy()
            item["new_image_path"] = relpath(curated_path)
            item["image_path"] = relpath(curated_path)
            kept_rows.append(item)
            continue

        if basename in deleted_locations:
            reason = "manual_deleted"
            locations = ",".join(sorted(deleted_locations[basename]))
        elif basename in normal_locations:
            reason = "label_dir_mismatch"
            locations = ",".join(sorted(normal_locations[basename]))
        else:
            reason = "missing_from_new1"
            locations = ""
        removed_rows.append({**row.to_dict(), "manual_remove_reason": reason, "new1_locations": locations})

    kept = pd.DataFrame(kept_rows)
    removed = pd.DataFrame(removed_rows)

    strict_mask = (kept["source_dataset"] == args.remove_source) & kept["label"].isin(args.remove_labels)
    strict = kept[~strict_mask].copy()
    strict_removed = kept[strict_mask].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    strict_output_path.parent.mkdir(parents=True, exist_ok=True)
    removed_output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_path, index=False)
    strict.to_csv(strict_output_path, index=False)
    removed.to_csv(removed_output_path, index=False)

    print("\n人工清洗 manifest 已生成")
    print("=" * 80)
    print(f"基础 manifest：{input_path.resolve()}")
    print(f"人工清洗目录：{curated_dir.resolve()}")
    print(f"full manifest：{output_path.resolve()}")
    print(f"strict manifest：{strict_output_path.resolve()}")
    print(f"剔除明细：{removed_output_path.resolve()}")
    print(f"原始样本数：{len(df)}")
    print(f"人工删除样本数：{len(removed)}")
    print(f"full 保留样本数：{len(kept)}")
    print(f"strict 额外移除样本数：{len(strict_removed)}")
    print(f"strict 保留样本数：{len(strict)}")

    print("\n人工删除原因分布：")
    if removed.empty:
        print("无")
    else:
        print(removed["manual_remove_reason"].value_counts().to_string())

    print("\nnew_1 正常目录文件数 vs full manifest 标签数：")
    actual_counts = file_count_by_label(curated_dir, class_names)
    manifest_counts = kept["label"].value_counts().reindex(class_names, fill_value=0).rename("manifest_count")
    print(pd.concat([actual_counts, manifest_counts], axis=1).to_string())

    print("\nfull 划分分布：")
    print(
        kept.groupby(["split", "label"])
        .size()
        .rename("count")
        .reset_index()
        .to_string(index=False)
    )

    print("\nstrict 额外移除分布：")
    if strict_removed.empty:
        print("无")
    else:
        print(
            strict_removed.groupby(["split", "source_dataset", "label"])
            .size()
            .rename("count")
            .reset_index()
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
