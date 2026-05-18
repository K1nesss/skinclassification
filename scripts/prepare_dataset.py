from __future__ import annotations

import argparse
import ast
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.io import ensure_project_dirs, load_config
from src.utils.seed import seed_everything

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["acne", "eczema", "dermatitis", "pigmentation", "others"]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned split manifest.")
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def map_label(source: str, raw_label: str) -> str:
    text = normalize_name(raw_label)
    if "acne" in text or "rosacea" in text:
        return "acne"
    if source == "skindisnet":
        if "eczema" in text or re.search(r"\bec\b", text):
            return "eczema"
        if any(k in text for k in ["atopic dermatitis", "contact dermatitis", "seborrheic dermatitis"]):
            return "dermatitis"
        if any(k in text for k in ["ad", "cd", "sd"]):
            return "dermatitis"
        return "others"
    if "eczema" in text:
        return "eczema"
    if any(k in text for k in ["dermatitis", "poison ivy"]):
        return "dermatitis"
    if any(k in text for k in ["pigment", "melasma", "lentigo", "dark spot"]):
        return "pigmentation"
    return "others"


def iter_zip_images(zip_path: Path, source: str, prefix_filter: str | None = None):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            suffix = Path(info.filename).suffix.lower()
            if info.is_dir() or suffix not in IMAGE_EXTS:
                continue
            if prefix_filter and not info.filename.startswith(prefix_filter):
                continue
            parts = info.filename.replace("\\", "/").split("/")
            raw_label = parts[0]
            if source == "skindisnet" and len(parts) > 1:
                raw_label = parts[1]
            yield source, raw_label, f"{zip_path}!{info.filename}", zf.read(info)


def iter_dir_images(root: Path, source: str):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            rel = path.relative_to(root)
            raw_label = rel.parts[0] if rel.parts else root.name
            yield source, raw_label, str(path), path.read_bytes()


def best_scin_label(weighted_label: str) -> str:
    try:
        labels = ast.literal_eval(weighted_label)
    except Exception:
        return "unknown"
    if not isinstance(labels, dict) or not labels:
        return "unknown"
    ranked = sorted(labels.items(), key=lambda item: float(item[1]), reverse=True)
    focused = []
    for diagnosis, score in ranked:
        mapped = map_label("scin", diagnosis)
        if mapped != "others":
            focused.append((diagnosis, score))
    return str(focused[0][0] if focused else ranked[0][0])


def iter_scin_images(scin_dir: Path):
    dataset_dir = scin_dir / "dataset"
    cases_path = dataset_dir / "scin_cases.csv"
    labels_path = dataset_dir / "scin_labels.csv"
    if not cases_path.exists() or not labels_path.exists():
        yield from iter_dir_images(scin_dir, "scin")
        return

    cases = pd.read_csv(cases_path)
    labels = pd.read_csv(labels_path)
    merged = cases.merge(labels[["case_id", "weighted_skin_condition_label"]], on="case_id", how="left")
    for _, row in merged.iterrows():
        raw_label = best_scin_label(str(row.get("weighted_skin_condition_label", "")))
        for col in ["image_1_path", "image_2_path", "image_3_path"]:
            rel = row.get(col)
            if pd.isna(rel) or not str(rel).strip():
                continue
            image_path = scin_dir / str(rel).replace("/", "\\")
            if not image_path.exists():
                image_path = scin_dir / str(rel).replace("\\", "/")
            if image_path.exists():
                yield "scin", raw_label, str(image_path), image_path.read_bytes()


def discover_raw_images(raw_dir: Path, use_sources: set[str] | None = None):
    dermnet_dir = raw_dir / "Dermnet"
    if use_sources is None or "dermnet" in use_sources:
        for zip_name in ["train.zip", "test.zip"]:
            zip_path = dermnet_dir / zip_name
            if zip_path.exists():
                yield from iter_zip_images(zip_path, "dermnet")

    skindisnet_zip = raw_dir / "SkinDisNet_2.zip"
    if (use_sources is None or "skindisnet" in use_sources) and skindisnet_zip.exists():
        yield from iter_zip_images(skindisnet_zip, "skindisnet", prefix_filter="Preprocessed/")

    mendeley_dir = raw_dir / "Mendeley Skin Disease Classification Dataset"
    if (use_sources is None or "mendeley" in use_sources) and mendeley_dir.exists():
        for zip_path in mendeley_dir.glob("*.zip"):
            yield from iter_zip_images(zip_path, "mendeley")
        for child in mendeley_dir.iterdir():
            if child.is_dir():
                yield from iter_dir_images(child, "mendeley")

    if use_sources is None or "scin" in use_sources:
        for scin_name in ["SCIN", "scin"]:
            scin_dir = raw_dir / scin_name
            if scin_dir.exists():
                yield from iter_scin_images(scin_dir)


def safe_image_record(
    cfg: dict,
    source: str,
    raw_label: str,
    original_path: str,
    data: bytes,
    seen_hashes: set[str],
) -> tuple[dict | None, str]:
    sha = hashlib.sha256(data).hexdigest()
    if sha in seen_hashes:
        return None, "duplicate"
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        width, height = image.size
    except Exception:
        return None, "invalid_image"
    if min(width, height) < int(cfg["data"]["min_size"]):
        return None, "too_small"

    label = map_label(source, raw_label)
    label_id = CLASS_NAMES.index(label)
    out_dir = Path(cfg["paths"]["processed_image_dir"]) / source / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sha[:16]}.jpg"
    if not out_path.exists():
        image.save(out_path, quality=95)
    seen_hashes.add(sha)
    return {
        "source_dataset": source,
        "original_label": raw_label,
        "label": label,
        "label_id": label_id,
        "original_path": original_path,
        "image_path": str(out_path),
        "width": width,
        "height": height,
        "sha256": sha,
    }, "kept"


def assign_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = int(cfg["project"]["seed"])
    external_sources = set(cfg["data"]["split_external_sources"])
    df = df.copy()
    df["split"] = ""
    external_mask = df["source_dataset"].isin(external_sources)
    df.loc[external_mask, "split"] = "external_test"

    main = df[~external_mask].copy()
    if main.empty:
        return df
    val_ratio = float(cfg["data"]["val_ratio"])
    test_ratio = float(cfg["data"]["test_ratio"])
    temp_ratio = val_ratio + test_ratio

    train_idx, temp_idx = train_test_split(
        main.index,
        test_size=temp_ratio,
        random_state=seed,
        stratify=main["label_id"],
    )
    temp = main.loc[temp_idx]
    val_share = val_ratio / temp_ratio
    val_idx, test_idx = train_test_split(
        temp.index,
        test_size=1 - val_share,
        random_state=seed,
        stratify=temp["label_id"],
    )
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def apply_optional_cap(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cap = cfg["data"].get("max_samples_per_class")
    if not cap:
        return apply_majority_downsample(df, cfg)
    capped = []
    seed = int(cfg["project"]["seed"])
    for _, group in df.groupby(["source_dataset", "label"], sort=False):
        capped.append(group.sample(n=min(len(group), int(cap)), random_state=seed))
    return apply_majority_downsample(pd.concat(capped, ignore_index=True), cfg)


def apply_majority_downsample(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    settings = cfg["data"].get("downsample_majority") or {}
    if not settings.get("enabled", False):
        return df

    target_label = str(settings.get("label", "others"))
    target_split = str(settings.get("split", "train"))
    strategy = str(settings.get("strategy", "max_per_source"))
    max_per_source = int(settings.get("max_per_source", 3000))
    ratio = float(settings.get("ratio_to_max_non_majority", 2.0))
    external_sources = set(cfg["data"].get("split_external_sources") or [])
    skip_external = bool(settings.get("skip_external_sources", True))
    seed = int(cfg["project"]["seed"])
    df = df.copy()

    if strategy == "ratio_to_max_non_majority":
        train_mask = df["split"].eq(target_split)
        non_majority_counts = df[train_mask & ~df["label"].eq(target_label)]["label"].value_counts()
        if non_majority_counts.empty:
            return df
        target_total = int(round(non_majority_counts.max() * ratio))
        majority_idx = df[train_mask & df["label"].eq(target_label)].index
        if len(majority_idx) <= target_total:
            return df
        keep_majority = df.loc[majority_idx].sample(n=target_total, random_state=seed).index
        drop_idx = majority_idx.difference(keep_majority)
        return df.drop(index=drop_idx).reset_index(drop=True)

    parts = []
    for (split, source, label), group in df.groupby(["split", "source_dataset", "label"], sort=False):
        should_cap = (
            split == target_split
            and label == target_label
            and (not skip_external or source not in external_sources)
        )
        if should_cap and len(group) > max_per_source:
            parts.append(group.sample(n=max_per_source, random_state=seed))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True)


def write_reports(df: pd.DataFrame, cfg: dict, status_counts: dict[str, int] | None = None) -> None:
    interim = Path(cfg["paths"]["interim_dir"])
    df.to_csv(interim / "split_samples.csv", index=False)
    df.to_csv(interim / "cleaned_samples.csv", index=False)
    df.groupby(["label"]).size().rename("count").reset_index().to_csv(
        interim / "class_distribution.csv", index=False
    )
    df.groupby(["source_dataset", "label"]).size().rename("count").reset_index().to_csv(
        interim / "source_class_distribution.csv", index=False
    )
    df.groupby(["split", "label"]).size().rename("count").reset_index().to_csv(
        interim / "split_distribution.csv", index=False
    )
    if status_counts is not None:
        pd.Series(status_counts, name="count").rename_axis("status").reset_index().to_csv(
            interim / "cleaning_summary.csv", index=False
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["seed"]))

    raw_dir = Path(cfg["paths"]["raw_dir"])
    print_section("数据集准备")
    print(f"配置文件：{args.config}")
    print(f"原始数据目录：{raw_dir.resolve()}")
    print(f"处理后图片目录：{Path(cfg['paths']['processed_image_dir']).resolve()}")
    print(f"样本清单输出：{Path(cfg['paths']['manifest']).resolve()}")
    print(f"统一类别：{', '.join(CLASS_NAMES)}")
    print(f"外部测试来源：{', '.join(cfg['data']['split_external_sources'])}")
    print(f"内部划分比例：训练集={1 - float(cfg['data']['val_ratio']) - float(cfg['data']['test_ratio']):.2f}，"
          f"验证集={float(cfg['data']['val_ratio']):.2f}，测试集={float(cfg['data']['test_ratio']):.2f}")
    print("原始数据存在性检查：")
    print(f"  Dermnet train.zip：{(raw_dir / 'Dermnet' / 'train.zip').exists()}")
    print(f"  Dermnet test.zip：{(raw_dir / 'Dermnet' / 'test.zip').exists()}")
    print(f"  SkinDisNet_2.zip：{(raw_dir / 'SkinDisNet_2.zip').exists()}")
    print(f"  Mendeley 目录：{(raw_dir / 'Mendeley Skin Disease Classification Dataset').exists()}")
    print(f"  SCIN 目录：{(raw_dir / 'SCIN').exists() or (raw_dir / 'scin').exists()}")

    seen_hashes: set[str] = set()
    records = []
    status_counts: dict[str, int] = {"kept": 0, "duplicate": 0, "invalid_image": 0, "too_small": 0}
    source_seen: dict[str, int] = {}
    use_sources_cfg = cfg["data"].get("use_sources")
    use_sources = set(use_sources_cfg) if use_sources_cfg else None
    print(f"启用数据来源：{', '.join(sorted(use_sources)) if use_sources else 'all'}")
    for item in tqdm(discover_raw_images(raw_dir, use_sources=use_sources), desc="Scanning images"):
        source = item[0]
        source_seen[source] = source_seen.get(source, 0) + 1
        record, status = safe_image_record(cfg, *item, seen_hashes=seen_hashes)
        status_counts[status] = status_counts.get(status, 0) + 1
        if record:
            records.append(record)
    if not records:
        raise RuntimeError(f"No valid images found under {raw_dir}")
    df = pd.DataFrame(records)
    print_section("原始扫描统计")
    print("各来源候选图片数：")
    print(pd.Series(source_seen, name="candidate_images").sort_index().to_string())
    print("\n清洗状态统计：")
    status_names = {
        "kept": "保留",
        "duplicate": "重复图片",
        "invalid_image": "损坏/无法打开",
        "too_small": "尺寸过小",
    }
    status_series = pd.Series(
        {status_names.get(k, k): v for k, v in status_counts.items()},
        name="数量",
    )
    print(status_series.to_string())
    print(f"\n可用图片数（截断前）：{len(df)}")

    df = assign_splits(df, cfg)
    before_cap_count = len(df)
    df = apply_optional_cap(df, cfg)
    downsample_settings = cfg["data"].get("downsample_majority") or {}
    if downsample_settings.get("enabled", False):
        print(
            f"已启用 majority downsample：label={downsample_settings.get('label', 'others')}，"
            f"split={downsample_settings.get('split', 'train')}，"
            f"strategy={downsample_settings.get('strategy', 'max_per_source')}，"
            f"max_per_source={downsample_settings.get('max_per_source', 3000)}，"
            f"ratio_to_max_non_majority={downsample_settings.get('ratio_to_max_non_majority', 2.0)}，"
            f"skip_external_sources={downsample_settings.get('skip_external_sources', True)}；"
            f"保留 {len(df)} / {before_cap_count} 张图片。"
        )
    if cfg["data"].get("max_samples_per_class"):
        print(f"已应用 max_samples_per_class={cfg['data']['max_samples_per_class']}；保留 {len(df)} 张图片。")
    write_reports(df, cfg, status_counts=status_counts)
    print_section("最终样本清单统计")
    print("各划分的类别数量：")
    print(df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False))
    print("\n各数据源的类别数量：")
    print(df.groupby(["source_dataset", "label"]).size().rename("count").reset_index().to_string(index=False))
    print("\n总体类别分布：")
    print(df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    external_count = int((df["split"] == "external_test").sum())
    print(f"\n外部测试集图片数：{external_count}")
    print(f"样本清单总行数：{len(df)}")
    print(f"已写入样本清单：{Path(cfg['paths']['manifest']).resolve()}")
    print(f"已写入中间统计文件目录：{Path(cfg['paths']['interim_dir']).resolve()}")


if __name__ == "__main__":
    main()
