from __future__ import annotations

import argparse
import ast
import hashlib
import io
import math
import re
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.io import load_config

try:
    import cv2
except Exception:  # pragma: no cover - fallback for minimal local environments
    cv2 = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "acne_rosacea",
    "eczema",
    "dermatitis",
    "pigmentation_disorder",
    "vitiligo",
    "nail_psoriasis",
    "psoriasis_lichen_planus",
    "fungal_infection",
    "seborrheic_keratosis_benign_tumor",
    "viral_warts",
]

TARGET_SPECS: dict[str, list[tuple[str, str]]] = {
    "acne_rosacea": [
        ("dermnet", "Acne and Rosacea Photos"),
        ("mendeley", "acne"),
        ("scin", "Acne"),
    ],
    "eczema": [
        ("scin", "Eczema"),
        ("dermnet", "Eczema Photos"),
        ("skindisnet", "Eczema (EC)"),
    ],
    "dermatitis": [
        ("scin", "Allergic Contact Dermatitis"),
        ("dermnet", "Atopic Dermatitis Photos"),
        ("skindisnet", "Contact Dermatitis (CD)"),
        ("dermnet", "Poison Ivy Photos and other Contact Dermatitis"),
        ("scin", "Irritant Contact Dermatitis"),
        ("scin", "Acute dermatitis, NOS"),
        ("scin", "CD Contact dermatitis"),
    ],
    "pigmentation_disorder": [
        ("dermnet", "Light Diseases and Disorders of Pigmentation"),
        ("mendeley", "hyperpigmentation"),
    ],
    "vitiligo": [
        ("mendeley", "Vitiligo"),
    ],
    "nail_psoriasis": [
        ("mendeley", "Nail psoriasis"),
    ],
    "psoriasis_lichen_planus": [
        ("dermnet", "Psoriasis pictures Lichen Planus and related diseases"),
    ],
    "fungal_infection": [
        ("dermnet", "Tinea Ringworm Candidiasis and other Fungal Infections"),
        ("dermnet", "Nail Fungus and other Nail Disease"),
        ("skindisnet", "Tinea Corporis (TC)"),
    ],
    "seborrheic_keratosis_benign_tumor": [
        ("dermnet", "Seborrheic Keratoses and other Benign Tumors"),
    ],
    "viral_warts": [
        ("dermnet", "Warts Molluscum and other Viral Infections"),
    ],
}

REASON_CN = {
    "kept": "保留",
    "duplicate": "重复图片",
    "invalid_image": "损坏或无法打开",
    "too_small": "尺寸过小",
    "black_occlusion_large": "黑色遮挡区域过大",
    "mosaic_or_pixelated": "马赛克/像素块占比过大",
    "lesion_not_visible": "病灶区域几乎不可见",
    "overexposed": "图片过曝",
    "too_dark": "图片过暗",
    "severe_blur": "严重模糊",
}

REASON_TYPE = {
    "duplicate": "automatic",
    "invalid_image": "automatic",
    "too_small": "automatic",
    "black_occlusion_large": "automatic",
    "overexposed": "automatic",
    "too_dark": "automatic",
    "severe_blur": "automatic",
    "mosaic_or_pixelated": "heuristic",
    "lesion_not_visible": "heuristic",
}

REASON_PRIORITY = [
    "invalid_image",
    "duplicate",
    "too_small",
    "black_occlusion_large",
    "overexposed",
    "too_dark",
    "severe_blur",
    "mosaic_or_pixelated",
    "lesion_not_visible",
]


@dataclass
class RawItem:
    source_dataset: str
    original_label: str
    target_class: str
    original_path: str
    data: bytes
    suffix: str


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the new 10-class cleaned dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--clean", action="store_true", help="删除并重建 data/new、data/pass、data/processed 和 data/interim")
    parser.add_argument("--max-per-class", type=int, default=None, help="仅用于调试：每个目标类别最多处理多少张候选图片")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def safe_name(text: str, max_len: int = 70) -> str:
    name = re.sub(r"[^a-zA-Z0-9]+", "_", str(text)).strip("_").lower()
    return (name or "unknown")[:max_len]


def map_focus_label(raw_label: str) -> str | None:
    text = normalize_name(raw_label)
    if "acne" in text or "rosacea" in text:
        return "acne"
    if "eczema" in text:
        return "eczema"
    if "dermatitis" in text or "poison ivy" in text:
        return "dermatitis"
    if any(k in text for k in ["pigment", "melasma", "lentigo", "dark spot"]):
        return "pigmentation"
    return None


def best_scin_label(weighted_label: str) -> str:
    try:
        labels = ast.literal_eval(str(weighted_label))
    except Exception:
        return "unknown"
    if not isinstance(labels, dict) or not labels:
        return "unknown"
    ranked = sorted(labels.items(), key=lambda item: float(item[1]), reverse=True)
    primary = str(ranked[0][0])
    if map_focus_label(primary) is not None:
        return primary
    focused = [(str(name), score) for name, score in ranked if map_focus_label(str(name)) is not None]
    return focused[0][0] if focused else primary


def iter_zip_items(zip_path: Path, source: str, target_class: str, wanted_label: str, prefix_filter: str | None = None):
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            suffix = Path(info.filename).suffix.lower()
            if info.is_dir() or suffix not in IMAGE_EXTS:
                continue
            normalized = info.filename.replace("\\", "/")
            if prefix_filter and not normalized.startswith(prefix_filter):
                continue
            parts = [part for part in normalized.split("/") if part]
            if not parts:
                continue
            raw_label = parts[0]
            if source == "skindisnet" and len(parts) > 1:
                raw_label = parts[1]
            if raw_label != wanted_label:
                continue
            yield RawItem(source, raw_label, target_class, f"{zip_path}!{info.filename}", zf.read(info), suffix)


def iter_dir_items(root: Path, source: str, target_class: str, wanted_label: str):
    if not root.exists():
        return
    wanted_key = normalize_name(wanted_label)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(root)
        raw_label = rel.parts[0] if rel.parts else root.name
        if normalize_name(raw_label) == wanted_key:
            yield RawItem(source, raw_label, target_class, str(path), path.read_bytes(), path.suffix.lower())


def iter_dermnet(raw_dir: Path, target_class: str, label: str):
    dermnet_dir = raw_dir / "Dermnet"
    for zip_name in ["train.zip", "test.zip"]:
        zip_path = dermnet_dir / zip_name
        if zip_path.exists():
            yield from iter_zip_items(zip_path, "dermnet", target_class, label)


def iter_skindisnet(raw_dir: Path, target_class: str, label: str):
    zip_path = raw_dir / "SkinDisNet_2.zip"
    if zip_path.exists():
        yield from iter_zip_items(zip_path, "skindisnet", target_class, label, prefix_filter="Preprocessed/")


def iter_mendeley(raw_dir: Path, target_class: str, label: str):
    mendeley_dir = raw_dir / "Mendeley Skin Disease Classification Dataset"
    if not mendeley_dir.exists():
        return
    label_key = normalize_name(label)
    for zip_path in sorted(mendeley_dir.glob("*.zip")):
        if normalize_name(zip_path.stem) != label_key:
            continue
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                suffix = Path(info.filename).suffix.lower()
                if info.is_dir() or suffix not in IMAGE_EXTS:
                    continue
                yield RawItem("mendeley", label, target_class, f"{zip_path}!{info.filename}", zf.read(info), suffix)
    yield from iter_dir_items(mendeley_dir, "mendeley", target_class, label)


def iter_scin(raw_dir: Path, target_class: str, label: str):
    scin_dir = next((raw_dir / name for name in ["scin", "SCIN"] if (raw_dir / name).exists()), None)
    if scin_dir is None:
        return
    dataset_dir = scin_dir / "dataset"
    cases_path = dataset_dir / "scin_cases.csv"
    labels_path = dataset_dir / "scin_labels.csv"
    if not cases_path.exists() or not labels_path.exists():
        yield from iter_dir_items(scin_dir, "scin", target_class, label)
        return
    cases = pd.read_csv(cases_path)
    labels_df = pd.read_csv(labels_path)
    merged = cases.merge(labels_df[["case_id", "weighted_skin_condition_label"]], on="case_id", how="left")
    target_key = normalize_name(label)
    for _, row in merged.iterrows():
        raw_label = best_scin_label(str(row.get("weighted_skin_condition_label", "")))
        if normalize_name(raw_label) != target_key:
            continue
        for col in ["image_1_path", "image_2_path", "image_3_path"]:
            rel = row.get(col)
            if pd.isna(rel) or not str(rel).strip():
                continue
            image_path = scin_dir / str(rel).replace("/", "\\")
            if not image_path.exists():
                image_path = scin_dir / str(rel).replace("\\", "/")
            if image_path.exists():
                yield RawItem("scin", raw_label, target_class, str(image_path), image_path.read_bytes(), image_path.suffix.lower())


def iter_target_items(raw_dir: Path):
    dispatch = {
        "dermnet": iter_dermnet,
        "skindisnet": iter_skindisnet,
        "mendeley": iter_mendeley,
        "scin": iter_scin,
    }
    for target_class, specs in TARGET_SPECS.items():
        for source, label in specs:
            yield from dispatch[source](raw_dir, target_class, label)


def metric_defaults(cfg: dict) -> dict:
    q = cfg.get("data_quality", {})
    return {
        "min_size": int(q.get("min_size", cfg["data"].get("min_size", 64))),
        "black_ratio_threshold": float(q.get("black_ratio_threshold", 0.35)),
        "dark_mean_threshold": float(q.get("dark_mean_threshold", 35.0)),
        "dark_ratio_threshold": float(q.get("dark_ratio_threshold", 0.65)),
        "bright_mean_threshold": float(q.get("bright_mean_threshold", 235.0)),
        "bright_ratio_threshold": float(q.get("bright_ratio_threshold", 0.65)),
        "blur_laplacian_threshold": float(q.get("blur_laplacian_threshold", 18.0)),
        "mosaic_block_ratio_threshold": float(q.get("mosaic_block_ratio_threshold", 1.9)),
        "mosaic_boundary_threshold": float(q.get("mosaic_boundary_threshold", 7.0)),
        "lesion_edge_density_threshold": float(q.get("lesion_edge_density_threshold", 0.010)),
        "lesion_saturation_std_threshold": float(q.get("lesion_saturation_std_threshold", 10.0)),
        "save_max_side": int(q.get("save_max_side", 0) or 0),
        "jpeg_quality": int(q.get("jpeg_quality", 95)),
    }


def image_metrics(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"))
    if cv2 is not None:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        hsv_saturation = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:, :, 1]
    else:
        gray = np.asarray(image.convert("L"))
        hsv_saturation = np.asarray(image.convert("HSV"))[:, :, 1]
    luma = gray.astype(np.float32)
    h, w = gray.shape
    if cv2 is not None:
        resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float((edges > 0).mean())
    else:
        resized = np.asarray(Image.fromarray(gray).resize((256, 256), Image.Resampling.BILINEAR))
        padded = np.pad(luma, 1, mode="edge")
        laplacian = (
            -4 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        laplacian_var = float(laplacian.var())
        gy, gx = np.gradient(luma)
        edge_density = float((np.sqrt(gx * gx + gy * gy) > 25).mean())
    gx = np.abs(np.diff(resized.astype(np.float32), axis=1))
    gy = np.abs(np.diff(resized.astype(np.float32), axis=0))
    vertical_boundary = gx[:, 7::8].mean() if gx[:, 7::8].size else 0.0
    vertical_non = np.delete(gx, np.arange(7, gx.shape[1], 8), axis=1).mean() if gx.shape[1] > 8 else gx.mean()
    horizontal_boundary = gy[7::8, :].mean() if gy[7::8, :].size else 0.0
    horizontal_non = np.delete(gy, np.arange(7, gy.shape[0], 8), axis=0).mean() if gy.shape[0] > 8 else gy.mean()
    boundary = float((vertical_boundary + horizontal_boundary) / 2.0)
    non_boundary = float((vertical_non + horizontal_non) / 2.0)
    return {
        "width": float(w),
        "height": float(h),
        "mean_luma": float(luma.mean()),
        "dark_ratio": float((luma < 20).mean()),
        "bright_ratio": float((luma > 245).mean()),
        "black_ratio": float(((arr[:, :, 0] < 12) & (arr[:, :, 1] < 12) & (arr[:, :, 2] < 12)).mean()),
        "laplacian_var": laplacian_var,
        "mosaic_boundary": boundary,
        "mosaic_block_ratio": float(boundary / (non_boundary + 1e-6)),
        "edge_density": edge_density,
        "saturation_std": float(hsv_saturation.std()),
    }


def detect_reasons(metrics: dict[str, float], thresholds: dict) -> list[str]:
    reasons = []
    if min(metrics["width"], metrics["height"]) < thresholds["min_size"]:
        reasons.append("too_small")
    if metrics["black_ratio"] >= thresholds["black_ratio_threshold"]:
        reasons.append("black_occlusion_large")
    if metrics["mean_luma"] >= thresholds["bright_mean_threshold"] or metrics["bright_ratio"] >= thresholds["bright_ratio_threshold"]:
        reasons.append("overexposed")
    if metrics["mean_luma"] <= thresholds["dark_mean_threshold"] or metrics["dark_ratio"] >= thresholds["dark_ratio_threshold"]:
        reasons.append("too_dark")
    if metrics["laplacian_var"] <= thresholds["blur_laplacian_threshold"]:
        reasons.append("severe_blur")
    if (
        metrics["mosaic_block_ratio"] >= thresholds["mosaic_block_ratio_threshold"]
        and metrics["mosaic_boundary"] >= thresholds["mosaic_boundary_threshold"]
    ):
        reasons.append("mosaic_or_pixelated")
    if (
        metrics["edge_density"] <= thresholds["lesion_edge_density_threshold"]
        and metrics["saturation_std"] <= thresholds["lesion_saturation_std_threshold"]
        and "too_dark" not in reasons
        and "overexposed" not in reasons
    ):
        reasons.append("lesion_not_visible")
    return reasons


def primary_reason(reasons: list[str]) -> str:
    for reason in REASON_PRIORITY:
        if reason in reasons:
            return reason
    return reasons[0] if reasons else "kept"


def prepare_image_for_save(image: Image.Image, max_side: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max_side and max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def save_rejected_bytes(data: bytes, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)


def save_jpg(image: Image.Image, out_path: Path, quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="JPEG", quality=quality)


def clean_outputs(cfg: dict) -> None:
    targets = [
        Path(cfg["paths"].get("new_dir", "data/new")),
        Path(cfg["paths"].get("pass_dir", "data/pass")),
        Path(cfg["paths"]["processed_image_dir"]),
        Path(cfg["paths"]["interim_dir"]),
    ]
    for path in targets:
        if path.exists():
            shutil.rmtree(path)


def copy_processed_splits(new_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed = int(cfg["project"]["seed"])
    val_ratio = float(cfg["data"]["val_ratio"])
    test_ratio = float(cfg["data"]["test_ratio"])
    temp_ratio = val_ratio + test_ratio
    processed_dir = Path(cfg["paths"]["processed_image_dir"])
    class_names = cfg["project"]["class_names"]
    parts = []
    for label, group in new_df.groupby("label", sort=False):
        if len(group) < 3:
            raise RuntimeError(f"类别 {label} 样本少于 3 张，无法划分 train/val/test。")
        train_idx, temp_idx = train_test_split(group.index, test_size=temp_ratio, random_state=seed)
        temp = group.loc[temp_idx]
        if len(temp) < 2:
            val_idx = temp.index[:0]
            test_idx = temp.index
        else:
            val_share = val_ratio / temp_ratio
            val_idx, test_idx = train_test_split(temp.index, test_size=1 - val_share, random_state=seed)
        split_map = {idx: "train" for idx in train_idx}
        split_map.update({idx: "val" for idx in val_idx})
        split_map.update({idx: "test" for idx in test_idx})
        split_group = group.copy()
        split_group["split"] = split_group.index.map(split_map)
        parts.append(split_group)
    df = pd.concat(parts).reset_index(drop=True)
    df["label_id"] = df["label"].map({name: idx for idx, name in enumerate(class_names)}).astype(int)
    processed_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying processed splits"):
        src = Path(row["new_image_path"])
        dst = processed_dir / row["split"] / row["label"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        processed_paths.append(str(dst))
    df["image_path"] = processed_paths
    return df


def write_summary_files(all_df: pd.DataFrame, kept_df: pd.DataFrame, rejected_df: pd.DataFrame, split_df: pd.DataFrame, cfg: dict) -> None:
    new_dir = Path(cfg["paths"].get("new_dir", "data/new"))
    pass_dir = Path(cfg["paths"].get("pass_dir", "data/pass"))
    interim_dir = Path(cfg["paths"]["interim_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    for path in [new_dir, pass_dir, interim_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    kept_df.to_csv(new_dir / "manifest.csv", index=False)
    all_df.to_csv(pass_dir / "quality_report.csv", index=False)
    rejected_df.to_csv(pass_dir / "rejected_manifest.csv", index=False)
    split_df.to_csv(interim_dir / "split_samples.csv", index=False)
    split_df.to_csv(interim_dir / "cleaned_samples.csv", index=False)
    split_df.to_csv(Path(cfg["paths"]["manifest"]), index=False)
    split_df.to_csv(Path(cfg["paths"]["processed_image_dir"]) / "manifest.csv", index=False)

    status_counts = all_df["status"].value_counts().rename_axis("status").rename("count").reset_index()
    status_counts.to_csv(interim_dir / "cleaning_summary.csv", index=False)
    status_counts.to_csv(pass_dir / "quality_summary.csv", index=False)

    split_df.groupby("label").size().rename("count").reset_index().to_csv(interim_dir / "class_distribution.csv", index=False)
    split_df.groupby(["source_dataset", "label"]).size().rename("count").reset_index().to_csv(
        interim_dir / "source_class_distribution.csv", index=False
    )
    split_df.groupby(["split", "label"]).size().rename("count").reset_index().to_csv(
        interim_dir / "split_distribution.csv", index=False
    )

    by_reason_source = (
        all_df.groupby(["status", "source_dataset", "label"]).size().rename("count").reset_index().sort_values(["status", "source_dataset", "label"])
    )
    by_reason_source.to_csv(pass_dir / "quality_summary_by_source_class.csv", index=False)

    lines = [
        "# 新十分类数据集清洗报告",
        "",
        f"- 候选图片总数：{len(all_df)}",
        f"- 保留图片数：{len(kept_df)}",
        f"- 过滤图片数：{len(rejected_df)}",
        f"- 纯净数据集目录：`{new_dir}`",
        f"- 被过滤图片目录：`{pass_dir}`",
        f"- 训练划分目录：`{Path(cfg['paths']['processed_image_dir'])}`",
        "",
        "## 按处理状态统计",
        "",
        "```text",
        status_counts.to_string(index=False),
        "```",
        "",
        "## 最终 train / val / test 类别分布",
        "",
        "```text",
        split_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False),
        "```",
        "",
        "## 过滤原因说明",
        "",
    ]
    for reason, cn in REASON_CN.items():
        if reason == "kept":
            continue
        lines.append(f"- `{reason}`：{cn}，检测类型：{REASON_TYPE.get(reason, 'automatic')}")
    (pass_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def load_font(size: int):
    for name in ["msyh.ttc", "simhei.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(30, 30, 30), max_width=220):
    x, y = xy
    words = str(text).split()
    lines, current = [], ""
    for word in words:
        test = word if not current else current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines[:3]:
        draw.text((x, y), line, fill=fill, font=font)
        y += 15


def make_quality_figures(rejected_df: pd.DataFrame, cfg: dict) -> None:
    if rejected_df.empty:
        return
    pass_dir = Path(cfg["paths"].get("pass_dir", "data/pass"))
    fig_dir = pass_dir / "quality_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    q_cfg = cfg.get("data_quality", {})
    max_pages = q_cfg.get("max_visual_pages_per_reason", None)
    cell = 180
    caption = 58
    margin = 12
    title_h = 42
    cols = 5
    font_title = load_font(20)
    font_caption = load_font(11)
    for reason, group in rejected_df.groupby("status", sort=False):
        rows = group.reset_index(drop=True)
        pages = math.ceil(len(rows) / 10)
        if max_pages:
            pages = min(pages, int(max_pages))
        for page_idx in range(pages):
            page = rows.iloc[page_idx * 10 : (page_idx + 1) * 10]
            page_rows = math.ceil(len(page) / cols)
            canvas = Image.new("RGB", (cols * cell + (cols + 1) * margin, title_h + page_rows * (cell + caption) + (page_rows + 1) * margin), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((margin, 10), f"{reason} / {REASON_CN.get(reason, reason)} ({page_idx + 1}/{pages})", fill=(20, 20, 20), font=font_title)
            for idx, row in page.iterrows():
                pos = idx - page_idx * 10
                r = pos // cols
                c = pos % cols
                x = margin + c * (cell + margin)
                y = title_h + margin + r * (cell + caption + margin)
                try:
                    img = Image.open(row["pass_path"]).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    img = ImageOps.fit(img, (cell, cell), method=Image.Resampling.LANCZOS)
                    canvas.paste(img, (x, y))
                except Exception:
                    draw.rectangle((x, y, x + cell, y + cell), fill=(235, 235, 235), outline=(180, 180, 180))
                    draw.text((x + 10, y + 80), "invalid image", fill=(120, 0, 0), font=font_caption)
                draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(210, 210, 210))
                caption_text = f"{row['source_dataset']} | {row['label']} | {row['original_label']}"
                metric_text = f"blur={row.get('laplacian_var', ''):.1f} bright={row.get('bright_ratio', ''):.2f}"
                draw_text(draw, (x, y + cell + 6), caption_text, font_caption, max_width=cell)
                draw_text(draw, (x, y + cell + 36), metric_text, font_caption, fill=(80, 80, 80), max_width=cell)
            canvas.save(fig_dir / f"{reason}_{page_idx + 1:03d}.png", quality=95)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    class_names = cfg["project"]["class_names"]
    if class_names != CLASS_NAMES:
        raise RuntimeError(f"config.yaml 中的 class_names 与新十分类不一致：{class_names}")
    if args.clean:
        clean_outputs(cfg)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    new_dir = Path(cfg["paths"].get("new_dir", "data/new"))
    pass_dir = Path(cfg["paths"].get("pass_dir", "data/pass"))
    interim_dir = Path(cfg["paths"]["interim_dir"])
    for path in [new_dir, pass_dir, interim_dir, Path(cfg["paths"]["processed_image_dir"]), Path(cfg["paths"]["reports_dir"])]:
        path.mkdir(parents=True, exist_ok=True)

    thresholds = metric_defaults(cfg)
    seen_hashes: set[str] = set()
    all_records = []
    kept_records = []
    rejected_records = []

    print_section("构建新十分类纯净数据集")
    print(f"配置文件：{args.config}")
    print(f"原始数据目录：{raw_dir.resolve()}")
    print(f"纯净数据集目录：{new_dir.resolve()}")
    print(f"过滤图片目录：{pass_dir.resolve()}")
    print(f"训练划分目录：{Path(cfg['paths']['processed_image_dir']).resolve()}")
    print(f"类别：{', '.join(CLASS_NAMES)}")

    candidate_iter = iter_target_items(raw_dir)
    debug_class_counts: Counter[str] = Counter()
    for item in tqdm(candidate_iter, desc="Scanning selected raw images"):
        if args.max_per_class is not None and debug_class_counts[item.target_class] >= args.max_per_class:
            continue
        debug_class_counts[item.target_class] += 1
        sha = hashlib.sha256(item.data).hexdigest()
        metrics: dict[str, float] = {}
        reasons: list[str] = []
        image = None
        try:
            image = Image.open(io.BytesIO(item.data))
            image = ImageOps.exif_transpose(image).convert("RGB")
            metrics = image_metrics(image)
        except Exception:
            reasons = ["invalid_image"]

        if not reasons:
            if sha in seen_hashes:
                reasons.append("duplicate")
            reasons.extend(detect_reasons(metrics, thresholds))

        status = primary_reason(reasons)
        filename = f"{safe_name(item.source_dataset)}__{safe_name(item.original_label)}__{sha[:16]}.jpg"
        base_record = {
            "source_dataset": item.source_dataset,
            "original_label": item.original_label,
            "label": item.target_class,
            "label_id": CLASS_NAMES.index(item.target_class),
            "original_path": item.original_path,
            "sha256": sha,
            "status": status,
            "status_cn": REASON_CN.get(status, status),
            "reason_type": REASON_TYPE.get(status, "kept"),
            "all_reasons": ";".join(reasons) if reasons else "kept",
            "width": int(metrics.get("width", 0)),
            "height": int(metrics.get("height", 0)),
            **metrics,
        }

        if status == "kept":
            out_path = new_dir / item.target_class / filename
            save_jpg(prepare_image_for_save(image, thresholds["save_max_side"]), out_path, thresholds["jpeg_quality"])
            seen_hashes.add(sha)
            record = {**base_record, "new_image_path": str(out_path), "image_path": str(out_path), "pass_path": ""}
            kept_records.append(record)
            all_records.append(record)
        else:
            reject_dir = pass_dir / status / item.target_class
            if image is None:
                out_path = reject_dir / f"{safe_name(item.source_dataset)}__{safe_name(item.original_label)}__{sha[:16]}{item.suffix or '.bin'}"
                save_rejected_bytes(item.data, out_path)
            else:
                out_path = reject_dir / filename
                save_jpg(prepare_image_for_save(image, thresholds["save_max_side"]), out_path, thresholds["jpeg_quality"])
            record = {**base_record, "new_image_path": "", "image_path": "", "pass_path": str(out_path)}
            rejected_records.append(record)
            all_records.append(record)

    if not kept_records:
        raise RuntimeError("没有保留下任何图片，请检查 raw 数据路径和清洗阈值。")

    all_df = pd.DataFrame(all_records)
    kept_df = pd.DataFrame(kept_records)
    rejected_df = pd.DataFrame(rejected_records)
    split_df = copy_processed_splits(kept_df, cfg)
    write_summary_files(all_df, kept_df, rejected_df, split_df, cfg)
    make_quality_figures(rejected_df, cfg)

    print_section("新数据集构建完成")
    print("处理状态统计：")
    print(all_df["status"].value_counts().rename_axis("status").rename("count").reset_index().to_string(index=False))
    print("\n最终类别分布：")
    print(split_df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_string())
    print("\n最终划分分布：")
    print(split_df.groupby(["split", "label"]).size().rename("count").reset_index().to_string(index=False))
    print(f"\n纯净数据集 manifest：{(new_dir / 'manifest.csv').resolve()}")
    print(f"被过滤图片 manifest：{(pass_dir / 'rejected_manifest.csv').resolve()}")
    print(f"质量报告：{(pass_dir / 'quality_report.md').resolve()}")
    print(f"训练 manifest：{Path(cfg['paths']['manifest']).resolve()}")


if __name__ == "__main__":
    main()
