from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.skin_dataset import build_eval_loader
from src.models.build_model import load_checkpoint_model
from src.utils.io import ensure_project_dirs, load_config
from src.utils.metrics import compute_classification_metrics


TARGET_CLASSES = ["eczema", "dermatitis", "psoriasis_lichen_planus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-sample predictions for error analysis.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--search-on-val", action="store_true")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


def metadata_rows(meta: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx in range(batch_size):
        row: dict[str, Any] = {}
        for key, value in meta.items():
            if torch.is_tensor(value):
                item = value[idx].item() if value.ndim > 0 else value.item()
            elif isinstance(value, (list, tuple)):
                item = value[idx]
            else:
                item = value
            row[key] = item
        rows.append(row)
    return rows


@torch.no_grad()
def collect_probs(checkpoint: str, cfg: dict, split: str, device: torch.device, include_rows: bool):
    loader = build_eval_loader(cfg, split)
    model = load_checkpoint_model(checkpoint, int(cfg["project"]["num_classes"]), device)
    y_true: list[int] = []
    probs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for images, labels, meta in tqdm(loader, desc=f"Predicting {Path(checkpoint).stem} {split}"):
        images = images.to(device, non_blocking=True)
        batch_probs = torch.softmax(model(images), dim=1).cpu().numpy()
        probs.append(batch_probs)
        y_true.extend(labels.numpy().astype(int).tolist())
        if include_rows:
            rows.extend(metadata_rows(meta, images.size(0)))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(y_true, dtype=np.int64), np.vstack(probs), rows


def normalize_weights(weights: list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("Ensemble weights must be non-negative.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    return arr / total


def combine_probs(model_probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(model_probs, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def candidate_weights(num_models: int, step: float):
    if num_models == 1:
        yield np.ones(1, dtype=np.float64)
        return
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--weight-step must divide 1.0 cleanly, e.g. 0.1 or 0.05.")
    for parts in itertools.product(range(units + 1), repeat=num_models):
        if sum(parts) == units:
            yield np.asarray(parts, dtype=np.float64) / units


def target_score(metrics: dict) -> float:
    target_f1 = [metrics["per_class"][name]["f1"] for name in TARGET_CLASSES]
    return min(target_f1) * 3.0 + float(np.mean(target_f1)) + metrics["macro_f1"] * 0.25


def search_weights(y_true: np.ndarray, model_probs: list[np.ndarray], class_names: list[str], step: float) -> np.ndarray:
    best_weights: np.ndarray | None = None
    best_score = -np.inf
    for weights in candidate_weights(len(model_probs), step):
        probs = combine_probs(model_probs, weights)
        metrics = compute_classification_metrics(y_true, probs.argmax(axis=1), probs, class_names)
        score = target_score(metrics)
        if score > best_score:
            best_score = score
            best_weights = weights
    assert best_weights is not None
    return best_weights


def build_output_frame(rows: list[dict[str, Any]], y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    pred_ids = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else confidence
    df = pd.DataFrame(rows)
    df["true_id"] = y_true
    df["pred_id"] = pred_ids
    df["true_label"] = [class_names[idx] for idx in y_true]
    df["pred_label"] = [class_names[idx] for idx in pred_ids]
    df["correct"] = df["true_id"] == df["pred_id"]
    df["confidence"] = confidence
    df["top1_top2_margin"] = margin
    target_set = set(TARGET_CLASSES)
    df["true_is_target"] = df["true_label"].isin(target_set)
    df["pred_is_target"] = df["pred_label"].isin(target_set)
    df["target_confusion"] = df["true_is_target"] & df["pred_is_target"] & ~df["correct"]
    for idx, class_name in enumerate(class_names):
        df[f"prob_{class_name}"] = probs[:, idx]
    return df


def print_error_summary(df: pd.DataFrame) -> None:
    print("\n目标三类错误分布：")
    target_errors = df[df["true_label"].isin(TARGET_CLASSES) & ~df["correct"]]
    if target_errors.empty:
        print("  无目标类错误。")
    else:
        print(
            target_errors.groupby(["true_label", "pred_label"])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["true_label", "count"], ascending=[True, False])
            .to_string(index=False)
        )

    print("\n目标三类按来源数据集的正确/错误：")
    target_rows = df[df["true_label"].isin(TARGET_CLASSES)]
    print(
        target_rows.groupby(["source_dataset", "true_label", "correct"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["source_dataset", "true_label", "correct"])
        .to_string(index=False)
    )

    print("\n所有类别按来源数据集的准确率：")
    source_acc = (
        df.groupby(["source_dataset", "true_label"])["correct"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy", "sum": "correct"})
        .sort_values(["true_label", "source_dataset"])
    )
    source_acc["accuracy"] = source_acc["accuracy"].map(lambda value: f"{value:.4f}")
    print(source_acc.to_string(index=False))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    class_names = cfg["project"]["class_names"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoints = [str(Path(path)) for path in args.checkpoints]

    if args.search_on_val:
        val_probs = []
        val_true_ref = None
        for checkpoint in checkpoints:
            y_true, probs, _ = collect_probs(checkpoint, cfg, "val", device, include_rows=False)
            if val_true_ref is None:
                val_true_ref = y_true
            elif not np.array_equal(val_true_ref, y_true):
                raise ValueError("Validation loaders produced different label order.")
            val_probs.append(probs)
        weights = search_weights(val_true_ref, val_probs, class_names, float(args.weight_step))
        print("验证集搜索得到权重：" + ", ".join(f"{weight:.3f}" for weight in weights.tolist()))
    elif args.weights is not None:
        if len(args.weights) != len(checkpoints):
            raise ValueError("--weights length must match --checkpoints length.")
        weights = normalize_weights(args.weights)
    else:
        weights = np.ones(len(checkpoints), dtype=np.float64) / len(checkpoints)

    split_probs = []
    true_ref = None
    rows_ref: list[dict[str, Any]] | None = None
    for idx, checkpoint in enumerate(checkpoints):
        y_true, probs, rows = collect_probs(checkpoint, cfg, args.split, device, include_rows=(idx == 0))
        if true_ref is None:
            true_ref = y_true
            rows_ref = rows
        elif not np.array_equal(true_ref, y_true):
            raise ValueError("Evaluation loaders produced different label order.")
        split_probs.append(probs)
    assert true_ref is not None and rows_ref is not None

    combined = combine_probs(split_probs, weights)
    df = build_output_frame(rows_ref, true_ref, combined, class_names)
    output_name = args.output_name or f"predictions_{args.split}"
    output_path = Path(cfg["paths"]["reports_dir"]) / f"{output_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    metrics = compute_classification_metrics(true_ref, combined.argmax(axis=1), combined, class_names)
    print("\n预测导出完成：")
    print(f"  CSV：{output_path}")
    print(f"  Accuracy={metrics['accuracy']:.4f}, Macro-F1={metrics['macro_f1']:.4f}")
    for label in TARGET_CLASSES:
        item = metrics["per_class"][label]
        print(f"  {label}: P={item['precision']:.4f}, R={item['recall']:.4f}, F1={item['f1']:.4f}")
    print_error_summary(df)


if __name__ == "__main__":
    main()
