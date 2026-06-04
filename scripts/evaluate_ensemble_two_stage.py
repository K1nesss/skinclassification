from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.skin_dataset import build_eval_loader
from src.models.build_model import load_checkpoint_model
from src.utils.io import ensure_project_dirs, load_config, save_json
from src.utils.metrics import compute_classification_metrics
from src.visualization.plot_confusion_matrix import plot_confusion_matrix
from src.visualization.plot_metrics_bar import plot_per_class_metrics


DEFAULT_SPECIALIST_CLASSES = ["eczema", "dermatitis"]
TARGET_CLASSES = ["eczema", "dermatitis", "psoriasis_lichen_planus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ensemble with a specialist refinement stage.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-checkpoints", nargs="+", required=True)
    parser.add_argument("--specialist-checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--classes", nargs="+", default=DEFAULT_SPECIALIST_CLASSES)
    parser.add_argument(
        "--other-class-name",
        default=None,
        help="Name of specialist output that means keep the ensemble prediction.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--search-on-val", action="store_true")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--gate-mode", choices=["pred", "mass", "pred_or_mass"], default="pred")
    parser.add_argument("--gate-threshold", type=float, default=0.45)
    parser.add_argument(
        "--refine-source-datasets",
        nargs="+",
        default=None,
        help="Optionally run the specialist only for these source_dataset values.",
    )
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
def collect_probs(checkpoint: str, cfg: dict, split: str, device: torch.device, num_classes: int, include_rows: bool = False):
    loader = build_eval_loader(cfg, split)
    model = load_checkpoint_model(checkpoint, num_classes, device)
    y_true: list[int] = []
    probs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    n_images = 0
    for images, labels, meta in tqdm(loader, desc=f"Predicting {Path(checkpoint).stem} {split}"):
        images = images.to(device, non_blocking=True)
        batch_probs = torch.softmax(model(images), dim=1).cpu().numpy()
        probs.append(batch_probs)
        y_true.extend(labels.numpy().astype(int).tolist())
        if include_rows:
            rows.extend(metadata_rows(meta, images.size(0)))
        n_images += images.size(0)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(y_true, dtype=np.int64), np.vstack(probs), (time.perf_counter() - start) / max(1, n_images), rows


def normalize_weights(weights: list[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if np.any(arr < 0):
        raise ValueError("Ensemble weights must be non-negative.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    return arr / total


def combine_probs(model_probs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, np.stack(model_probs, axis=0), axes=(0, 0))


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


def should_refine(base_pred: int, base_probs: np.ndarray, specialist_ids: list[int], gate_mode: str, threshold: float) -> bool:
    pred_in_cluster = base_pred in specialist_ids
    cluster_mass = float(base_probs[specialist_ids].sum())
    if gate_mode == "pred":
        return pred_in_cluster
    if gate_mode == "mass":
        return cluster_mass >= threshold
    return pred_in_cluster or cluster_mass >= threshold


def apply_specialist(
    base_probs: np.ndarray,
    specialist_probs: np.ndarray,
    class_names: list[str],
    specialist_classes: list[str],
    other_class_name: str | None,
    gate_mode: str,
    gate_threshold: float,
    rows: list[dict[str, Any]] | None = None,
    refine_source_datasets: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    specialist_ids = [class_names.index(name) for name in specialist_classes]
    output_names = list(specialist_classes)
    if other_class_name:
        output_names.append(other_class_name)
    specialist_to_global = {
        idx: (class_names.index(name) if name in class_names else None)
        for idx, name in enumerate(output_names)
    }

    final_probs = base_probs.copy()
    base_preds = base_probs.argmax(axis=1)
    final_preds = base_preds.copy()
    refined_count = 0
    changed_count = 0
    refine_sources = set(refine_source_datasets or [])
    for idx, base_pred in enumerate(base_preds.tolist()):
        if refine_sources and rows is not None and rows[idx].get("source_dataset") not in refine_sources:
            continue
        if not should_refine(base_pred, base_probs[idx], specialist_ids, gate_mode, gate_threshold):
            continue
        specialist_output = int(specialist_probs[idx].argmax())
        specialist_pred = specialist_to_global[specialist_output]
        refined_count += 1
        if specialist_pred is None:
            continue
        cluster_mass = float(base_probs[idx, specialist_ids].sum())
        final_preds[idx] = specialist_pred
        final_probs[idx, specialist_ids] = specialist_probs[idx, : len(specialist_ids)] * max(cluster_mass, 1e-6)
        if specialist_pred != int(base_pred):
            changed_count += 1
    return final_preds, final_probs, {"refined_count": refined_count, "changed_count": changed_count}


def print_summary(metrics: dict, weights: np.ndarray, checkpoints: list[str]) -> None:
    print("\n" + "-" * 80)
    print("集成二阶段评估结果摘要")
    print("-" * 80)
    for checkpoint, weight in zip(checkpoints, weights.tolist()):
        print(f"  基础权重 {weight:.3f}: {checkpoint}")
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy：{metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"ECE 校准误差：{metrics['ece']:.4f}")
    print(
        "二阶段触发："
        f"{metrics['two_stage']['refined_count']}/{metrics['two_stage']['total_count']}，"
        f"改判 {metrics['two_stage']['changed_count']} 张"
    )
    print("\n每类指标：")
    for label, item in metrics["per_class"].items():
        print(
            f"  {label}: P={item['precision']:.4f}, "
            f"R={item['recall']:.4f}, F1={item['f1']:.4f}, 样本数={item['support']}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    class_names = cfg["project"]["class_names"]
    unknown = [name for name in args.classes if name not in class_names]
    if unknown:
        raise ValueError(f"Unknown specialist classes: {unknown}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base_checkpoints = [str(Path(path)) for path in args.base_checkpoints]
    base_num_classes = int(cfg["project"]["num_classes"])
    specialist_num_classes = len(args.classes) + (1 if args.other_class_name else 0)

    if args.search_on_val:
        val_probs = []
        val_true_ref = None
        for checkpoint in base_checkpoints:
            y_true, probs, _, _ = collect_probs(checkpoint, cfg, "val", device, base_num_classes)
            if val_true_ref is None:
                val_true_ref = y_true
            elif not np.array_equal(val_true_ref, y_true):
                raise ValueError("Validation loaders produced different label order.")
            val_probs.append(probs)
        weights = search_weights(val_true_ref, val_probs, class_names, float(args.weight_step))
        print("验证集搜索得到基础集成权重：" + ", ".join(f"{weight:.3f}" for weight in weights.tolist()))
    elif args.weights is not None:
        if len(args.weights) != len(base_checkpoints):
            raise ValueError("--weights length must match --base-checkpoints length.")
        weights = normalize_weights(args.weights)
    else:
        weights = np.ones(len(base_checkpoints), dtype=np.float64) / len(base_checkpoints)

    split_probs = []
    true_ref = None
    rows_ref: list[dict[str, Any]] | None = None
    seconds_per_image = 0.0
    for idx, checkpoint in enumerate(base_checkpoints):
        y_true, probs, seconds, rows = collect_probs(
            checkpoint,
            cfg,
            args.split,
            device,
            base_num_classes,
            include_rows=(idx == 0),
        )
        if true_ref is None:
            true_ref = y_true
            rows_ref = rows
        elif not np.array_equal(true_ref, y_true):
            raise ValueError("Evaluation loaders produced different label order.")
        split_probs.append(probs)
        seconds_per_image += seconds
    assert true_ref is not None and rows_ref is not None

    _, specialist_probs, specialist_seconds, _ = collect_probs(
        args.specialist_checkpoint,
        cfg,
        args.split,
        device,
        specialist_num_classes,
    )
    base_probs = combine_probs(split_probs, weights)
    final_preds, final_probs, stage_info = apply_specialist(
        base_probs,
        specialist_probs,
        class_names,
        list(args.classes),
        args.other_class_name,
        args.gate_mode,
        float(args.gate_threshold),
        rows=rows_ref,
        refine_source_datasets=args.refine_source_datasets,
    )
    metrics = compute_classification_metrics(true_ref, final_preds, final_probs, class_names)
    metrics["seconds_per_image"] = float(seconds_per_image + specialist_seconds)
    metrics["ensemble"] = {
        "base_checkpoints": base_checkpoints,
        "weights": weights.tolist(),
        "search_on_val": bool(args.search_on_val),
        "weight_step": float(args.weight_step),
    }
    metrics["two_stage"] = {
        "specialist_checkpoint": str(Path(args.specialist_checkpoint)),
        "specialist_classes": list(args.classes),
        "other_class_name": args.other_class_name,
        "gate_mode": args.gate_mode,
        "gate_threshold": float(args.gate_threshold),
        "refine_source_datasets": args.refine_source_datasets,
        "total_count": int(len(true_ref)),
        **{key: int(value) for key, value in stage_info.items()},
    }

    output_name = args.output_name or f"ensemble_two_stage_{args.split}_metrics"
    result_path = Path(cfg["paths"]["reports_dir"]) / f"{output_name}.json"
    save_json(metrics, result_path)
    fig_dir = Path(cfg["paths"]["figures_dir"])
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, fig_dir / f"{output_name}_confusion_matrix.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        fig_dir / f"{output_name}_confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_per_class_metrics(metrics["per_class"], fig_dir / f"{output_name}_per_class_metrics.png")
    print_summary(metrics, weights, base_checkpoints)
    print("\n输出文件：")
    print(f"  指标 JSON：{result_path}")
    print(f"  混淆矩阵：{fig_dir / f'{output_name}_confusion_matrix.png'}")
    print(f"  归一化混淆矩阵：{fig_dir / f'{output_name}_confusion_matrix_normalized.png'}")
    print(f"  每类指标图：{fig_dir / f'{output_name}_per_class_metrics.png'}")


if __name__ == "__main__":
    main()
