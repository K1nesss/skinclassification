from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

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


TARGET_CLASSES = ["eczema", "dermatitis", "psoriasis_lichen_planus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and tune a probability ensemble.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--manifest", default=None, help="Override paths.manifest for this evaluation.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--search-on-val", action="store_true")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument(
        "--tta-hflip",
        action="store_true",
        help="Average predictions from the original image and a horizontal flip.",
    )
    parser.add_argument("--output-name", default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_probs(checkpoint: str, cfg: dict, split: str, device: torch.device, tta_hflip: bool = False):
    loader = build_eval_loader(cfg, split)
    model = load_checkpoint_model(checkpoint, int(cfg["project"]["num_classes"]), device)
    y_true: list[int] = []
    all_probs: list[np.ndarray] = []
    start = time.perf_counter()
    n_images = 0
    for images, labels, _ in tqdm(loader, desc=f"Predicting {Path(checkpoint).stem} {split}"):
        images = images.to(device, non_blocking=True)
        batch_tensor_probs = torch.softmax(model(images), dim=1)
        if tta_hflip:
            flip_probs = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
            batch_tensor_probs = (batch_tensor_probs + flip_probs) * 0.5
        batch_probs = batch_tensor_probs.cpu().numpy()
        all_probs.append(batch_probs)
        y_true.extend(labels.numpy().astype(int).tolist())
        n_images += images.size(0)
    seconds_per_image = (time.perf_counter() - start) / max(1, n_images)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(y_true, dtype=np.int64), np.vstack(all_probs), seconds_per_image


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


def target_score(metrics: dict, target_classes: list[str]) -> float:
    target_f1 = [metrics["per_class"][name]["f1"] for name in target_classes]
    min_target = min(target_f1)
    mean_target = float(np.mean(target_f1))
    return min_target * 3.0 + mean_target + metrics["macro_f1"] * 0.25


def candidate_weights(num_models: int, step: float):
    if num_models == 1:
        yield np.ones(1, dtype=np.float64)
        return
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("--weight-step must divide 1.0 cleanly, e.g. 0.1 or 0.05.")
    for parts in itertools.product(range(units + 1), repeat=num_models):
        if sum(parts) != units:
            continue
        yield np.asarray(parts, dtype=np.float64) / units


def search_weights(
    y_true: np.ndarray,
    model_probs: list[np.ndarray],
    class_names: list[str],
    target_classes: list[str],
    step: float,
) -> tuple[np.ndarray, dict]:
    best_weights: np.ndarray | None = None
    best_metrics: dict | None = None
    best_score = -np.inf
    for weights in candidate_weights(len(model_probs), step):
        probs = combine_probs(model_probs, weights)
        metrics = compute_classification_metrics(y_true, probs.argmax(axis=1), probs, class_names)
        score = target_score(metrics, target_classes)
        if score > best_score:
            best_score = score
            best_weights = weights
            best_metrics = metrics
    assert best_weights is not None and best_metrics is not None
    return best_weights, best_metrics


def print_summary(metrics: dict, weights: np.ndarray, checkpoints: list[str]) -> None:
    print("\n" + "-" * 80)
    print("集成评估结果摘要")
    print("-" * 80)
    for checkpoint, weight in zip(checkpoints, weights.tolist()):
        print(f"  权重 {weight:.3f}: {checkpoint}")
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy：{metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"ECE 校准误差：{metrics['ece']:.4f}")
    if "seconds_per_image" in metrics:
        print(f"单张图片平均推理时间：{metrics['seconds_per_image']:.6f} 秒")
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
    if args.image_size:
        cfg["data"]["image_size"] = args.image_size
    if args.manifest:
        cfg["paths"]["manifest"] = args.manifest
    class_names = cfg["project"]["class_names"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoints = [str(Path(path)) for path in args.checkpoints]
    if args.search_on_val:
        val_probs = []
        val_true_ref = None
        for checkpoint in checkpoints:
            y_true, probs, _ = collect_probs(checkpoint, cfg, "val", device, bool(args.tta_hflip))
            if val_true_ref is None:
                val_true_ref = y_true
            elif not np.array_equal(val_true_ref, y_true):
                raise ValueError("Validation loaders produced different label order.")
            val_probs.append(probs)
        weights, val_metrics = search_weights(
            val_true_ref,
            val_probs,
            class_names,
            TARGET_CLASSES,
            float(args.weight_step),
        )
        print_summary(val_metrics, weights, checkpoints)
        print("以上为验证集选权结果，将用同一权重评估指定 split。")
    elif args.weights is not None:
        if len(args.weights) != len(checkpoints):
            raise ValueError("--weights length must match --checkpoints length.")
        weights = normalize_weights(args.weights)
    else:
        weights = np.ones(len(checkpoints), dtype=np.float64) / len(checkpoints)

    split_probs = []
    true_ref = None
    seconds_per_image = 0.0
    for checkpoint in checkpoints:
        y_true, probs, seconds = collect_probs(checkpoint, cfg, args.split, device, bool(args.tta_hflip))
        if true_ref is None:
            true_ref = y_true
        elif not np.array_equal(true_ref, y_true):
            raise ValueError("Evaluation loaders produced different label order.")
        split_probs.append(probs)
        seconds_per_image += seconds

    combined = combine_probs(split_probs, weights)
    metrics = compute_classification_metrics(true_ref, combined.argmax(axis=1), combined, class_names)
    metrics["seconds_per_image"] = float(seconds_per_image)
    metrics["ensemble"] = {
        "checkpoints": checkpoints,
        "weights": weights.tolist(),
        "search_on_val": bool(args.search_on_val),
        "weight_step": float(args.weight_step),
        "tta_hflip": bool(args.tta_hflip),
    }

    output_name = args.output_name or f"ensemble_{args.split}_metrics"
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
    print_summary(metrics, weights, checkpoints)
    print("\n输出文件：")
    print(f"  指标 JSON：{result_path}")
    print(f"  混淆矩阵：{fig_dir / f'{output_name}_confusion_matrix.png'}")
    print(f"  归一化混淆矩阵：{fig_dir / f'{output_name}_confusion_matrix_normalized.png'}")
    print(f"  每类指标图：{fig_dir / f'{output_name}_per_class_metrics.png'}")


if __name__ == "__main__":
    main()
