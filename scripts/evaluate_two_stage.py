from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.skin_dataset import build_eval_loader
from src.models.build_model import load_checkpoint_model
from src.utils.io import ensure_project_dirs, load_config, save_json
from src.utils.metrics import compute_classification_metrics
from src.visualization.plot_confusion_matrix import plot_confusion_matrix
from src.visualization.plot_metrics_bar import plot_per_class_metrics


DEFAULT_SPECIALIST_CLASSES = [
    "eczema",
    "dermatitis",
    "psoriasis_lichen_planus",
    "fungal_infection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a two-stage classifier with a confusing-class specialist.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--specialist-checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--classes", nargs="+", default=DEFAULT_SPECIALIST_CLASSES)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--gate-mode",
        choices=["pred", "mass", "pred_or_mass"],
        default="pred_or_mass",
    )
    parser.add_argument("--gate-threshold", type=float, default=0.45)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def should_refine(
    base_pred: int,
    base_probs: np.ndarray,
    specialist_ids: list[int],
    gate_mode: str,
    gate_threshold: float,
) -> bool:
    pred_in_cluster = base_pred in specialist_ids
    cluster_mass = float(base_probs[specialist_ids].sum())
    if gate_mode == "pred":
        return pred_in_cluster
    if gate_mode == "mass":
        return cluster_mass >= gate_threshold
    return pred_in_cluster or cluster_mass >= gate_threshold


@torch.no_grad()
def evaluate_two_stage(
    base_model: torch.nn.Module,
    specialist_model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
    specialist_classes: list[str],
    gate_mode: str,
    gate_threshold: float,
) -> dict:
    base_model.eval()
    specialist_model.eval()
    specialist_ids = [class_names.index(name) for name in specialist_classes]
    specialist_to_global = np.asarray(specialist_ids, dtype=np.int64)

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    refined_count = 0
    changed_count = 0
    start = time.perf_counter()
    n_images = 0

    for images, labels, _ in tqdm(loader, desc="Two-stage evaluating"):
        images = images.to(device, non_blocking=True)
        base_probs = torch.softmax(base_model(images), dim=1).cpu().numpy()
        base_preds = base_probs.argmax(axis=1)
        final_preds = base_preds.copy()
        final_probs = base_probs.copy()

        refine_indices = [
            idx
            for idx, pred in enumerate(base_preds.tolist())
            if should_refine(pred, base_probs[idx], specialist_ids, gate_mode, gate_threshold)
        ]
        if refine_indices:
            refine_tensor = images[refine_indices]
            specialist_probs = torch.softmax(specialist_model(refine_tensor), dim=1).cpu().numpy()
            for local_idx, sample_idx in enumerate(refine_indices):
                specialist_pred = int(specialist_to_global[specialist_probs[local_idx].argmax()])
                cluster_mass = float(base_probs[sample_idx, specialist_ids].sum())
                final_preds[sample_idx] = specialist_pred
                final_probs[sample_idx, specialist_ids] = specialist_probs[local_idx] * max(cluster_mass, 1e-6)
                refined_count += 1
                if specialist_pred != int(base_preds[sample_idx]):
                    changed_count += 1

        y_true.extend(labels.numpy().astype(int).tolist())
        y_pred.extend(final_preds.astype(int).tolist())
        y_prob.extend(final_probs.astype(np.float64))
        n_images += images.size(0)

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        np.asarray(y_prob, dtype=np.float64),
        class_names,
    )
    metrics["seconds_per_image"] = float((time.perf_counter() - start) / max(1, n_images))
    metrics["two_stage"] = {
        "specialist_classes": specialist_classes,
        "gate_mode": gate_mode,
        "gate_threshold": gate_threshold,
        "refined_count": int(refined_count),
        "changed_count": int(changed_count),
        "total_count": int(n_images),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print("\n" + "-" * 80)
    print("二阶段评估结果摘要")
    print("-" * 80)
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy：{metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    print(f"ECE 校准误差：{metrics['ece']:.4f}")
    print(f"单张图片平均推理时间：{metrics['seconds_per_image']:.6f} 秒")
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
    base_model = load_checkpoint_model(args.base_checkpoint, int(cfg["project"]["num_classes"]), device)
    specialist_model = load_checkpoint_model(args.specialist_checkpoint, len(args.classes), device)
    loader = build_eval_loader(cfg, args.split)

    metrics = evaluate_two_stage(
        base_model=base_model,
        specialist_model=specialist_model,
        loader=loader,
        device=device,
        class_names=class_names,
        specialist_classes=list(args.classes),
        gate_mode=args.gate_mode,
        gate_threshold=float(args.gate_threshold),
    )

    output_name = args.output_name or (
        f"{Path(args.base_checkpoint).stem}_two_stage_{Path(args.specialist_checkpoint).stem}"
        f"_{args.gate_mode}_{args.split}_metrics"
    )
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
    print_metrics(metrics)
    print("\n输出文件：")
    print(f"  指标 JSON：{result_path}")
    print(f"  混淆矩阵：{fig_dir / f'{output_name}_confusion_matrix.png'}")
    print(f"  归一化混淆矩阵：{fig_dir / f'{output_name}_confusion_matrix_normalized.png'}")
    print(f"  每类指标图：{fig_dir / f'{output_name}_per_class_metrics.png'}")


if __name__ == "__main__":
    main()
