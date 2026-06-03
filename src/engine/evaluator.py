from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.models.build_model import load_checkpoint_model
from src.utils.io import save_json
from src.utils.metrics import compute_classification_metrics
from src.visualization.plot_confusion_matrix import plot_confusion_matrix
from src.visualization.plot_metrics_bar import plot_per_class_metrics


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader, device: torch.device):
    model.eval()
    y_true, y_pred, y_prob, rows = [], [], [], []
    start = time.perf_counter()
    n_images = 0
    for images, labels, meta in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.append(probs)
        n_images += images.size(0)
        rows.append(meta)
    elapsed = time.perf_counter() - start
    return {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.vstack(y_prob) if y_prob else np.empty((0, 0)),
        "seconds_per_image": elapsed / max(1, n_images),
    }


def evaluate_model(model, loader, device, class_names: list[str]) -> dict:
    outputs = predict_loader(model, loader, device)
    metrics = compute_classification_metrics(
        outputs["y_true"], outputs["y_pred"], outputs["y_prob"], class_names
    )
    metrics["seconds_per_image"] = float(outputs["seconds_per_image"])
    return metrics


def evaluate_checkpoint(cfg, checkpoint, loader, device, result_path: str | Path) -> dict:
    model = load_checkpoint_model(checkpoint, int(cfg["project"]["num_classes"]), device)
    class_names = cfg["project"]["class_names"]
    metrics = evaluate_model(model, loader, device, class_names)
    result_path = Path(result_path)
    save_json(metrics, result_path)

    fig_dir = Path(cfg["paths"]["figures_dir"])
    stem = result_path.stem
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        fig_dir / f"{stem}_confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names,
        fig_dir / f"{stem}_confusion_matrix_normalized.png",
        normalize=True,
    )
    plot_per_class_metrics(metrics["per_class"], fig_dir / f"{stem}_per_class_metrics.png")
    print("\n" + "-" * 80)
    print("评估结果摘要")
    print("-" * 80)
    print(f"Accuracy：{metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy：{metrics['balanced_accuracy']:.4f}")
    print(f"Macro Precision：{metrics['macro_precision']:.4f}")
    print(f"Macro Recall：{metrics['macro_recall']:.4f}")
    print(f"Macro-F1：{metrics['macro_f1']:.4f}")
    if "ece" in metrics:
        print(f"ECE 校准误差：{metrics['ece']:.4f}")
    print(f"单张图片平均推理时间：{metrics['seconds_per_image']:.6f} 秒")
    print("\n每类指标：")
    for label, item in metrics["per_class"].items():
        print(
            f"  {label}: P={item['precision']:.4f}, "
            f"R={item['recall']:.4f}, F1={item['f1']:.4f}, 样本数={item['support']}"
        )
    print("\n输出文件：")
    print(f"  指标 JSON：{result_path}")
    print(f"  混淆矩阵：{fig_dir / f'{stem}_confusion_matrix.png'}")
    print(f"  归一化混淆矩阵：{fig_dir / f'{stem}_confusion_matrix_normalized.png'}")
    print(f"  每类指标图：{fig_dir / f'{stem}_per_class_metrics.png'}")
    return metrics
