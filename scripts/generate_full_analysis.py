from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tqdm import tqdm

from src.datasets.skin_dataset import build_eval_loader
from src.models.build_model import build_model, get_target_layer, load_checkpoint_model
from src.utils.io import load_config
from src.visualization.feature_map import save_feature_map_grid
from src.visualization.gradcam_utils import make_gradcam_images
from src.visualization.matplotlib_style import setup_plot_style
from src.visualization.plot_confusion_matrix import plot_confusion_matrix
from src.visualization.plot_metrics_bar import plot_per_class_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full report artifacts.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None, help="Model checkpoint prefix. Default: best test Macro-F1.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-feature-samples", type=int, default=1000)
    parser.add_argument("--gradcam-per-class", type=int, default=3)
    parser.add_argument("--error-samples", type=int, default=12)
    return parser.parse_args()


def metric_path(cfg: dict, model: str, split: str) -> Path:
    return Path(cfg["paths"]["reports_dir"]) / f"{model}_best_{split}_metrics.json"


def checkpoint_path(cfg: dict, model: str) -> Path:
    return Path(cfg["paths"]["checkpoints_dir"]) / f"{model}_best.pt"


def available_base_models(cfg: dict) -> list[str]:
    models = cfg["models"]["supported"]
    return [m for m in models if metric_path(cfg, m, "test").exists() and checkpoint_path(cfg, m).exists()]


def choose_model(cfg: dict, requested: str | None) -> str:
    if requested:
        return requested
    candidates = available_base_models(cfg)
    if not candidates:
        raise FileNotFoundError("没有找到模型评估结果，请先完成训练和 evaluate.py。")
    best = max(candidates, key=lambda m: json.loads(metric_path(cfg, m, "test").read_text(encoding="utf-8"))["macro_f1"])
    return best


def load_metrics(cfg: dict, model: str, split: str) -> dict | None:
    path = metric_path(cfg, model, split)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(cfg: dict) -> tuple[Path, Path, Path, Path]:
    figures = Path(cfg["paths"]["figures_dir"])
    gradcam = Path(cfg["paths"]["gradcam_dir"])
    feature_maps = Path(cfg["paths"]["feature_maps_dir"])
    error_cases = Path(cfg["paths"]["error_cases_dir"])
    for path in [figures, gradcam, feature_maps, error_cases]:
        path.mkdir(parents=True, exist_ok=True)
    return figures, gradcam, feature_maps, error_cases


@torch.no_grad()
def predict_with_meta(model: torch.nn.Module, loader, device: torch.device) -> dict:
    y_true, y_pred, y_prob, rows = [], [], [], []
    for images, labels, meta in tqdm(loader, desc="Collecting predictions"):
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.append(probs)
        batch_size = len(labels)
        for i in range(batch_size):
            row = {}
            for key, value in meta.items():
                item = value[i]
                if hasattr(item, "item"):
                    item = item.item()
                row[key] = item
            rows.append(row)
    return {
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_prob": np.vstack(y_prob),
        "rows": rows,
    }


def plot_named_training_curves(cfg: dict, models: list[str], figures: Path) -> None:
    histories = []
    for model in models:
        path = Path(cfg["paths"]["logs_dir"]) / model / "history.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["model"] = model
            histories.append(df)
    if not histories:
        return
    data = pd.concat(histories, ignore_index=True)
    setup_plot_style()
    curve_specs = [
        ("train_loss", "Train Loss", "11_train_val_loss_curve.png"),
        ("val_accuracy", "Validation Accuracy", "12_val_accuracy_curve.png"),
        ("val_macro_f1", "Validation Macro-F1", "13_val_macro_f1_curve.png"),
        ("lr", "Learning Rate", "14_learning_rate_curve.png"),
    ]
    for col, title, name in curve_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model, group in data.groupby("model"):
            ax.plot(group["epoch"], group[col], marker="o", linewidth=1.5, label=model)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / name, dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in data.groupby("model"):
        loss_norm = group["train_loss"] / max(group["train_loss"].max(), 1e-8)
        gap = loss_norm - group["val_macro_f1"]
        ax.plot(group["epoch"], gap, marker="o", linewidth=1.5, label=model)
    ax.set_title("Overfitting Indicator")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Train Loss - Val Macro-F1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "15_overfitting_gap.png", dpi=220)
    plt.close(fig)


def count_params(model_name: str, num_classes: int) -> int:
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    return sum(p.numel() for p in model.parameters())


def plot_model_comparison(cfg: dict, models: list[str], figures: Path) -> None:
    rows = []
    for model in models:
        metrics = load_metrics(cfg, model, "test")
        if not metrics:
            continue
        rows.append(
            {
                "model": model,
                "Accuracy": metrics["accuracy"],
                "Balanced Acc": metrics["balanced_accuracy"],
                "Macro-F1": metrics["macro_f1"],
                "Weighted-F1": metrics["classification_report"]["weighted avg"]["f1-score"],
                "Inference time": metrics.get("seconds_per_image", 0.0),
                "Params": count_params(model, int(cfg["project"]["num_classes"])),
                "ECE": metrics.get("ece", 0.0),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2
    for offset, col in zip([-1.5, -0.5, 0.5, 1.5], ["Accuracy", "Balanced Acc", "Macro-F1", "Weighted-F1"]):
        ax.bar(x + offset * width, df[col], width=width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "16_model_comparison_metrics.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["Params"] / 1e6, df["Macro-F1"], s=90)
    for _, row in df.iterrows():
        ax.annotate(row["model"], (row["Params"] / 1e6, row["Macro-F1"]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Parameters vs Macro-F1")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "17_params_vs_f1.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["Inference time"], color="#10b981")
    ax.set_title("Inference Time Comparison")
    ax.set_ylabel("Seconds per image")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "18_inference_time_comparison.png", dpi=220)
    plt.close(fig)

    radar_cols = ["Accuracy", "Balanced Acc", "Macro-F1", "Weighted-F1"]
    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for _, row in df.iterrows():
        values = [row[col] for col in radar_cols]
        values += values[:1]
        ax.plot(angles, values, label=row["model"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_cols)
    ax.set_ylim(0, 1)
    ax.set_title("Model Radar Chart")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(figures / "19_model_radar_chart.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model"], df["ECE"], color="#ef4444")
    ax.set_title("Expected Calibration Error")
    ax.set_ylabel("ECE")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "38_expected_calibration_error.png", dpi=220)
    plt.close(fig)


def plot_internal_external(cfg: dict, models: list[str], figures: Path) -> None:
    rows = []
    for model in models:
        test = load_metrics(cfg, model, "test")
        ext = load_metrics(cfg, model, "external_test")
        if test and ext:
            rows.append({"model": model, "Internal test": test["macro_f1"], "External test": ext["macro_f1"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["Internal test"], width=width, label="Internal test")
    ax.bar(x + width / 2, df["External test"], width=width, label="External test")
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Internal vs External Macro-F1")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "23_internal_vs_external_metrics.png", dpi=220)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], figures: Path) -> None:
    setup_plot_style()
    y_bin = np.eye(len(class_names))[y_true]

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "26_multiclass_roc_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, label=f"{name} AUC={auc(recall, precision):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Multiclass Precision-Recall Curve")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "27_multiclass_pr_curve.png", dpi=220)
    plt.close(fig)


def plot_error_and_calibration(outputs: dict, class_names: list[str], figures: Path, error_dir: Path, max_errors: int) -> None:
    y_true = outputs["y_true"]
    y_pred = outputs["y_pred"]
    y_prob = outputs["y_prob"]
    rows = outputs["rows"]
    confidences = y_prob.max(axis=1)
    correct = y_true == y_pred

    setup_plot_style()
    cm = pd.crosstab(
        pd.Series([class_names[i] for i in y_true], name="true"),
        pd.Series([class_names[i] for i in y_pred], name="pred"),
    )
    pairs = []
    for true_name in cm.index:
        for pred_name in cm.columns:
            if true_name != pred_name:
                pairs.append((f"{true_name} -> {pred_name}", int(cm.loc[true_name, pred_name])))
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:10]
    if pairs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([p[0] for p in pairs][::-1], [p[1] for p in pairs][::-1], color="#f97316")
        ax.set_title("Top Confused Class Pairs")
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures / "35_top_confused_pairs.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidences[correct], bins=20, alpha=0.7, label="Correct")
    ax.hist(confidences[~correct], bins=20, alpha=0.7, label="Wrong")
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Images")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "36_confidence_distribution.png", dpi=220)
    plt.close(fig)

    bins = np.linspace(0, 1, 11)
    bin_centers, accs, confs = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_centers.append((lo + hi) / 2)
        accs.append(float(correct[mask].mean()))
        confs.append(float(confidences[mask].mean()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(confs, accs, marker="o", label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "37_reliability_diagram.png", dpi=220)
    plt.close(fig)

    wrong_idx = np.where(~correct)[0][:max_errors]
    if len(wrong_idx):
        cols = 4
        rows_n = math.ceil(len(wrong_idx) / cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3, rows_n * 3))
        axes = np.asarray(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, idx in zip(axes, wrong_idx):
            img = Image.open(rows[idx]["image_path"]).convert("RGB")
            ax.imshow(img)
            ax.set_title(
                f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]} {confidences[idx]:.2f}",
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(error_dir / "34_error_cases_grid.png", dpi=220)
        plt.close(fig)


def save_triplet_grid(items: list[tuple[Image.Image, Image.Image, Image.Image, str]], output_path: Path, cols: int = 3) -> None:
    if not items:
        return
    setup_plot_style()
    rows = len(items)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.6))
    axes = np.asarray(axes).reshape(rows, cols)
    for r, (original, heat, overlay, title) in enumerate(items):
        for c, (img, name) in enumerate([(original, "Original"), (heat, "Heatmap"), (overlay, "Overlay")]):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            axes[r, c].set_title(f"{title}\n{name}" if c == 0 else name, fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_gradcam_artifacts(
    model: torch.nn.Module,
    outputs: dict,
    class_names: list[str],
    image_size: int,
    device: torch.device,
    gradcam_dir: Path,
    per_class: int,
) -> None:
    y_true, y_pred, rows = outputs["y_true"], outputs["y_pred"], outputs["rows"]
    correct = y_true == y_pred
    correct_items = []
    summary_items = []
    for class_idx, class_name in enumerate(class_names):
        idxs = np.where(correct & (y_true == class_idx))[0][:per_class]
        for idx in idxs:
            original = Image.open(rows[idx]["image_path"]).convert("RGB")
            heat, overlay = make_gradcam_images(model, original, image_size, device, class_idx=int(y_pred[idx]))
            title = f"{class_name} / pred {class_names[y_pred[idx]]}"
            correct_items.append((original, heat, overlay, title))
            summary_items.append((original, heat, overlay, title))
    save_triplet_grid(correct_items, gradcam_dir / "28_gradcam_correct_cases.png")
    save_triplet_grid(summary_items, gradcam_dir / "30_gradcam_class_summary.png")

    wrong_idxs = np.where(~correct)[0][: max(1, per_class * len(class_names))]
    wrong_items = []
    for idx in wrong_idxs:
        original = Image.open(rows[idx]["image_path"]).convert("RGB")
        heat, overlay = make_gradcam_images(model, original, image_size, device, class_idx=int(y_pred[idx]))
        title = f"T:{class_names[y_true[idx]]} / P:{class_names[y_pred[idx]]}"
        wrong_items.append((original, heat, overlay, title))
    save_triplet_grid(wrong_items, gradcam_dir / "29_gradcam_wrong_cases.png")


def generate_feature_map(model: torch.nn.Module, outputs: dict, image_size: int, device: torch.device, feature_dir: Path) -> None:
    from src.datasets.transforms import build_inference_transform

    image = Image.open(outputs["rows"][0]["image_path"]).convert("RGB")
    x = build_inference_transform(image_size)(image).unsqueeze(0).to(device)
    captured = {}

    def hook(_module, _inputs, output):
        captured["feature"] = output.detach()

    handle = get_target_layer(model).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    feat = captured.get("feature")
    if feat is None:
        return
    if feat.ndim == 4 and feat.shape[-1] > feat.shape[1]:
        feat = feat.permute(0, 3, 1, 2)
    save_feature_map_grid(feat, feature_dir / "31_feature_maps.png")


@torch.no_grad()
def extract_features(model: torch.nn.Module, loader, device: torch.device, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    captured = {}

    def hook(_module, _inputs, output):
        feat = output.detach()
        if feat.ndim == 4 and feat.shape[-1] > feat.shape[1]:
            feat = feat.permute(0, 3, 1, 2)
        if feat.ndim == 4:
            feat = feat.mean(dim=(2, 3))
        elif feat.ndim == 3:
            feat = feat.mean(dim=1)
        captured["feature"] = feat

    handle = get_target_layer(model).register_forward_hook(hook)
    try:
        for images, y, _ in tqdm(loader, desc="Extracting features"):
            images = images.to(device, non_blocking=True)
            model(images)
            feat = captured["feature"].cpu().numpy()
            features.append(feat)
            labels.extend(y.numpy().tolist())
            if len(labels) >= max_samples:
                break
    finally:
        handle.remove()
    x = np.vstack(features)[:max_samples]
    y = np.asarray(labels[:max_samples])
    return x, y


def plot_embedding(points: np.ndarray, labels: np.ndarray, class_names: list[str], output_path: Path, title: str) -> None:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.65, label=name)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def generate_embeddings(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_samples: int,
    class_names: list[str],
    figures: Path,
) -> None:
    x, y = extract_features(model, loader, device, max_samples)
    if len(y) < 5:
        return
    x_pca = PCA(n_components=min(50, x.shape[1]), random_state=42).fit_transform(x)
    perplexity = min(30, max(5, len(y) // 10))
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
    plot_embedding(tsne.fit_transform(x_pca), y, class_names, figures / "32_tsne_feature_distribution.png", "t-SNE Feature Distribution")

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42)
        points = reducer.fit_transform(x)
        title = "UMAP Feature Distribution"
    except Exception:
        points = PCA(n_components=2, random_state=42).fit_transform(x)
        title = "PCA Feature Distribution (UMAP unavailable)"
    plot_embedding(points, y, class_names, figures / "33_umap_feature_distribution.png", title)


def generate_demo_screenshots(outputs: dict, class_names: list[str], figures: Path, gradcam_dir: Path, models: list[str], cfg: dict) -> None:
    idx = 0
    image = Image.open(outputs["rows"][idx]["image_path"]).convert("RGB")
    probs = outputs["y_prob"][idx]
    top = probs.argsort()[::-1][:3]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Streamlit Demo Upload Page")
    fig.tight_layout()
    fig.savefig(figures / "39_demo_upload_page.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image)
    axes[0].axis("off")
    axes[0].set_title("Uploaded Image")
    axes[1].barh([class_names[i] for i in top][::-1], [probs[i] for i in top][::-1], color="#3b82f6")
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Top-3 Prediction")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "40_demo_prediction_result.png", dpi=220)
    plt.close(fig)

    gradcam_file = gradcam_dir / "28_gradcam_correct_cases.png"
    if gradcam_file.exists():
        gradcam_img = Image.open(gradcam_file).convert("RGB")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.imshow(gradcam_img)
        ax.axis("off")
        ax.set_title("Streamlit Demo Grad-CAM Result")
        fig.tight_layout()
        fig.savefig(figures / "41_demo_gradcam_result.png", dpi=220)
        plt.close(fig)

    rows = []
    for model in models:
        metrics = load_metrics(cfg, model, "test")
        if metrics:
            rows.append((model, metrics["accuracy"], metrics["macro_f1"]))
    if rows:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        table = ax.table(
            cellText=[[m, f"{acc:.4f}", f"{f1:.4f}"] for m, acc, f1 in rows],
            colLabels=["Model", "Accuracy", "Macro-F1"],
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax.set_title("Demo Model Comparison")
        fig.tight_layout()
        fig.savefig(figures / "42_demo_model_compare.png", dpi=220)
        plt.close(fig)


def write_completion_report(cfg: dict, model: str, figures: Path, gradcam: Path, feature_maps: Path, error_cases: Path) -> None:
    reports = Path(cfg["paths"]["reports_dir"])
    report = reports / "full_analysis_artifacts.md"
    expected = [
        figures / "10_class_balance_before_after.png",
        figures / "11_train_val_loss_curve.png",
        figures / "12_val_accuracy_curve.png",
        figures / "13_val_macro_f1_curve.png",
        figures / "14_learning_rate_curve.png",
        figures / "15_overfitting_gap.png",
        figures / "16_model_comparison_metrics.png",
        figures / "17_params_vs_f1.png",
        figures / "18_inference_time_comparison.png",
        figures / "19_model_radar_chart.png",
        figures / "20_confusion_matrix_internal_raw.png",
        figures / "21_confusion_matrix_internal_normalized.png",
        figures / "22_confusion_matrix_scin.png",
        figures / "23_internal_vs_external_metrics.png",
        figures / "24_per_class_metrics.png",
        figures / "26_multiclass_roc_curve.png",
        figures / "27_multiclass_pr_curve.png",
        gradcam / "28_gradcam_correct_cases.png",
        gradcam / "29_gradcam_wrong_cases.png",
        gradcam / "30_gradcam_class_summary.png",
        feature_maps / "31_feature_maps.png",
        figures / "32_tsne_feature_distribution.png",
        figures / "33_umap_feature_distribution.png",
        error_cases / "34_error_cases_grid.png",
        figures / "35_top_confused_pairs.png",
        figures / "36_confidence_distribution.png",
        figures / "37_reliability_diagram.png",
        figures / "38_expected_calibration_error.png",
        figures / "39_demo_upload_page.png",
        figures / "40_demo_prediction_result.png",
        figures / "41_demo_gradcam_result.png",
        figures / "42_demo_model_compare.png",
        figures / "43_source_sample_grid.png",
        figures / "44_split_sample_grid.png",
        figures / "45_technical_route.png",
        figures / "46_system_architecture.png",
        figures / "47_data_processing_pipeline.png",
    ]
    lines = [f"# 完整分析产物检查", "", f"分析模型：`{model}`", "", "| 文件 | 状态 |", "|---|---|"]
    for path in expected:
        lines.append(f"| `{path}` | {'已生成' if path.exists() else '未生成/不适用'} |")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"完整分析产物检查报告：{report}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    figures, gradcam, feature_maps, error_cases = ensure_dirs(cfg)
    models = available_base_models(cfg)
    model_name = choose_model(cfg, args.model)
    checkpoint = checkpoint_path(cfg, model_name)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    class_names = cfg["project"]["class_names"]

    print("=" * 80)
    print("完整实验分析产物生成")
    print("=" * 80)
    print(f"使用模型：{model_name}")
    print(f"checkpoint：{checkpoint}")
    print(f"分析划分：{args.split}")
    print(f"运行设备：{device}")

    loader = build_eval_loader(cfg, args.split)
    model = load_checkpoint_model(checkpoint, int(cfg["project"]["num_classes"]), device)
    outputs = predict_with_meta(model, loader, device)

    plot_named_training_curves(cfg, models, figures)
    plot_model_comparison(cfg, models, figures)
    plot_internal_external(cfg, models, figures)

    metrics = load_metrics(cfg, model_name, args.split)
    if metrics:
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, figures / "20_confusion_matrix_internal_raw.png", normalize=False)
        plot_confusion_matrix(metrics["confusion_matrix"], class_names, figures / "21_confusion_matrix_internal_normalized.png", normalize=True)
        plot_per_class_metrics(metrics["per_class"], figures / "24_per_class_metrics.png")
    external_metrics = load_metrics(cfg, model_name, "external_test")
    if external_metrics:
        plot_confusion_matrix(
            external_metrics["confusion_matrix"],
            class_names,
            figures / "22_confusion_matrix_scin.png",
            normalize=True,
        )

    plot_roc_pr(outputs["y_true"], outputs["y_prob"], class_names, figures)
    plot_error_and_calibration(outputs, class_names, figures, error_cases, args.error_samples)
    generate_gradcam_artifacts(
        model,
        outputs,
        class_names,
        int(cfg["data"]["image_size"]),
        device,
        gradcam,
        args.gradcam_per_class,
    )
    generate_feature_map(model, outputs, int(cfg["data"]["image_size"]), device, feature_maps)
    generate_embeddings(model, loader, device, args.max_feature_samples, class_names, figures)
    generate_demo_screenshots(outputs, class_names, figures, gradcam, models, cfg)
    write_completion_report(cfg, model_name, figures, gradcam, feature_maps, error_cases)

    print("完整分析产物生成完成。")


if __name__ == "__main__":
    main()
