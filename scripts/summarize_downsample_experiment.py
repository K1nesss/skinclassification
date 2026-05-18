from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.visualization.matplotlib_style import setup_plot_style


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize train-only others downsample experiment.")
    parser.add_argument("--model", default="swin_b")
    parser.add_argument("--max-others", type=int, default=3000)
    parser.add_argument("--ratio", type=float, default=2.0)
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(name: str, prefix: str, metrics: dict | None) -> dict:
    if not metrics:
        return {
            "variant": name,
            "sampler": None,
            "prefix": prefix,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
            "others_recall": None,
            "per_class": {},
        }
    return {
        "variant": name,
        "sampler": "none" if "no sampler" in name.lower() else "weighted_sampler",
        "prefix": prefix,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
        "others_recall": metrics["per_class"]["others"]["recall"],
        "per_class": metrics["per_class"],
    }


def fmt(value) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def delta(value, base) -> str:
    if value is None or base is None:
        return "-"
    return f"{float(value) - float(base):+.4f}"


def write_report(rows: list[dict], model: str, reports_dir: Path) -> Path:
    out = reports_dir / f"{model}_train_downsample_comparison_summary.md"
    base = rows[0]
    class_names = ["acne", "eczema", "dermatitis", "pigmentation", "others"]
    lines = [
        f"# {model} train-only others 下采样三组对比实验",
        "",
        "实验设置：暂不使用 SCIN；数据来源为 Dermnet、Mendeley、SkinDisNet；先固定划分 train/val/test，再只对 `train` 中的 `others` 下采样；`val/test` 完全保持不变；各方案的训练采样策略见表，不启用 `loss_class_weights`。",
        "",
        "## 总体指标",
        "",
        "| 方案 | 训练采样策略 | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro-F1 | Weighted-F1 | others Recall | Macro-F1变化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row.get('sampler') or '-'} | {fmt(row['accuracy'])} | {fmt(row['balanced_accuracy'])} | "
            f"{fmt(row['macro_precision'])} | {fmt(row['macro_recall'])} | {fmt(row['macro_f1'])} | "
            f"{fmt(row['weighted_f1'])} | {fmt(row['others_recall'])} | {delta(row['macro_f1'], base['macro_f1'])} |"
        )

    for metric_name, key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]:
        lines.extend(
            [
                "",
                f"## 每类 {metric_name}",
                "",
                "| 类别 | " + " | ".join(row["variant"] for row in rows) + " |",
                "|---" + "|---:" * len(rows) + "|",
            ]
        )
        for class_name in class_names:
            values = []
            for row in rows:
                item = row["per_class"].get(class_name, {})
                values.append(fmt(item.get(key)))
            lines.append(f"| {class_name} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 该实验的 `test` 集保持一致，因此可以用于严格比较训练阶段下采样策略是否有效。",
            "- 如果目标是让每个类别都能被更稳定地识别，应优先观察 Macro-F1、Balanced Accuracy、每类 Recall 和每类 F1，而不是只看 Accuracy。",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def plot_comparison(rows: list[dict], model: str, figures_dir: Path) -> Path:
    out = figures_dir / f"{model}_train_downsample_comparison.png"
    data = [row for row in rows if row["macro_f1"] is not None]
    if not data:
        return out
    setup_plot_style()
    labels = [row["variant"] for row in data]
    x = range(len(data))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 1.5 * width for i in x], [row["accuracy"] for row in data], width=width, label="Accuracy")
    ax.bar([i - 0.5 * width for i in x], [row["balanced_accuracy"] for row in data], width=width, label="Balanced Acc")
    ax.bar([i + 0.5 * width for i in x], [row["macro_f1"] for row in data], width=width, label="Macro-F1")
    ax.bar([i + 1.5 * width for i in x], [row["weighted_f1"] for row in data], width=width, label="Weighted-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model} Train-only Others Downsample Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    reports_dir = Path("outputs/reports")
    figures_dir = Path("outputs/figures")
    variants = [
        ("No downsample", f"{args.model}_no_downsample"),
        (f"Train others <= {args.max_others}/source", f"{args.model}_train_downsample_others{args.max_others}"),
        (f"Train others = {args.ratio:g}x max class (no sampler)", f"{args.model}_train_downsample_others_ratio2x"),
    ]
    rows = []
    for label, prefix in variants:
        metrics = load_json(reports_dir / f"{prefix}_best_test_metrics.json")
        rows.append(metric_row(label, prefix, metrics))

    report = write_report(rows, args.model, reports_dir)
    figure = plot_comparison(rows, args.model, figures_dir)
    print(f"train-only 下采样对比报告：{report}")
    print(f"train-only 下采样对比图：{figure}")


if __name__ == "__main__":
    main()
