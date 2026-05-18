from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt

from src.visualization.matplotlib_style import setup_plot_style
from src.utils.io import load_config


SCHEMES = [
    ("plain", "none", False, "No balance"),
    ("weighted_sampler", "weighted_sampler", False, "Weighted sampler"),
    ("class_weight", "none", True, "Class-weighted loss"),
    ("weighted_sampler_class_weight", "weighted_sampler", True, "Sampler + class weight"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run class-balance ablation experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="convnext_base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "-" * 80)
    print("执行命令：" + " ".join(cmd))
    print("-" * 80)
    code = subprocess.call(cmd)
    if code != 0:
        raise SystemExit(code)


def train_and_evaluate(args: argparse.Namespace, cfg: dict) -> list[dict]:
    results = []
    reports_dir = Path(cfg["paths"]["reports_dir"])
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    for suffix, balance_strategy, loss_weights, label in SCHEMES:
        run_name = f"{args.model}_ablation_{suffix}"
        checkpoint = checkpoints_dir / f"{run_name}_best.pt"
        result_path = reports_dir / f"{run_name}_best_test_metrics.json"

        if not (args.skip_existing and checkpoint.exists()):
            train_cmd = [
                sys.executable,
                "train.py",
                "--config",
                args.config,
                "--model",
                args.model,
                "--run-name",
                run_name,
                "--balance-strategy",
                balance_strategy,
            ]
            if loss_weights:
                train_cmd.append("--loss-class-weights")
            else:
                train_cmd.append("--no-loss-class-weights")
            if args.epochs:
                train_cmd.extend(["--epochs", str(args.epochs)])
            if args.batch_size:
                train_cmd.extend(["--batch-size", str(args.batch_size)])
            if args.device:
                train_cmd.extend(["--device", args.device])
            run_cmd(train_cmd)
        else:
            print(f"已存在 checkpoint，跳过训练：{checkpoint}")

        eval_cmd = [
            sys.executable,
            "evaluate.py",
            "--config",
            args.config,
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
        ]
        if args.batch_size:
            eval_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.device:
            eval_cmd.extend(["--device", args.device])
        if not (args.skip_existing and result_path.exists()):
            run_cmd(eval_cmd)
        else:
            print(f"已存在评估结果，跳过评估：{result_path}")

        metrics = json.loads(result_path.read_text(encoding="utf-8"))
        results.append(
            {
                "scheme": suffix,
                "label": label,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
            }
        )
    return results


def write_summary(results: list[dict], cfg: dict, model: str) -> None:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    md_path = reports_dir / f"{model}_balance_ablation_summary.md"
    lines = [
        f"# {model} 类别平衡消融实验",
        "",
        "| 方案 | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['label']} | {item['accuracy']:.4f} | {item['balanced_accuracy']:.4f} | "
            f"{item['macro_f1']:.4f} | {item['weighted_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "说明：该实验固定模型结构，只改变类别不平衡处理策略，用于分析采样平衡和损失加权对分类性能的影响。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    setup_plot_style()
    labels = [item["label"] for item in results]
    x = range(len(results))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    ax.bar([i - width for i in x], [item["accuracy"] for item in results], width=width, label="Accuracy")
    ax.bar(x, [item["balanced_accuracy"] for item in results], width=width, label="Balanced Acc")
    ax.bar([i + width for i in x], [item["macro_f1"] for item in results], width=width, label="Macro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Class Balance Ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "10_class_balance_before_after.png", dpi=220)
    fig.savefig(figures_dir / "25_balanced_accuracy_comparison.png", dpi=220)
    plt.close(fig)

    print(f"类别平衡消融报告：{md_path}")
    print(f"类别平衡消融图：{figures_dir / '10_class_balance_before_after.png'}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print("=" * 80)
    print("类别平衡消融实验")
    print("=" * 80)
    print(f"模型：{args.model}")
    print("方案：普通采样、WeightedRandomSampler、class weight、二者同时使用")
    results = train_and_evaluate(args, cfg)
    write_summary(results, cfg, args.model)


if __name__ == "__main__":
    main()
