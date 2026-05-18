from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import yaml

from src.utils.io import load_config
from src.visualization.matplotlib_style import setup_plot_style


DEFAULT_AUG = {
    "random_resized_crop_scale": [0.75, 1.0],
    "horizontal_flip_p": 0.5,
    "color_jitter": {"brightness": 0.18, "contrast": 0.18, "saturation": 0.12, "hue": 0.03},
    "random_rotation_degrees": 12,
    "random_erasing_p": 0.15,
    "background_perturb_p": 0.25,
}

WEAK_AUG = {
    "random_resized_crop_scale": [0.90, 1.0],
    "horizontal_flip_p": 0.5,
    "color_jitter": {"brightness": 0.08, "contrast": 0.08, "saturation": 0.05, "hue": 0.01},
    "random_rotation_degrees": 5,
    "random_erasing_p": 0.05,
    "background_perturb_p": 0.10,
}

STRONG_AUG = {
    "random_resized_crop_scale": [0.60, 1.0],
    "horizontal_flip_p": 0.5,
    "color_jitter": {"brightness": 0.30, "contrast": 0.30, "saturation": 0.22, "hue": 0.05},
    "random_rotation_degrees": 18,
    "random_erasing_p": 0.25,
    "background_perturb_p": 0.35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hyperparameter ablation experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="convnext_base")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--include-image-size", action="store_true", default=True)
    parser.add_argument("--no-image-size", action="store_false", dest="include_image_size")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("\n" + "-" * 80)
    print("执行命令：" + " ".join(cmd))
    print("-" * 80)
    code = subprocess.call(cmd)
    if code != 0:
        raise SystemExit(code)


def build_experiments(cfg: dict, include_image_size: bool) -> list[dict]:
    base_image_size = int(cfg["data"]["image_size"])
    base_lr = float(cfg["training"]["learning_rate"])
    base_wd = float(cfg["training"]["weight_decay"])

    experiments: list[dict] = []
    if include_image_size:
        for value in [224, 256, 320]:
            experiments.append(
                {
                    "group": "image_size",
                    "value": str(value),
                    "suffix": f"image_size_{value}",
                    "updates": {"data": {"image_size": value}},
                }
            )
    else:
        experiments.append(
            {
                "group": "image_size",
                "value": str(base_image_size),
                "suffix": f"image_size_{base_image_size}",
                "updates": {"data": {"image_size": base_image_size}},
            }
        )

    for value in [5e-5, 1e-4, 3e-4]:
        experiments.append(
            {
                "group": "learning_rate",
                "value": f"{value:g}",
                "suffix": f"lr_{str(value).replace('-', 'm').replace('.', 'p')}",
                "updates": {"training": {"learning_rate": float(value)}},
            }
        )

    for value in [0.001, 0.01, 0.05]:
        experiments.append(
            {
                "group": "weight_decay",
                "value": f"{value:g}",
                "suffix": f"wd_{str(value).replace('.', 'p')}",
                "updates": {"training": {"weight_decay": float(value)}},
            }
        )

    for value, aug in [("weak", WEAK_AUG), ("default", DEFAULT_AUG), ("strong", STRONG_AUG)]:
        experiments.append(
            {
                "group": "augmentation",
                "value": value,
                "suffix": f"aug_{value}",
                "updates": {"augmentation": aug},
            }
        )

    # Remove duplicated baseline runs across groups only by suffix. Keeping the
    # same numerical baseline in different groups makes each chart self-contained.
    for exp in experiments:
        exp["base_image_size"] = base_image_size
        exp["base_lr"] = base_lr
        exp["base_weight_decay"] = base_wd
    return experiments


def deep_update(target: dict, updates: dict) -> dict:
    result = copy.deepcopy(target)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def write_temp_config(cfg: dict, exp: dict, model: str, epochs: int, batch_size: int | None) -> Path:
    updated = deep_update(cfg, exp["updates"])
    updated["training"]["model"] = model
    updated["training"]["epochs"] = epochs
    if batch_size:
        updated["training"]["batch_size"] = batch_size
    temp_dir = Path(tempfile.mkdtemp(prefix="skin_hparam_"))
    path = temp_dir / f"{model}_{exp['suffix']}.yaml"
    path.write_text(yaml.safe_dump(updated, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def train_and_eval(args: argparse.Namespace, cfg: dict, experiments: list[dict]) -> list[dict]:
    rows = []
    reports_dir = Path(cfg["paths"]["reports_dir"])
    checkpoints_dir = Path(cfg["paths"]["checkpoints_dir"])
    for exp in experiments:
        run_name = f"{args.model}_hparam_{exp['suffix']}"
        checkpoint = checkpoints_dir / f"{run_name}_best.pt"
        result_path = reports_dir / f"{run_name}_best_test_metrics.json"
        temp_config = write_temp_config(cfg, exp, args.model, args.epochs, args.batch_size)

        if not (args.skip_existing and checkpoint.exists()):
            train_cmd = [
                sys.executable,
                "train.py",
                "--config",
                str(temp_config),
                "--model",
                args.model,
                "--run-name",
                run_name,
            ]
            if args.batch_size:
                train_cmd.extend(["--batch-size", str(args.batch_size)])
            if args.device:
                train_cmd.extend(["--device", args.device])
            run_cmd(train_cmd)
        else:
            print(f"已存在 checkpoint，跳过训练：{checkpoint}")

        if not (args.skip_existing and result_path.exists()):
            eval_cmd = [
                sys.executable,
                "evaluate.py",
                "--config",
                str(temp_config),
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
            ]
            if args.batch_size:
                eval_cmd.extend(["--batch-size", str(args.batch_size)])
            if args.device:
                eval_cmd.extend(["--device", args.device])
            run_cmd(eval_cmd)
        else:
            print(f"已存在评估结果，跳过评估：{result_path}")

        metrics = json.loads(result_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "group": exp["group"],
                "value": exp["value"],
                "run_name": run_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
                "seconds_per_image": metrics.get("seconds_per_image", 0.0),
            }
        )
    return rows


def plot_group(rows: list[dict], group: str, output_path: Path, title: str) -> None:
    data = [row for row in rows if row["group"] == group]
    if not data:
        return
    setup_plot_style()
    labels = [row["value"] for row in data]
    x = range(len(data))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width for i in x], [row["accuracy"] for row in data], width=width, label="Accuracy")
    ax.bar(x, [row["balanced_accuracy"] for row in data], width=width, label="Balanced Acc")
    ax.bar([i + width for i in x], [row["macro_f1"] for row in data], width=width, label="Macro-F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_outputs(rows: list[dict], cfg: dict, model: str) -> None:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"{model}_hparam_ablation_results.csv"
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = reports_dir / f"{model}_hparam_ablation_summary.md"
    lines = [
        f"# {model} 超参数消融实验",
        "",
        "| 实验组 | 参数值 | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | 单张推理耗时/s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['value']} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} | {row['seconds_per_image']:.6f} |"
        )
    best = max(rows, key=lambda row: row["macro_f1"])
    lines.extend(
        [
            "",
            f"最佳 Macro-F1 配置：`{best['group']}={best['value']}`，Macro-F1={best['macro_f1']:.4f}。",
            "",
            "说明：该脚本默认只使用一个主力模型进行超参数实验，避免所有模型全量网格搜索导致训练时间过长。",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    plot_group(rows, "image_size", figures_dir / "48_hparam_image_size_ablation.png", "Image Size Ablation")
    plot_group(rows, "learning_rate", figures_dir / "49_hparam_learning_rate_ablation.png", "Learning Rate Ablation")
    plot_group(rows, "weight_decay", figures_dir / "50_hparam_weight_decay_ablation.png", "Weight Decay Ablation")
    plot_group(rows, "augmentation", figures_dir / "51_hparam_augmentation_ablation.png", "Augmentation Strength Ablation")

    print(f"超参数实验 CSV：{csv_path}")
    print(f"超参数实验报告：{md_path}")
    print("超参数实验图表：")
    print(f"  {figures_dir / '48_hparam_image_size_ablation.png'}")
    print(f"  {figures_dir / '49_hparam_learning_rate_ablation.png'}")
    print(f"  {figures_dir / '50_hparam_weight_decay_ablation.png'}")
    print(f"  {figures_dir / '51_hparam_augmentation_ablation.png'}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    experiments = build_experiments(cfg, args.include_image_size)
    print("=" * 80)
    print("超参数消融实验")
    print("=" * 80)
    print(f"模型：{args.model}")
    print(f"训练轮数：{args.epochs}")
    print(f"实验数量：{len(experiments)}")
    rows = train_and_eval(args, cfg, experiments)
    write_outputs(rows, cfg, args.model)


if __name__ == "__main__":
    main()
