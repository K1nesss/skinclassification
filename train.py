from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.datasets.skin_dataset import build_dataloaders
from src.engine.trainer import train_model
from src.models.build_model import build_model
from src.utils.io import ensure_project_dirs, load_config
from src.utils.seed import seed_everything


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练皮肤疾病分类模型。")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--manifest", default=None, help="Override paths.manifest for this run.")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run-name", default=None, help="Optional run id used for logs/checkpoints.")
    parser.add_argument(
        "--balance-strategy",
        choices=["none", "weighted_sampler", "class_aware_sampler"],
        default=None,
        help="Override class balance strategy.",
    )
    parser.add_argument(
        "--loss-class-weights",
        action="store_true",
        help="Enable class-weighted CrossEntropyLoss.",
    )
    parser.add_argument(
        "--no-loss-class-weights",
        action="store_true",
        help="Disable class-weighted CrossEntropyLoss.",
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Accumulate gradients for this many mini-batches before each optimizer step.",
    )
    parser.add_argument("--loss-type", choices=["cross_entropy", "focal"], default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--hard-class-multiplier", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["seed"]))

    if args.model:
        cfg["training"]["model"] = args.model
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.image_size:
        cfg["data"]["image_size"] = args.image_size
    if args.manifest:
        cfg["paths"]["manifest"] = args.manifest
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.run_name:
        cfg["training"]["run_name"] = args.run_name
    if args.balance_strategy:
        cfg["training"]["balance_strategy"] = args.balance_strategy
    if args.loss_class_weights:
        cfg["training"]["loss_class_weights"] = True
    if args.no_loss_class_weights:
        cfg["training"]["loss_class_weights"] = False
    if args.grad_accum_steps:
        cfg["training"]["gradient_accumulation_steps"] = args.grad_accum_steps
    if args.loss_type:
        cfg["training"]["loss_type"] = args.loss_type
    if args.label_smoothing is not None:
        cfg["training"]["label_smoothing"] = args.label_smoothing
    if args.focal_gamma is not None:
        cfg["training"]["focal_gamma"] = args.focal_gamma
    if args.hard_class_multiplier is not None:
        cfg["training"]["hard_class_multiplier"] = args.hard_class_multiplier

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print_section("训练配置")
    print(f"配置文件：{args.config}")
    print(f"模型：{cfg['training']['model']}")
    print(f"运行设备：{device}")
    if device.type == "cuda":
        print(f"CUDA 显卡：{torch.cuda.get_device_name(device)}")
    print(f"训练轮数：{cfg['training']['epochs']}")
    print(f"批大小：{cfg['training']['batch_size']}")
    print(f"输入尺寸：{cfg['data']['image_size']}")
    print(f"学习率：{cfg['training']['learning_rate']}")
    print(f"权重衰减：{cfg['training']['weight_decay']}")
    print(f"类别平衡策略：{cfg['training']['balance_strategy']}")
    if cfg["training"].get("balance_strategy") == "class_aware_sampler":
        print(f"重点类别：{', '.join(cfg['training'].get('hard_classes', []))}")
        print(f"重点类别采样倍率：{cfg['training'].get('hard_class_multiplier', 1.5)}")
    print(f"是否按类别加权损失：{cfg['training'].get('loss_class_weights', False)}")
    print(f"梯度累积步数：{cfg['training'].get('gradient_accumulation_steps', 1)}")
    print(f"是否启用 AMP 混合精度：{cfg['training'].get('amp', True)}")
    print(f"样本清单：{Path(cfg['paths']['manifest']).resolve()}")

    loaders, class_weights = build_dataloaders(cfg)
    model = build_model(
        cfg["training"]["model"],
        num_classes=int(cfg["project"]["num_classes"]),
        pretrained=bool(cfg["training"]["pretrained"]),
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量：{total_params:,}")
    print(f"可训练参数量：{trainable_params:,}")

    run_dir = Path(cfg["paths"]["logs_dir"]) / cfg["training"].get("run_name", cfg["training"]["model"])
    train_model(
        cfg=cfg,
        model=model,
        loaders=loaders,
        class_weights=class_weights,
        device=device,
        run_dir=run_dir,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )


if __name__ == "__main__":
    main()
