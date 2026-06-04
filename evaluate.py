from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.datasets.skin_dataset import build_eval_loader
from src.engine.evaluator import evaluate_checkpoint
from src.utils.io import ensure_project_dirs, load_config


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估训练好的模型 checkpoint。")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.image_size:
        cfg["data"]["image_size"] = args.image_size
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print_section("评估配置")
    print(f"配置文件：{args.config}")
    print(f"checkpoint：{Path(args.checkpoint).resolve()}")
    print(f"评估划分：{args.split}")
    print(f"运行设备：{device}")
    if device.type == "cuda":
        print(f"CUDA 显卡：{torch.cuda.get_device_name(device)}")
    print(f"批大小：{cfg['training']['batch_size']}")
    print(f"输入尺寸：{cfg['data']['image_size']}")
    print(f"样本清单：{Path(cfg['paths']['manifest']).resolve()}")
    loader = build_eval_loader(cfg, split=args.split)
    print(f"评估样本数：{len(loader.dataset)}，batch 数：{len(loader)}")
    result_path = Path(cfg["paths"]["reports_dir"]) / f"{Path(args.checkpoint).stem}_{args.split}_metrics.json"
    evaluate_checkpoint(cfg, args.checkpoint, loader, device, result_path)


if __name__ == "__main__":
    main()
