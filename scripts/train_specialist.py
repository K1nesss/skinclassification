from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.class_balance import make_class_weights, make_weighted_sampler
from src.datasets.skin_dataset import SkinManifestDataset
from src.datasets.transforms import build_eval_transform, build_train_transform
from src.engine.trainer import train_model
from src.models.build_model import build_model
from src.utils.io import ensure_project_dirs, load_config
from src.utils.seed import seed_everything


DEFAULT_SPECIALIST_CLASSES = [
    "eczema",
    "dermatitis",
    "psoriasis_lichen_planus",
    "fungal_infection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a specialist classifier for confusing classes.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="convnext_base")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_SPECIALIST_CLASSES)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--balance-strategy",
        choices=["none", "weighted_sampler"],
        default="weighted_sampler",
    )
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def remap_subset(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    subset = df[df["label"].isin(classes)].copy()
    if subset.empty:
        raise ValueError(f"No samples found for specialist classes: {classes}")
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    subset["original_label_id"] = subset["label_id"]
    subset["label_id"] = subset["label"].map(label_to_id).astype(int)
    return subset


def make_loader(
    df: pd.DataFrame,
    cfg: dict,
    split: str,
    batch_size: int,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    transform = build_train_transform(cfg) if split == "train" else build_eval_transform(int(cfg["data"]["image_size"]))
    return DataLoader(
        SkinManifestDataset(df, transform),
        batch_size=batch_size,
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_specialist_dataloaders(cfg: dict, classes: list[str], balance_strategy: str):
    manifest = pd.read_csv(cfg["paths"]["manifest"])
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))

    train_df = remap_subset(manifest[manifest["split"] == "train"], classes)
    val_df = remap_subset(manifest[manifest["split"] == "val"], classes)
    test_df = remap_subset(manifest[manifest["split"] == "test"], classes)

    labels = train_df["label_id"].astype(int).tolist()
    sampler = None
    if balance_strategy == "weighted_sampler":
        sampler = make_weighted_sampler(labels, len(classes))

    loaders = {
        "train": make_loader(train_df, cfg, "train", batch_size, num_workers, sampler=sampler),
        "val": make_loader(val_df, cfg, "val", batch_size, num_workers),
        "test": make_loader(test_df, cfg, "test", batch_size, num_workers),
    }
    return loaders, make_class_weights(labels, len(classes))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    seed_everything(int(cfg["project"]["seed"]))

    unknown = [name for name in args.classes if name not in cfg["project"]["class_names"]]
    if unknown:
        raise ValueError(f"Unknown specialist classes: {unknown}")

    cfg["project"]["class_names"] = list(args.classes)
    cfg["project"]["num_classes"] = len(args.classes)
    cfg["training"]["model"] = args.model
    cfg["training"]["run_name"] = args.run_name or f"{args.model}_specialist"
    cfg["training"]["balance_strategy"] = args.balance_strategy
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\n" + "=" * 80)
    print("相似类专门模型训练配置")
    print("=" * 80)
    print(f"配置文件：{args.config}")
    print(f"模型：{cfg['training']['model']}")
    print(f"运行名称：{cfg['training']['run_name']}")
    print(f"专门类别：{', '.join(args.classes)}")
    print(f"运行设备：{device}")
    if device.type == "cuda":
        print(f"CUDA 显卡：{torch.cuda.get_device_name(device)}")
    print(f"训练轮数：{cfg['training']['epochs']}")
    print(f"批大小：{cfg['training']['batch_size']}")
    print(f"类别平衡策略：{args.balance_strategy}")

    loaders, class_weights = build_specialist_dataloaders(cfg, list(args.classes), args.balance_strategy)
    model = build_model(
        cfg["training"]["model"],
        num_classes=len(args.classes),
        pretrained=bool(cfg["training"]["pretrained"]),
    )
    run_dir = Path(cfg["paths"]["logs_dir"]) / cfg["training"]["run_name"]
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
