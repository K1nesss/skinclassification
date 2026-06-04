from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.datasets.class_balance import (
    make_class_aware_sampler,
    make_class_weights,
    make_weighted_sampler,
)
from src.datasets.transforms import build_eval_transform, build_train_transform


class SkinManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_id"]), row.to_dict()


def _load_manifest(cfg: dict) -> pd.DataFrame:
    manifest_path = Path(cfg["paths"]["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Run scripts/build_new_dataset.py first."
        )
    return pd.read_csv(manifest_path)


def build_dataloaders(cfg: dict):
    df = _load_manifest(cfg)
    class_names = cfg["project"]["class_names"]
    num_classes = len(class_names)
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    image_size = int(cfg["data"]["image_size"])

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Train/val splits are empty. Rebuild the dataset manifest.")

    train_dataset = SkinManifestDataset(train_df, build_train_transform(cfg))
    val_dataset = SkinManifestDataset(val_df, build_eval_transform(image_size))
    test_dataset = SkinManifestDataset(test_df, build_eval_transform(image_size))

    sampler = None
    shuffle = True
    labels = train_df["label_id"].astype(int).tolist()
    balance_strategy = cfg["training"].get("balance_strategy")
    if balance_strategy == "weighted_sampler":
        sampler = make_weighted_sampler(labels, num_classes)
        shuffle = False
    elif balance_strategy == "class_aware_sampler":
        hard_classes = cfg["training"].get("hard_classes", [])
        hard_label_ids = {
            class_names.index(label)
            for label in hard_classes
            if label in class_names
        }
        if not hard_label_ids:
            raise ValueError("class_aware_sampler requires training.hard_classes.")
        sampler = make_class_aware_sampler(
            labels,
            num_classes,
            hard_label_ids,
            float(cfg["training"].get("hard_class_multiplier", 1.5)),
        )
        shuffle = False

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    class_weights = make_class_weights(train_df["label_id"].astype(int).tolist(), num_classes)
    return loaders, class_weights


def build_eval_loader(cfg: dict, split: str) -> DataLoader:
    df = _load_manifest(cfg)
    split_df = df[df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No samples found for split: {split}")
    dataset = SkinManifestDataset(
        split_df,
        build_eval_transform(int(cfg["data"]["image_size"])),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        pin_memory=True,
    )
