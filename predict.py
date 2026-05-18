from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.datasets.transforms import build_inference_transform
from src.models.build_model import load_checkpoint_model
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one image.")
    parser.add_argument("image")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint_model(args.checkpoint, int(cfg["project"]["num_classes"]), device)
    transform = build_inference_transform(int(cfg["data"]["image_size"]))

    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    class_names = cfg["project"]["class_names"]
    for prob, idx in zip(*torch.topk(probs, k=min(3, len(class_names)))):
        print(f"{class_names[int(idx)]}: {float(prob):.4f}")


if __name__ == "__main__":
    main()

