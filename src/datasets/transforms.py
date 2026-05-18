from __future__ import annotations

import random

import torch
from PIL import Image, ImageFilter
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BackgroundPerturbation:
    """Lightweight edge/background perturbation for skin images."""

    def __init__(self, p: float = 0.25) -> None:
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        image = image.convert("RGB")
        w, h = image.size
        border = max(4, int(min(w, h) * random.uniform(0.04, 0.10)))
        blurred = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2)))
        result = image.copy()
        result.paste(blurred.crop((0, 0, w, border)), (0, 0))
        result.paste(blurred.crop((0, h - border, w, h)), (0, h - border))
        result.paste(blurred.crop((0, 0, border, h)), (0, 0))
        result.paste(blurred.crop((w - border, 0, w, h)), (w - border, 0))
        return result


def build_train_transform(cfg: dict) -> transforms.Compose:
    image_size = int(cfg["data"]["image_size"])
    aug = cfg["augmentation"]
    return transforms.Compose(
        [
            BackgroundPerturbation(float(aug["background_perturb_p"])),
            transforms.RandomResizedCrop(
                image_size,
                scale=tuple(aug["random_resized_crop_scale"]),
            ),
            transforms.RandomHorizontalFlip(float(aug["horizontal_flip_p"])),
            transforms.ColorJitter(**aug["color_jitter"]),
            transforms.RandomRotation(float(aug["random_rotation_degrees"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=float(aug["random_erasing_p"]), value="random"),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_inference_transform(image_size: int) -> transforms.Compose:
    return build_eval_transform(image_size)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

