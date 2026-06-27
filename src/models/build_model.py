from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


TORCHVISION_MODELS = {
    "resnet18",
    "densenet121",
    "efficientnet_b0",
    "mobilenet_v3_small",
    "convnext_tiny",
    "convnext_base",
    "swin_s",
    "swin_b",
}


def _is_timm_model(model_name: str) -> bool:
    return model_name.startswith("timm:")


def _timm_model_name(model_name: str) -> str:
    return model_name.removeprefix("timm:")


def _weights(model_name: str, pretrained: bool):
    if not pretrained:
        return None
    enum_by_name = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_base": models.ConvNeXt_Base_Weights.DEFAULT,
        "swin_s": models.Swin_S_Weights.DEFAULT,
        "swin_b": models.Swin_B_Weights.DEFAULT,
    }
    return enum_by_name[model_name]


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    if _is_timm_model(model_name):
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "Model names prefixed with 'timm:' require the timm package. "
                "Install project requirements or run: pip install timm"
            ) from exc
        model = timm.create_model(
            _timm_model_name(model_name),
            pretrained=pretrained,
            num_classes=num_classes,
        )
        model.model_name = model_name
        return model

    try:
        weights = _weights(model_name, pretrained)
        model = getattr(models, model_name)(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        print(f"Could not load pretrained weights for {model_name}: {exc}")
        print("Falling back to randomly initialized weights.")
        model = getattr(models, model_name)(weights=None)

    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith("swin"):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.model_name = model_name
    return model


def checkpoint_model_name(checkpoint: str | Path) -> str:
    data = torch.load(checkpoint, map_location="cpu")
    if "model_name" in data:
        return data["model_name"]
    stem = Path(checkpoint).stem.lower()
    for name in [
        "mobilenet_v3_small",
        "efficientnet_b0",
        "convnext_tiny",
        "convnext_base",
        "densenet121",
        "resnet18",
        "swin_s",
        "swin_b",
    ]:
        if name in stem:
            return name
    raise ValueError("Checkpoint does not contain model_name and filename is ambiguous.")


def load_checkpoint_model(
    checkpoint: str | Path,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    data = torch.load(checkpoint, map_location=device)
    model_name = data.get("model_name") or checkpoint_model_name(checkpoint)
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    state = data.get("model_state", data)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_target_layer(model: nn.Module) -> nn.Module:
    name = getattr(model, "model_name", model.__class__.__name__.lower())
    if _is_timm_model(name):
        if hasattr(model, "stages"):
            return model.stages[-1]
        if hasattr(model, "blocks"):
            return model.blocks[-1]
        if hasattr(model, "layers"):
            return model.layers[-1]
        raise ValueError(f"No Grad-CAM target layer configured for {name}")
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("densenet"):
        return model.features.denseblock4
    if name.startswith("efficientnet"):
        return model.features[-1]
    if name.startswith("mobilenet"):
        return model.features[-1]
    if name.startswith("convnext"):
        return model.features[-1]
    if name.startswith("swin"):
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer configured for {name}")
