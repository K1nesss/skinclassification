from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.datasets.transforms import build_inference_transform
from src.models.build_model import get_target_layer
from src.visualization.matplotlib_style import setup_plot_style


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()
        acts = self.activations
        grads = self.gradients
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations.")
        if acts.ndim == 4 and acts.shape[-1] > acts.shape[1]:
            # Handles channels-last features from some transformer-style modules.
            acts = acts.permute(0, 3, 1, 2)
            grads = grads.permute(0, 3, 1, 2)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"))
    cam_resized = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * rgb + alpha * heatmap)
    return Image.fromarray(overlay)


def make_gradcam_images(model, image: Image.Image, image_size: int, device: torch.device, class_idx: int | None = None):
    transform = build_inference_transform(image_size)
    x = transform(image).unsqueeze(0).to(device)
    cam = GradCAM(model, get_target_layer(model))
    try:
        heat = cam(x, class_idx=class_idx)
    finally:
        cam.close()
    overlay = overlay_cam(image, heat)
    heat_rgb = Image.fromarray(cv2.applyColorMap(np.uint8(255 * cv2.resize(heat, image.size)), cv2.COLORMAP_JET)[:, :, ::-1])
    return heat_rgb, overlay


def save_gradcam_triplet(original: Image.Image, heat: Image.Image, overlay: Image.Image, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    setup_plot_style()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(axes, [original, heat, overlay], ["Original", "Heatmap", "Overlay"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
