from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from app.disease_info import CLASS_INFO, DISCLAIMER
from src.datasets.transforms import build_inference_transform
from src.models.build_model import load_checkpoint_model
from src.utils.io import load_config
from src.visualization.gradcam_utils import make_gradcam_images


def parse_cli_config() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args, _ = parser.parse_known_args()
    return args.config


@st.cache_resource
def cached_model(checkpoint: str, num_classes: int, device_name: str):
    return load_checkpoint_model(checkpoint, num_classes, torch.device(device_name))


def checkpoint_options(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("*_best.pt"))


def main() -> None:
    cfg = load_config(parse_cli_config())
    st.set_page_config(page_title="Skin Classification Demo", layout="wide")
    st.title("Skin Disease Classification")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    ckpts = checkpoint_options(Path(cfg["paths"]["checkpoints_dir"]))
    with st.sidebar:
        st.header("Model")
        if ckpts:
            chosen = st.selectbox("Checkpoint", ckpts, format_func=lambda p: p.name)
        else:
            st.warning("No checkpoint found in outputs/checkpoints.")
            chosen = None
        st.caption(f"Device: {device_name}")

    tabs = st.tabs(["Predict", "Experiment Results", "Notice"])
    with tabs[0]:
        uploaded = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded and chosen:
            image = Image.open(uploaded).convert("RGB")
            model = cached_model(str(chosen), int(cfg["project"]["num_classes"]), device_name)
            transform = build_inference_transform(int(cfg["data"]["image_size"]))
            x = transform(image).unsqueeze(0).to(device_name)
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)[0].cpu()
            class_names = cfg["project"]["class_names"]
            top_probs, top_idx = torch.topk(probs, k=min(3, len(class_names)))
            pred = class_names[int(top_idx[0])]
            conf = float(top_probs[0])

            left, right = st.columns([1, 1])
            with left:
                st.image(image, caption="Input image", use_container_width=True)
            with right:
                st.metric("Prediction", pred, f"{conf:.2%}")
                for prob, idx in zip(top_probs, top_idx):
                    label = class_names[int(idx)]
                    st.progress(float(prob), text=f"{label}: {float(prob):.2%}")
                if conf < float(cfg["evaluation"]["confidence_threshold"]):
                    st.warning("Low confidence. Treat the result as uncertain.")
                st.info(CLASS_INFO.get(pred, "No class note."))

            with st.spinner("Generating Grad-CAM"):
                heat, overlay = make_gradcam_images(
                    model,
                    image,
                    int(cfg["data"]["image_size"]),
                    torch.device(device_name),
                    class_idx=int(top_idx[0]),
                )
            c1, c2, c3 = st.columns(3)
            c1.image(image, caption="Original", use_container_width=True)
            c2.image(heat, caption="Grad-CAM heatmap", use_container_width=True)
            c3.image(overlay, caption="Overlay", use_container_width=True)

    with tabs[1]:
        fig_dir = Path(cfg["paths"]["figures_dir"])
        figures = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
        if not figures:
            st.info("No figures generated yet. Run scripts/generate_all_figures.py and evaluations first.")
        else:
            for path in figures:
                st.image(str(path), caption=path.name, use_container_width=True)

    with tabs[2]:
        st.warning(DISCLAIMER)
        manifest = Path(cfg["paths"]["manifest"])
        if manifest.exists():
            df = pd.read_csv(manifest)
            st.dataframe(df.groupby(["split", "label"]).size().rename("count").reset_index())


if __name__ == "__main__":
    main()
