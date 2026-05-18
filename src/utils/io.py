from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_project_dirs(cfg: dict[str, Any]) -> None:
    for key, value in cfg["paths"].items():
        path = Path(value)
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["interim_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["processed_image_dir"]).mkdir(parents=True, exist_ok=True)


def project_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])

