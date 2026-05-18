from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all configured models.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    models = args.models or cfg["models"]["supported"]
    for model in models:
        cmd = [sys.executable, "train.py", "--config", args.config, "--model", model]
        print("Running:", " ".join(cmd))
        code = subprocess.call(cmd)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
