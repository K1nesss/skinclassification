from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Streamlit demo.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", default="8501")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py",
        "--server.port",
        str(args.port),
        "--",
        "--config",
        args.config,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

