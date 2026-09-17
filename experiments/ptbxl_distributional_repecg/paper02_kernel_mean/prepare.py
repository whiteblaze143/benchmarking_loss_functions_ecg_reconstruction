from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/mithunmanivannan/.venv/bin/python")


def run(*arguments: str) -> None:
    subprocess.run([str(PYTHON), str(EXPERIMENT / "scripts/prepare_ptbxl.py"), *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    config = EXPERIMENT / "configs/common.yaml"
    output = EXPERIMENT / "outputs/paper02_kernel_mean"
    scaler = output / "development_scaler.json"
    if not scaler.exists():
        run("fit-scaler", "--config", str(config), "--split", "development_train", "--scaler", str(scaler), "--workers", str(args.workers))
    for split in ("development_train", "development_select"):
        run(
            "build-split",
            "--config",
            str(config),
            "--split",
            split,
            "--scaler",
            str(scaler),
            "--output",
            str(output),
            "--workers",
            str(args.workers),
        )


if __name__ == "__main__":
    main()
