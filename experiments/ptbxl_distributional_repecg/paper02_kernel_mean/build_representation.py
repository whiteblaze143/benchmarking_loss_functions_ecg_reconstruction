from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
OUTPUT = EXPERIMENT / "outputs/paper02_kernel_mean"


if __name__ == "__main__":
    subprocess.run(
        [
            "/home/mithunmanivannan/.venv/bin/python",
            str(EXPERIMENT / "scripts/build_paper02_representations.py"),
            "--train-cache",
            str(OUTPUT / "development_train"),
            "--select-cache",
            str(OUTPUT / "development_select"),
            "--output",
            str(OUTPUT / "development_representations"),
            "--reservoir",
            "500000",
            "--audit-pairs",
            "1000",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
    )
