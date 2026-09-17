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
            str(EXPERIMENT / "scripts/evaluate_paper02_stability.py"),
            "--cache",
            str(OUTPUT / "development_select"),
            "--kernel-fit",
            str(OUTPUT / "development_representations/kernel_fit.npz"),
            "--output",
            str(OUTPUT / "development_robustness"),
            "--replicates",
            "2000",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
    )
