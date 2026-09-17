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
            str(EXPERIMENT / "scripts/evaluate_paper02_development.py"),
            "--representations",
            str(OUTPUT / "development_representations"),
            "--training",
            str(OUTPUT / "development_training"),
            "--output",
            str(OUTPUT / "development_evaluation"),
            "--stability",
            str(OUTPUT / "development_robustness/odd_even_stability.json"),
            "--replicates",
            "2000",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
    )
