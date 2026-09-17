from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    subprocess.run(["bash", str(EXPERIMENT / "scripts/run_paper02_grid.sh")], cwd=ROOT, check=True)
