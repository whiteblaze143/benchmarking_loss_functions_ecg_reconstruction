from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import torch

REPO = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SRC = REPO / "experiments/ptbxl_distributional_repecg/src"
SCRIPTS = REPO / "experiments/ptbxl_distributional_repecg/scripts"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_import_all_training_scripts():
    for num in range(1, 16):
        paper_dir = SCRIPTS / f"paper{num:02d}"
        train_script = paper_dir / f"train_paper{num:02d}_shared_grid.py"
        assert train_script.exists(), f"Missing script: {train_script}"
        
        # Load module dynamically to verify syntax and imports
        spec = importlib.util.spec_from_file_location(f"train_paper{num:02d}", train_script)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Verify required attributes
        assert hasattr(mod, "VARIANTS"), f"{train_script} missing VARIANTS"
        assert hasattr(mod, "LEARNING_RATES"), f"{train_script} missing LEARNING_RATES"
        assert hasattr(mod, "_train_variant"), f"{train_script} missing _train_variant"
        assert hasattr(mod, "main"), f"{train_script} missing main"
        print(f"[VERIFIED SCRIPT] {train_script.name}: module successfully loaded and inspected")


if __name__ == "__main__":
    test_import_all_training_scripts()
