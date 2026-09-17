from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from repecg.common.variants import get_variants_for_paper


REPO = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
EXPERIMENT = REPO / "experiments/ptbxl_distributional_repecg"
SCRIPTS = EXPERIMENT / "scripts"


def test_all_training_scripts_import() -> None:
    for paper_id in range(1, 16):
        train_script = SCRIPTS / f"paper{paper_id:02d}/train_paper{paper_id:02d}_shared_grid.py"
        assert train_script.exists()
        spec = importlib.util.spec_from_file_location(f"train_paper{paper_id:02d}", train_script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for attribute in ("LEARNING_RATES", "WEIGHT_DECAYS", "_train_variant", "main"):
            assert hasattr(module, attribute), (train_script, attribute)


def test_runner_variants_exactly_match_registry() -> None:
    for paper_id in range(1, 16):
        runner = SCRIPTS / f"paper{paper_id:02d}/run_paper{paper_id:02d}_grid.sh"
        text = runner.read_text()
        match = re.search(r'for variant in ([^;]+); do', text)
        assert match is not None, runner
        observed = re.findall(r'"([^"]+)"', match.group(1))
        assert observed == list(get_variants_for_paper(paper_id)), (paper_id, observed)
        assert 'cells/.done_${variant}' in text


def test_strict_shared_grid_tools_exist() -> None:
    assert (SCRIPTS / "aggregate_grid.py").is_file()
    assert (SCRIPTS / "evaluate_grid.py").is_file()
    assert (SCRIPTS / "run_smoke_master.sh").is_file()
