from __future__ import annotations

from pathlib import Path

import yaml


PAPER = Path(__file__).resolve().parents[1]


def test_independent_stage_contract() -> None:
    required = {
        "config.yaml",
        "prepare.py",
        "build_representation.py",
        "train.py",
        "evaluate.py",
        "ablations.py",
        "robustness.py",
        "figures.py",
        "run_all.sh",
    }
    assert required.issubset({path.name for path in PAPER.iterdir()})


def test_paper02_controls_are_explicit() -> None:
    config = yaml.safe_load((PAPER / "config.yaml").read_text())
    assert config["representation"]["matched_controls"] == [
        "mean_full_covariance",
        "exact_moment_matched_gaussian",
        "linear_kernel_mean",
    ]
