from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from repecg.evaluation.results_db import coverage, ingest_paper, register_expected_evaluations


def _metrics(value: float) -> dict[str, object]:
    return {"macro_auroc": value, "classwise_auroc": [value] * 5}


def test_ingest_is_complete_idempotent_and_conflict_safe(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    (output / "cells").mkdir(parents=True)
    (output / "development_evaluation").mkdir()
    variants = ["full"]
    (output / "manifest.json").write_text(json.dumps({"seed": 42, "variants": variants}))
    for index in range(9):
        cell = output / "cells" / f"cell{index}"
        cell.mkdir()
        (cell / "checkpoint.pt").write_bytes(b"checkpoint")
        (cell / "summary.json").write_text(json.dumps({
            "variant": "full", "learning_rate": 10 ** (-index - 1),
            "weight_decay": index / 100, "metrics": _metrics(index / 10),
        }))
    (output / "full_best.pt").write_bytes(b"best")
    evaluation = output / "development_evaluation" / "metrics_full.json"
    evaluation.write_text(json.dumps({"variant": "full", "metrics": _metrics(0.5)}))
    database = tmp_path / "results.sqlite3"

    assert ingest_paper(database, 1, output, "ptbxl", "fold8", "smoke")["runs"] == 10
    assert ingest_paper(database, 1, output, "ptbxl", "fold8", "smoke")["runs"] == 10
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 10
        assert connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0] == 60

    payload = json.loads(evaluation.read_text())
    payload["metrics"]["macro_auroc"] = 0.6
    evaluation.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="conflicting payload"):
        ingest_paper(database, 1, output, "ptbxl", "fold8", "smoke")


def test_ingest_rejects_incomplete_grid(tmp_path: Path) -> None:
    output = tmp_path / "paper"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"seed": 42, "variants": ["full"]}))
    (output / "cells").mkdir()
    (output / "development_evaluation").mkdir()
    with pytest.raises(RuntimeError, match="expected 9 cell summaries"):
        ingest_paper(tmp_path / "results.sqlite3", 1, output, "ptbxl", "fold8", "smoke")


def test_expected_evaluation_coverage(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite3"
    assert register_expected_evaluations(
        database,
        run_type="production",
        paper_id=1,
        datasets=["ptbxl", "ludb"],
        split="external_test",
        variants=["full", "linear_probe"],
        seed=42,
    ) == 4
    assert coverage(database, "production") == {"pending": 4}
