from pathlib import Path

import pandas as pd

from repecg.evaluation.tasks import load_dataset_tasks


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_native_dataset_task_registry_is_explicit() -> None:
    tasks = load_dataset_tasks(EXPERIMENT / "configs/dataset_tasks.json")
    assert set(tasks) == {
        "ptbxl", "echonext", "kingston_icu", "ludb", "rdb", "isp",
        "zhejiang", "emory_muse", "sunnybrook",
    }
    assert tasks["echonext"]["tasks"][0]["labels"][-1] == "shd_moderate_or_greater"
    assert tasks["kingston_icu"]["tasks"][0]["labels"] == ["SINUS", "AFIB_AFLT"]
    assert tasks["kingston_icu"]["evaluation_role"] == "zero_padding_negative_control"
    assert {task["task_id"] for task in tasks["isp"]["tasks"]} == {
        "isp_wave_delineation", "isp_sex", "isp_age",
    }
    assert {task["task_id"] for task in tasks["zhejiang"]["tasks"]} == {
        "zhejiang_otva_origin", "zhejiang_arrhythmia_type", "zhejiang_wave_delineation",
    }
    assert tasks["emory_muse"]["tasks"][0]["task_id"] == "emory_12sl_diagnoses"
    assert len(tasks["ludb"]["tasks"]) == 3


def test_ludb_declared_rhythm_source_exists() -> None:
    tasks = load_dataset_tasks(EXPERIMENT / "configs/dataset_tasks.json")
    task = next(task for task in tasks["ludb"]["tasks"] if task["task_id"] == "ludb_rhythm")
    source_file, source_column = task["label_source"].split(":", maxsplit=1)
    table = pd.read_csv(EXPERIMENT.parents[1] / "data" / "ludb" / source_file)
    assert source_column in table.columns
    assert table[source_column].notna().all()
