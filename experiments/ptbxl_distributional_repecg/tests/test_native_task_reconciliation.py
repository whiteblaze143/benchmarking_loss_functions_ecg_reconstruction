import json
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_native_task_reconciliation_covers_every_registered_task() -> None:
    registry = json.loads((EXPERIMENT / "configs/dataset_tasks.json").read_text())
    expected = {
        task["task_id"] for specification in registry.values() for task in specification["tasks"]
    }
    audit = json.loads((EXPERIMENT / "outputs/native_task_reconciliation.json").read_text())
    assert audit["status"] == "complete"
    assert set(audit["task_ids"]) == expected
    assert set(audit["datasets"]) == set(registry)


def test_native_task_reconciliation_preserves_key_denominators() -> None:
    audit = json.loads((EXPERIMENT / "outputs/native_task_reconciliation.json").read_text())["datasets"]
    assert audit["echonext"]["splits"]["test"]["records"] == 5_442
    assert audit["kingston_icu"]["splits"]["Test"]["label_counts"] == {
        "AFIB_AFLT": 49,
        "SINUS": 249,
    }
    assert audit["emory_muse"]["eligible_records"] == 968_680
    assert audit["emory_muse"]["ineligible_records"] == 5_492
    assert audit["rdb"]["records"] == 2_398
    assert audit["zhejiang"]["records"] == 334
    assert audit["zhejiang"]["origin_counts"] == {"LVOT": 77, "RVOT": 257}
    assert audit["zhejiang"]["arrhythmia_type_counts"] == {"PVC": 329, "VT": 5}
    assert audit["isp"]["records"] == 475
