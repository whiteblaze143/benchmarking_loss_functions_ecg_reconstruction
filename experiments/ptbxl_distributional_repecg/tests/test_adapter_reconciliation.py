import json
from pathlib import Path


EXPERIMENT = Path(__file__).resolve().parents[1]


def test_adapter_reconciliation_has_exact_external_roster() -> None:
    payload = json.loads((EXPERIMENT / "outputs/adapter_reconciliation.json").read_text())
    assert set(payload["adapters"]) == {
        "echonext", "kingston_icu", "ludb", "rdb", "isp",
        "zhejiang", "emory_muse", "sunnybrook",
    }
    assert payload["status"] == "blocked"
    assert payload["blocked_datasets"] == ["echonext", "zhejiang"]


def test_adapter_reconciliation_counts_and_maps_are_explicit() -> None:
    adapters = json.loads((EXPERIMENT / "outputs/adapter_reconciliation.json").read_text())["adapters"]
    assert adapters["echonext"]["locally_available_records"] == 5_442
    assert adapters["echonext"]["locally_unavailable_records"] == 94_558
    assert adapters["kingston_icu"]["eligible_records"] == 581
    assert adapters["kingston_icu"]["excluded_records"] == 15
    assert adapters["kingston_icu"]["independent_basis_mask"] == [1, 1, 0, 0, 0, 0, 0, 0]
    assert adapters["emory_muse"]["eligible_records"] == 968_680
    assert adapters["emory_muse"]["excluded_records"] == 30_164
    assert adapters["zhejiang"]["source_sampling_hz"] == 2_000
    assert adapters["zhejiang"]["model_sampling_hz"] == 500
    assert adapters["zhejiang"]["units"].startswith("UNVERIFIED")
