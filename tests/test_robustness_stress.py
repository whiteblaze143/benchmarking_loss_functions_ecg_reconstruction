from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from scripts.robustness_stress import (
    apply_condition,
    build_conditions,
    load_fitbit_noise,
)


def test_core_and_optional_fitbit_condition_coverage() -> None:
    core = build_conditions()
    extended = build_conditions(include_fitbit=True)
    assert len(core) == 17
    assert len({condition.id for condition in core}) == 17
    assert len(extended) == 20
    assert {condition.id for condition in extended} - {condition.id for condition in core} == {
        "fitbit_20db",
        "fitbit_10db",
        "fitbit_baseline_wander",
    }


def test_noise_pairing_is_deterministic_by_condition_record_and_lead() -> None:
    clean = torch.linspace(-1, 1, 2 * 12 * 500).reshape(2, 12, 500)
    ids = torch.tensor([101, 202])
    condition = build_conditions()[0]
    first = apply_condition(clean, [0, 1, 7], ids, condition, {})
    second = apply_condition(clean, [0, 1, 7], ids, condition, {})
    assert torch.equal(first, second)
    assert torch.equal(first[:, 2], clean[:, 2])
    assert not torch.equal(first[:, 0], clean[:, 0])


def test_fitbit_loader_requires_and_verifies_provenance(tmp_path) -> None:
    arrays = {
        "fitbit_noise": np.arange(1000, dtype=np.float32)[None, :],
        "fitbit_baseline_wander": np.sin(np.linspace(0, 5, 1000))[None, :].astype(np.float32),
    }
    artifacts = {}
    for key, values in arrays.items():
        path = tmp_path / f"{key}.npy"
        np.save(path, values)
        artifacts[key] = {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    provenance = {
        "source_dataset": "fixture",
        "source_version": "1",
        "source_records": ["record-1"],
        "extraction_method": "test fixture",
        "sample_rate_hz": 500,
        "units": "arbitrary residual",
        "artifacts": artifacts,
    }
    (tmp_path / "PROVENANCE.json").write_text(json.dumps(provenance))
    loaded, verified = load_fitbit_noise(tmp_path)
    assert set(loaded) == set(arrays)
    assert verified == provenance
    assert all(value.dtype == np.float32 for value in loaded.values())
