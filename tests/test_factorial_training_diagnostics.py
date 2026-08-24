import pandas as pd
import pytest

from scripts.build_factorial_training_diagnostics import (
    build_controlled_kernel_contrasts,
    decode_factorial_mask,
)


def test_decode_factorial_mask_maps_every_position():
    assert decode_factorial_mask("1101014") == {
        "mse_active": 1,
        "correlation_active": 1,
        "derivative_active": 0,
        "vcg_active": 1,
        "energy_distance_active": 0,
        "lead_consistency_active": 1,
        "mmd_kernel": 4,
    }


@pytest.mark.parametrize(
    "mask",
    ["0000000", "1000005", "10000x0", "100000", "10000000"],
)
def test_decode_factorial_mask_rejects_out_of_contract_identity(mask):
    with pytest.raises(ValueError, match="Invalid factorial mask"):
        decode_factorial_mask(mask)


def test_controlled_kernel_contrasts_hold_seed_and_binary_prefix_fixed():
    frame = pd.DataFrame([
        {
            "model_id": "f_1000000_s42", "factorial_mask": "1000000",
            "seed": 42, "mmd_kernel": 0, "first_val_mse": 0.10,
            "last_val_mse": 0.04, "duration_seconds": 1200,
            "checkpoint_sha256": "a" * 64,
        },
        {
            "model_id": "f_1000002_s42", "factorial_mask": "1000002",
            "seed": 42, "mmd_kernel": 2, "first_val_mse": 0.12,
            "last_val_mse": 0.03, "duration_seconds": 1500,
            "checkpoint_sha256": "b" * 64,
        },
        {
            "model_id": "f_1000012_s42", "factorial_mask": "1000012",
            "seed": 42, "mmd_kernel": 2, "first_val_mse": 0.08,
            "last_val_mse": 0.02, "duration_seconds": 900,
            "checkpoint_sha256": "c" * 64,
        },
    ])
    result = build_controlled_kernel_contrasts(frame)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.baseline_model_id == "f_1000000_s42"
    assert row.candidate_model_id == "f_1000002_s42"
    assert row.candidate_kernel == 2
    assert row.delta_last_val_mse == pytest.approx(-0.01)
    assert row.delta_duration_minutes == pytest.approx(5.0)


def test_controlled_kernel_contrasts_reject_duplicate_kernel_identity():
    frame = pd.DataFrame([
        {
            "model_id": model_id, "factorial_mask": mask, "seed": 42,
            "mmd_kernel": kernel, "first_val_mse": 0.1,
            "last_val_mse": 0.05, "duration_seconds": 60,
            "checkpoint_sha256": model_id,
        }
        for model_id, mask, kernel in (
            ("a", "1000000", 0), ("b", "1000001", 1), ("c", "1000001", 1)
        )
    ])
    with pytest.raises(RuntimeError, match="Duplicate kernel level"):
        build_controlled_kernel_contrasts(frame)
