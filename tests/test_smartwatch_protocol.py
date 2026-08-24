from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.evaluate_comprehensive_registry import ReconstructionAdapter
from scripts.smartwatch_protocol import (
    align_watch_to_reference,
    canonical_record_key,
    estimate_heart_rate_bpm,
    parse_protocol_target,
    validate_single_watch_lead,
)


class _ConstantUNet(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (inputs.shape[0], 12, inputs.shape[-1]),
            7.0,
            device=inputs.device,
        )


def test_parse_all_calibrated_protocol_types():
    assert parse_protocol_target(Path("amp_test/amp1500/amp1500_2")) == {
        "experiment_type": "r_wave_amplitude",
        "target_value": 1500.0,
        "target_unit": "uV",
    }
    assert parse_protocol_target(Path("freq_test/f220/f220_4"))["target_value"] == 220
    assert parse_protocol_target(Path("st-segment/st-m8/st-m8_0"))["target_value"] == -800
    assert parse_protocol_target(Path("st-segment/ST-m8/ST-m8_0"))["target_value"] == -800
    assert parse_protocol_target(Path("st-segment/st-p4/st-p4_1"))["target_value"] == 400
    assert parse_protocol_target(Path("sqr-2hz/sqr-2hz_3"))["target_unit"] == "Hz"
    assert (
        canonical_record_key(Path("st-segment/ST-m8/ST-m8_0"))
        == "st-segment/st-m8/st-m8_0"
    )


def test_watch_lead_is_header_driven():
    assert validate_single_watch_lead(["II"]) == ("II", 1)


def test_fitbit_uppercase_st_record_pairs_with_lowercase_philips_key():
    fitbit = Path("st-segment/ST-m6/ST-m6_4")
    philips = Path("st-segment/st-m6/st-m6_4")
    assert canonical_record_key(fitbit) == canonical_record_key(philips)


def test_alignment_and_heart_rate_on_shifted_periodic_fixture():
    fs = 500
    period = fs
    base = np.zeros(6500, dtype=np.float32)
    base[np.arange(250, 6250, period)] = 1.0
    reference = np.repeat(base[:5500, None], 12, axis=1)
    watch = np.concatenate([np.zeros(700, dtype=np.float32), base])
    watch_window, reference_window, audit = align_watch_to_reference(
        watch,
        reference,
        reference_lead_index=1,
        fs=fs,
        target_len=5000,
    )
    assert watch_window.shape == (5000,)
    assert reference_window.shape == (5000, 12)
    assert audit["alignment_pearson"] > 0.99
    assert abs(estimate_heart_rate_bpm(watch_window, fs) - 60.0) < 0.1


def test_adapter_can_preserve_only_the_physically_acquired_watch_lead():
    adapter = ReconstructionAdapter(
        spec={"kind": "unet", "observed_leads": [0, 1, 7]},
        model=_ConstantUNet(),
        device=torch.device("cpu"),
    )
    inputs = torch.zeros(2, 12, 5000)
    inputs[:, 0] = 1.0
    inputs[:, 1] = 2.0
    inputs[:, 7] = 3.0

    default = adapter.reconstruct(inputs)
    assert torch.equal(default[:, [0, 1, 7]], inputs[:, [0, 1, 7]])

    watch = adapter.reconstruct(inputs, preserve_observed=[1])
    assert torch.equal(watch[:, 1], inputs[:, 1])
    assert torch.all(watch[:, 0] == 7.0)
    assert torch.all(watch[:, 7] == 7.0)

    with pytest.raises(ValueError, match="model input contract"):
        adapter.reconstruct(inputs, preserve_observed=[2])
