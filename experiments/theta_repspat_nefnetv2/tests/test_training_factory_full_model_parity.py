from pathlib import Path

import pytest
import torch

from theta_repspat.vendor import instantiate_author_model


ROOT = Path(__file__).resolve().parents[1]
AUTHOR_ROOT = ROOT / "author_code" / "nefnet_v2"


@pytest.mark.parametrize("lead_num", [1, 3])
def test_training_factory_full_model_copy_parity(lead_num):
    torch.manual_seed(7100 + lead_num)
    reference = instantiate_author_model(AUTHOR_ROOT, torch.device("cpu"), lead_num=lead_num)
    executed = instantiate_author_model(AUTHOR_ROOT, torch.device("cpu"), lead_num=lead_num)
    executed.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    executed.eval()

    batch = 1
    samples = 4608
    inputs = torch.randn(batch, lead_num, samples)
    input_angles = torch.randn(batch, lead_num, 2)
    query_angle = torch.randn(batch, 1, 2) if lead_num == 1 else torch.randn(batch, 2)
    with torch.no_grad():
        expected = reference(inputs, input_angles, query_angle)
        actual = executed(inputs, input_angles, query_angle)
    assert actual.shape == (batch, 1, samples)
    assert torch.max(torch.abs(expected - actual)).item() <= 1e-6
    expected_class = "layer1" if lead_num == 1 else "layer"
    assert type(executed).__name__ == expected_class
