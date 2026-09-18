from __future__ import annotations

import torch

from repecg.common.models import StructuralInnovationModel
from repecg.common.variants import get_variants_for_paper


def test_diagnosis_bce_routes_gradient_through_structural_predictor():
    torch.manual_seed(12)
    model = StructuralInnovationModel(input_dim=8, classes=5, variant=get_variants_for_paper(12)["full"])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.randn(16, 16, 8)), torch.rand(16, 5))
    loss.backward()
    assert model.ar_gru.weight_ih_l0.grad is not None and model.ar_gru.weight_ih_l0.grad.norm() > 0
    assert model.pred_mean.weight.grad is not None and model.pred_mean.weight.grad.norm() > 0


def test_unconditional_control_bypasses_structural_predictor():
    torch.manual_seed(13)
    model = StructuralInnovationModel(input_dim=8, classes=5, variant=get_variants_for_paper(12)["unconditional_z"])
    model(torch.randn(4, 16, 8)).sum().backward()
    assert model.ar_gru.weight_ih_l0.grad is None
    assert model.pred_mean.weight.grad is None


def test_time_reversal_is_not_implemented_as_a_paper12_variant():
    assert "time_reverse" not in get_variants_for_paper(12)
