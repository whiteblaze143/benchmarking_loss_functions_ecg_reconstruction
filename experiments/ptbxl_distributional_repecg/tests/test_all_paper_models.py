from __future__ import annotations

import torch

from repecg.common.paper_models import create_paper_model
from repecg.common.variants import get_variants_for_paper


def test_every_registered_variant_forward_and_backward() -> None:
    torch.manual_seed(42)
    labels = torch.randint(0, 2, (2, 5)).float()
    for paper_id in range(1, 16):
        for name, variant in get_variants_for_paper(paper_id).items():
            features = torch.randn(2, 16, 16 if paper_id == 1 else 128, requires_grad=True)
            model = create_paper_model(paper_id, features.shape[-1], 5, variant)
            output = model(features, return_reconstruction=False) if paper_id == 7 else model(features)
            assert output.shape == labels.shape, (paper_id, name, output.shape)
            torch.nn.functional.binary_cross_entropy_with_logits(output, labels).backward()


def test_paper04_rank_deficient_input_is_finite() -> None:
    variant = get_variants_for_paper(4)["full"]
    model = create_paper_model(4, 128, 5, variant)
    output = model(torch.ones(8, 16, 128))
    assert torch.isfinite(output).all()


def test_paper05_occupancy_control_removes_dynamics_loss() -> None:
    variant = get_variants_for_paper(5)["occupancy_only"]
    model = create_paper_model(5, 128, 5, variant)
    loss = model.forward_koopman_loss(torch.randn(4, 16, 128))
    assert float(loss) == 0.0
