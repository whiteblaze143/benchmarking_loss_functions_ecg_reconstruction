from __future__ import annotations

import torch

from repecg.common.paper_models import create_paper_model
from repecg.common.variants import get_variants_for_paper


def test_every_registered_variant_forward_and_backward() -> None:
    torch.manual_seed(42)
    labels = torch.randint(0, 2, (2, 5)).float()
    for paper_id in range(1, 16):
        for name, variant in get_variants_for_paper(paper_id).items():
            if paper_id == 5:
                features = torch.randn(2, 128, requires_grad=True)
            elif paper_id == 6:
                features = torch.randn(2, 2, 16, 16, requires_grad=True)
            else:
                features = torch.randn(
                    2, 16, 16 if paper_id == 1 else 128, requires_grad=True
                )
            model = create_paper_model(paper_id, features.shape[-1], 5, variant)
            if paper_id == 8 and variant.mechanism == "kmeans_dictionary":
                model.set_codebook(torch.randn(64, features.shape[-1]))
            if paper_id == 7:
                responses = torch.randn(2, 4, 16, 128, requires_grad=True)
                if "categorical" in variant.mechanism:
                    output = model(None, responses, operator_ids=torch.randint(0, 8, (2, 4)))
                else:
                    output = model(torch.randn(2, 4, 8), responses)
            elif paper_id == 9:
                output = model(
                    torch.randn(2, 4, 8),
                    torch.randn(2, 4, 16, 128, requires_grad=True),
                )
            else:
                output = model(features)
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
    assert model(torch.randn(4, 128)).shape == (4, 5)


def test_paper08_kmeans_control_quantizes_with_persistent_codebook() -> None:
    variant = get_variants_for_paper(8)["kmeans_tokens"]
    model = create_paper_model(8, 128, 5, variant)
    with torch.no_grad():
        model.set_codebook(torch.randn(64, 128))
    state = model.state_dict()
    assert bool(state["codebook_ready"])
    assert state["codebook"].shape == (64, 128)
    assert model(torch.randn(2, 16, 128)).shape == (2, 5)
