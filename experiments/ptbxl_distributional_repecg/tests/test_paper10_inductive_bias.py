from __future__ import annotations

import torch
from torch.nn import functional as F

from repecg.common.models import InterventionalRepStatModel
from repecg.paper10_interventional import coral_loss, irmv1_penalty, make_world, paired_matching_loss


def test_paired_loss_rejects_norm_only_environment_leakage():
    clean = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    transformed = 3.0 * clean
    cosine_only = (F.normalize(clean, dim=-1) - F.normalize(transformed, dim=-1)).square().sum()
    assert cosine_only < 1e-12
    assert paired_matching_loss(clean, transformed).item() > 1.0


def test_diagnosis_head_receives_normalized_shared_state():
    torch.manual_seed(5)
    model = InterventionalRepStatModel(input_dim=8, width=32, classes=2, environments=3)
    x = torch.randn(4, 16, 8)
    state, _ = model.factorize(x)
    assert torch.allclose(model(x), model.diag_head(F.normalize(state, dim=-1)))


def test_paired_matching_distinguishes_same_record_from_mismatch():
    torch.manual_seed(7)
    clean = torch.randn(32, 8)
    transformed = clean + 0.03 * torch.randn_like(clean)
    same = paired_matching_loss(clean, transformed)
    mismatch = paired_matching_loss(clean, transformed.roll(1, dims=0))
    assert same < mismatch


def test_paper10_objectives_have_distinct_nonzero_backbone_gradients():
    torch.manual_seed(11)
    model = InterventionalRepStatModel(input_dim=8, width=32, classes=2, environments=3)
    x = torch.randn(12, 16, 8)
    x_pair = x + 0.1 * torch.randn_like(x)
    labels = torch.randint(0, 2, (12, 2), dtype=torch.float32)
    environments = torch.arange(12) % 3
    zs, za = model.factorize(x)
    pair_zs, _ = model.factorize(x_pair)
    objectives = {
        "pair": paired_matching_loss(zs, pair_zs),
        "coral": coral_loss(zs, environments),
        "irmv1": irmv1_penalty(model(x), labels, environments),
        "acquisition": F.cross_entropy(model.acq_head(za), environments),
    }
    gradients = {}
    for name, loss in objectives.items():
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        gradient = model.backbone[0].weight.grad
        assert gradient is not None and gradient.norm().item() > 0.0, name
        gradients[name] = gradient.detach().clone()
    assert not torch.allclose(gradients["pair"], gradients["coral"])
    assert not torch.allclose(gradients["pair"], gradients["irmv1"])


def test_destructive_world_cannot_preserve_diagnostic_information():
    world = make_world("D", "test", 512, seed=13)
    assert torch.unique(world["transformed"][:, :, :4]).numel() == 1
    assert 0.4 < world["labels"].float().mean().item() < 0.6


def test_four_synthetic_worlds_encode_only_the_predeclared_claims():
    independent = make_world("A", "train", 10_000, seed=17)
    shortcut_train = make_world("B", "train", 10_000, seed=17)
    shortcut_test = make_world("B", "test", 10_000, seed=17)
    causal = make_world("C", "test", 10_000, seed=17)
    assert abs((independent["environment"] == independent["labels"]).float().mean().item() - 0.5) < 0.02
    assert (shortcut_train["environment"] == shortcut_train["labels"]).float().mean().item() == 1.0
    assert abs((shortcut_test["environment"] == shortcut_test["labels"]).float().mean().item() - 0.5) < 0.02
    assert torch.equal(causal["environment"], causal["labels"])
