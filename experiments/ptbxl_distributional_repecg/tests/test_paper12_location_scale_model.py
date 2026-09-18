from __future__ import annotations

import torch
import numpy as np

from repecg.paper12_innovations import LocationScaleInnovationModel, PhaseCoordinateStandardizer
from scripts.paper12.run_g3_residual_audit import wrong_history_indices


def test_conditional_nll_updates_only_the_density_path():
    torch.manual_seed(120)
    model = LocationScaleInnovationModel(input_dim=8)
    features = torch.randn(12, 16, 8)
    model.density_loss(features).backward()
    assert model.predictor.weight_ih_l0.grad is not None and model.predictor.weight_ih_l0.grad.norm() > 0
    assert model.mean.weight.grad is not None and model.mean.weight.grad.norm() > 0
    assert model.log_scale.weight.grad is not None and model.log_scale.weight.grad.norm() > 0
    assert model.head[0].weight.grad is None


def test_stagewise_diagnosis_updates_only_head_after_density_freeze():
    torch.manual_seed(121)
    model = LocationScaleInnovationModel(input_dim=8)
    model.freeze_density()
    model.diagnosis_loss(torch.randn(12, 16, 8), torch.rand(12, 5)).backward()
    assert model.predictor.weight_ih_l0.grad is None
    assert model.mean.weight.grad is None
    assert model.log_scale.weight.grad is None
    assert model.head[0].weight.grad is not None and model.head[0].weight.grad.norm() > 0


def test_density_and_probe_stages_produce_distinct_parameter_updates():
    torch.manual_seed(124)
    density_model = LocationScaleInnovationModel(input_dim=8)
    probe_model = LocationScaleInnovationModel(input_dim=8)
    probe_model.load_state_dict(density_model.state_dict())
    features = torch.randn(12, 16, 8)
    labels = torch.rand(12, 5)

    density_optimizer = torch.optim.SGD(density_model.parameters(), lr=0.01)
    density_optimizer.zero_grad()
    density_model.density_loss(features).backward()
    density_optimizer.step()

    probe_model.freeze_density()
    probe_optimizer = torch.optim.SGD(filter(lambda parameter: parameter.requires_grad, probe_model.parameters()), lr=0.01)
    probe_optimizer.zero_grad()
    probe_model.diagnosis_loss(features, labels).backward()
    probe_optimizer.step()

    assert not torch.allclose(density_model.predictor.weight_ih_l0, probe_model.predictor.weight_ih_l0)
    assert not torch.allclose(density_model.head[0].weight, probe_model.head[0].weight)


def test_early_innovations_have_zero_gradient_to_future_phase_features():
    torch.manual_seed(122)
    model = LocationScaleInnovationModel(input_dim=8)
    features = torch.randn(2, 16, 8, requires_grad=True)
    innovation = model.components(features)[2]
    gradient = torch.autograd.grad(innovation[:, :5].square().sum(), features)[0]
    assert torch.count_nonzero(gradient[:, 5:]) == 0


def test_wrong_record_history_changes_innovations_without_changing_targets():
    torch.manual_seed(123)
    model = LocationScaleInnovationModel(input_dim=8)
    features = torch.randn(4, 16, 8)
    correct = model.components(features)[2]
    wrong = model.components(features, history_permutation=torch.tensor([1, 0, 3, 2]))[2]
    assert not torch.allclose(correct[:, 1:], wrong[:, 1:])
    assert torch.allclose(correct[:, 0], wrong[:, 0])


def test_diagnosis_head_excludes_unfitted_phase_zero():
    model = LocationScaleInnovationModel(input_dim=8)
    assert model.head[0].in_features == 15 * 8


def test_raw_scale_audit_exposes_values_before_clamping():
    model = LocationScaleInnovationModel(input_dim=8)
    features = torch.randn(3, 16, 8)
    raw = model.raw_log_scale(features)
    clamped = model.components(features)[1]
    assert torch.equal(clamped, raw.clamp(-4.0, 2.0))


def test_v2_standardizer_is_phase_coordinate_specific_and_train_only():
    torch.manual_seed(125)
    train = torch.randn(40, 16, 8) * torch.logspace(-3, 3, 8) + torch.arange(16)[None, :, None]
    standardizer = PhaseCoordinateStandardizer.fit(train)
    transformed = standardizer.transform(train)
    assert torch.allclose(transformed.mean(dim=0), torch.zeros(16, 8), atol=2e-5)
    assert torch.allclose(transformed.std(dim=0, unbiased=False), torch.ones(16, 8), atol=2e-5)
    held_out = train + 10
    _ = standardizer.transform(held_out)
    assert torch.equal(standardizer.mean, train.double().mean(dim=0, keepdim=True))


def test_v2_scale_head_initializes_at_unit_scale():
    model = LocationScaleInnovationModel(input_dim=8, unit_scale_init=True)
    raw = model.raw_log_scale(torch.randn(3, 16, 8))
    assert torch.count_nonzero(raw) == 0


def test_wrong_history_assignment_is_bijective_and_patient_disjoint():
    patient_ids = np.array([1, 1, 2, 3, 4, 5])
    labels = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.float32)
    for matched in (False, True):
        donor = wrong_history_indices(patient_ids, labels, label_matched=matched)
        assert len(np.unique(donor)) == len(donor)
        assert np.all(patient_ids[donor] != patient_ids)
