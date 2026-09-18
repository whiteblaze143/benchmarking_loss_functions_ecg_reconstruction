from __future__ import annotations

import torch
from torch.nn import functional as F

from scripts.paper11.inductive_bias.audit_continuous_compression import _examples as continuous_examples

from repecg.paper11_predictive_state import (
    ContinuousPredictor,
    PredictiveStateModel,
    hmm_filter_beliefs,
    hmm_next_mean_from_belief,
    hmm_next_mean_from_state,
    make_hmm_sequences,
    make_linear_state_space_sequences,
)


def test_prefix_state_does_not_change_when_only_future_changes():
    torch.manual_seed(1)
    model = PredictiveStateModel(input_dim=4, states=3)
    sequence = torch.randn(2, 8, 4)
    altered = sequence.clone()
    altered[:, 5:] += 100.0
    first, _ = model.predict_next(sequence)
    second, _ = model.predict_next(altered)
    assert torch.allclose(first[:, :5], second[:, :5])


def test_future_loss_updates_encoder_state_and_decoder():
    torch.manual_seed(2)
    model = PredictiveStateModel(input_dim=4, states=3)
    loss = model.future_loss(torch.randn(8, 6, 4))
    loss.backward()
    assert model.encoder.weight_ih_l0.grad is not None and model.encoder.weight_ih_l0.grad.norm() > 0
    assert model.state_classifier.weight.grad is not None and model.state_classifier.weight.grad.norm() > 0
    assert model.decoder.weight.grad is not None and model.decoder.weight.grad.norm() > 0


def test_prefix_prediction_has_zero_gradient_to_unseen_future_tokens():
    torch.manual_seed(3)
    model = PredictiveStateModel(input_dim=4, states=3)
    sequence = torch.randn(2, 8, 4, requires_grad=True)
    prediction, _ = model.predict_from_prefix(sequence[:, :4])
    gradient = torch.autograd.grad(prediction.square().sum(), sequence)[0]
    assert torch.count_nonzero(gradient[:, 4:]) == 0


def test_decoder_receives_only_the_categorical_state_embedding():
    torch.manual_seed(4)
    model = PredictiveStateModel(input_dim=4, states=3)
    prefix = torch.randn(2, 5, 4)
    observed = []
    handle = model.decoder.register_forward_pre_hook(lambda _module, inputs: observed.append(inputs[0].detach().clone()))
    prediction, state = model.predict_from_prefix(prefix)
    handle.remove()
    expected = state @ model.state_embedding
    assert len(observed) == 1
    assert torch.allclose(observed[0], expected)
    assert torch.allclose(prediction, model.decoder(expected))


def test_hmm_oracles_are_causal_and_true_state_is_not_worse_than_belief():
    sequence, latent = make_hmm_sequences(256, 12, seed=5, device=torch.device("cpu"), states=4)
    beliefs = hmm_filter_beliefs(sequence, states=4)
    altered = sequence.clone()
    altered[:, 7:] += 100.0
    assert torch.allclose(beliefs[:, :7], hmm_filter_beliefs(altered, states=4)[:, :7])
    target = sequence[:, 1:]
    true_state_mse = F.mse_loss(hmm_next_mean_from_state(latent, states=4), target)
    belief_mse = F.mse_loss(hmm_next_mean_from_belief(beliefs, states=4), target)
    assert true_state_mse <= belief_mse + 1e-6


def test_continuous_predictor_is_causal_and_state_space_world_has_expected_shape():
    torch.manual_seed(6)
    model = ContinuousPredictor(input_dim=12, hidden_dim=4)
    sequence, state = make_linear_state_space_sequences(16, 10, seed=6, device=torch.device("cpu"))
    altered = sequence.clone()
    altered[:, 5:] += 100.0
    assert sequence.shape == (16, 10, 12)
    assert state.shape == (16, 10, 4)
    assert torch.allclose(model.predict_from_prefix(sequence[:, :5]), model.predict_from_prefix(altered[:, :5]))


def test_continuous_audit_prefixes_and_targets_have_matching_endpoints():
    sequence = torch.arange(2 * 5, dtype=torch.float32).reshape(2, 5, 1)
    prefix, target = continuous_examples(sequence, prefix=2, shuffle=False, seed=0)
    assert torch.equal(prefix[:, -1, 0] + 1, target[:, 0])
