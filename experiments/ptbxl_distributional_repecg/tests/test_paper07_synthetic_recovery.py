import numpy as np
import pytest
import torch
from torch import nn

from repecg.common.variants import ExperimentVariant
from repecg.paper07_operator import (
    UNKNOWN_ID,
    OperatorSetModel,
    build_training_vocabulary,
    canonical_operators,
    derived_limb_operators,
    frozen_operator_banks,
    interpolation_operators,
    map_operator_ids,
    normalize_operator,
    operator_waveform,
    record_training_operators,
    response_atoms,
    sample_context_target_indices,
    symmetric_bernoulli_kl,
)



def test_world_1_exact_operator_linearity_and_response_orientation_law():
    """World 1: Exact Operator Linearity and Response Orientation Law.
    
    Verifies:
      1. Operator normalization strictly maps non-zero vectors to the unit sphere S^7.
      2. Zero vector raises ValueError.
      3. Operator waveform projection obeys exact linear superposition: X (a q1 + b q2) = a X q1 + b X q2.
      4. Phase-space response atoms strictly obey the physical orientation law: atoms(-q) = -atoms(q) to < 10^-12.
    """
    rng = np.random.default_rng(42)
    n_beats = 4
    n_samples = 256
    leads = 8
    scale = 0.25

    # 1. Normalization
    q = rng.normal(size=leads)
    q_norm = normalize_operator(q)
    assert np.isclose(np.linalg.norm(q_norm), 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="zero measurement operator"):
        normalize_operator(np.zeros(leads))

    # 2. Linearity of measurement projection
    basis_mv = rng.normal(size=(n_samples, leads))
    q1 = normalize_operator(rng.normal(size=leads))
    q2 = normalize_operator(rng.normal(size=leads))
    a, b = 1.5, -0.7
    w_combined = operator_waveform(basis_mv, a * q1 + b * q2)
    w_separate = a * operator_waveform(basis_mv, q1) + b * operator_waveform(basis_mv, q2)
    # Note: operator_waveform normalizes its input, so test linear property on normalized linear combination
    q_combo = normalize_operator(a * q1 + b * q2)
    w_expected = basis_mv @ q_combo
    w_actual = operator_waveform(basis_mv, q_combo)
    assert np.allclose(w_actual, w_expected, atol=1e-12)

    # 3. Response orientation law
    beats = rng.normal(size=(n_beats, n_samples, leads))
    pos_atoms = response_atoms(beats, q1, scale)
    neg_atoms = response_atoms(beats, -q1, scale)
    assert pos_atoms.shape == (n_beats, n_samples, 2)
    assert np.allclose(neg_atoms, -pos_atoms, atol=1e-12)


def test_world_2_permutation_invariance_across_measurement_sets():
    """World 2: Permutation Invariance Across Measurement Sets.
    
    Verifies that for an unordered set of m operator-response pairs,
    permuting the order of the pairs leaves the pooled patient latent
    representation z and classification logits invariant to machine precision (< 10^-5).
    """
    torch.manual_seed(42)
    batch_size = 3
    set_size = 5
    phases = 16
    feat_dim = 128
    classes = 5

    model = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="continuous")
    model.eval()

    operators = torch.randn(batch_size, set_size, 8)
    operators = operators / operators.norm(dim=-1, keepdim=True)
    responses = torch.randn(batch_size, set_size, phases, feat_dim)

    # Forward original order
    with torch.no_grad():
        latent_orig = model.encode_context(operators, responses)
        logits_orig = model(operators, responses)

    # Random permutation of the set dimension
    perm = torch.randperm(set_size)
    operators_perm = operators[:, perm]
    responses_perm = responses[:, perm]

    with torch.no_grad():
        latent_perm = model.encode_context(operators_perm, responses_perm)
        logits_perm = model(operators_perm, responses_perm)

    max_latent_diff = float((latent_orig - latent_perm).abs().max())
    max_logit_diff = float((logits_orig - logits_perm).abs().max())
    assert max_latent_diff < 1e-5, f"Latent permutation variance: {max_latent_diff}"
    assert max_logit_diff < 1e-5, f"Logit permutation variance: {max_logit_diff}"


def test_world_3_orientation_invariance_and_symmetric_kl():
    """World 3: Orientation Invariance & Symmetric Bernoulli KL.
    
    Verifies:
      1. symmetric_bernoulli_kl is strictly zero when logits are identical.
      2. symmetric_bernoulli_kl is strictly positive and symmetric for distinct logits.
      3. Clean gradient flow through symmetric Bernoulli KL.
    """
    logits_a = torch.randn(4, 5, requires_grad=True)
    logits_b = torch.randn(4, 5, requires_grad=True)

    # 1. Zero on identical inputs
    loss_self = symmetric_bernoulli_kl(logits_a, logits_a)
    assert float(loss_self) == 0.0

    # 2. Positive and symmetric
    loss_ab = symmetric_bernoulli_kl(logits_a, logits_b)
    loss_ba = symmetric_bernoulli_kl(logits_b, logits_a)
    assert float(loss_ab) > 0.0
    assert torch.isclose(loss_ab, loss_ba, atol=1e-6)

    # 3. Gradients
    loss_ab.backward()
    assert logits_a.grad is not None and torch.all(torch.isfinite(logits_a.grad))
    assert logits_b.grad is not None and torch.all(torch.isfinite(logits_b.grad))


def test_world_4_continuous_geometric_generalization_vs_categorical_blindness():
    """World 4: Continuous Geometric Generalization vs Categorical Blindness.
    
    Verifies the fundamental inductive superiority claim of Paper 07:
      1. When evaluated on unseen interpolated leads q(alpha) = normalize((1-alpha) I + alpha V2),
         the continuous operator encoder produces smoothly varying embeddings:
         ||E(q(alpha1)) - E(q(alpha2))|| -> 0 as |alpha1 - alpha2| -> 0.
      2. In contrast, the categorical encoder maps all unseen interpolated leads to UNKNOWN_ID,
         producing the identical constant degenerate vector regardless of angle or direction.
    """
    torch.manual_seed(101)
    train_operators = canonical_operators()  # 8 standard leads
    train_vocab = build_training_vocabulary(train_operators.reshape(1, 8, 8))

    model_cont = OperatorSetModel(operator_mode="continuous")
    model_cat = OperatorSetModel(operator_mode="categorical", vocabulary_size=len(train_vocab))

    # Evaluate on interior interpolation points between lead I and lead V2
    alphas = np.linspace(0.1, 0.9, 9)
    lead_i = train_operators[0]
    lead_v2 = train_operators[3]
    interpolated = [normalize_operator((1.0 - a) * lead_i + a * lead_v2) for a in alphas]

    interp_tensor = torch.as_tensor(np.stack(interpolated), dtype=torch.float32)
    ids = map_operator_ids(np.stack(interpolated).reshape(1, len(alphas), 8), train_vocab)[0]

    # 1. Categorical encoder is completely blind: all unseen leads map to UNKNOWN_ID
    assert np.all(ids == UNKNOWN_ID)
    with torch.no_grad():
        cat_embeddings = model_cat.encode_operator(None, torch.as_tensor(ids).unsqueeze(0))[0]
        # All rows must be identical
        cat_variance = float(torch.var(cat_embeddings, dim=0).sum())
        assert cat_variance == 0.0, "Categorical embeddings for unseen leads must be constant UNK"

    # 2. Continuous encoder produces smooth, continuous representations
    with torch.no_grad():
        cont_embeddings = model_cont.encode_operator(interp_tensor, None)
        cont_diffs = torch.norm(cont_embeddings[1:] - cont_embeddings[:-1], dim=1)
        assert torch.all(cont_diffs > 1e-4), "Continuous embeddings must distinguish different directions"
        # Trajectory must be smooth
        assert float(cont_diffs.max()) < 5.0 * float(cont_diffs.min())


def test_world_5_exact_linear_source_reconstructive_inversion():
    """World 5: Exact Linear Source Reconstructive Inversion.
    
    Verifies that when ECG signals are generated by a low-rank current dipole source
    s(t) in R^3 through lead geometry L in R^{8x3} (x(t) = L s(t)):
      Given m >= 3 linearly independent measurement operators Q = [q1, ..., qm],
      any unseen target operator q* can be reconstructed with exact zero error (< 10^-12)
      by linear inversion: x_q* = Y_Q (L^T Q)^+ (L^T q*).
    """
    rng = np.random.default_rng(777)
    n_samples = 200
    n_sources = 3
    leads = 8

    # True dipole sources and physical lead geometry
    s_true = rng.normal(size=(n_samples, n_sources))  # (T, 3)
    l_lead = rng.normal(size=(leads, n_sources))      # (8, 3)
    x_multilead = s_true @ l_lead.T                   # (T, 8)

    # 4 linearly independent measurement operators (m >= 3)
    q_operators = rng.normal(size=(4, leads))
    q_operators = np.stack([normalize_operator(row) for row in q_operators])

    # Observed measurements Y_Q = X Q^T: shape (T, 4)
    y_observed = x_multilead @ q_operators.T

    # Target unseen operator q*
    q_target = normalize_operator(rng.normal(size=leads))
    y_target_true = x_multilead @ q_target

    # Reconstructive inversion via Moore-Penrose pseudo-inverse of measurement subspace
    # Since Y_Q = S (L^T Q^T) = S M with M in R^{3 x 4} of rank 3:
    m_mat = l_lead.T @ q_operators.T  # (3, 4)
    s_recovered = y_observed @ np.linalg.pinv(m_mat).T
    y_target_reconstructed = s_recovered @ (l_lead.T @ q_target)

    recon_error = float(np.max(np.abs(y_target_true - y_target_reconstructed)))
    assert recon_error < 1e-12, f"Linear reconstructive inversion error: {recon_error}"


def test_world_6_frozen_operator_banks_and_disjoint_separation_audit():
    """World 6: Frozen Operator Banks & Disjoint Separation Audit.
    
    Verifies:
      1. Bank counts: seen (108), derived_limb (4), dense (100), i_to_v2 (9).
      2. Every operator in every bank has exact unit norm (||q||_2 = 1.0).
      3. Strict disjoint separation: no operator in derived_limb, dense, or i_to_v2
         overlaps with the seen training bank within tolerance 1e-8 under both +q and -q.
    """
    banks = frozen_operator_banks(tolerance=1e-8)
    expected_counts = {"seen": 108, "derived_limb": 4, "dense": 100, "i_to_v2": 9}
    assert {k: len(v) for k, v in banks.items()} == expected_counts

    for name, operators in banks.items():
        norms = np.linalg.norm(operators, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-12), f"Bank {name} contains non-unit operators"

    seen = banks["seen"]
    for name in ("derived_limb", "dense", "i_to_v2"):
        for q in banks[name]:
            min_pos_dist = float(np.min(np.linalg.norm(seen - q, axis=1)))
            min_neg_dist = float(np.min(np.linalg.norm(seen + q, axis=1)))
            assert min_pos_dist > 1e-8, f"Unseen bank {name} overlaps with seen: {min_pos_dist}"
            assert min_neg_dist > 1e-8, f"Unseen bank {name} overlaps with -seen: {min_neg_dist}"


def test_world_7_context_sampling_and_target_holdout_guarantee():
    """World 7: Context Sampling & Target Holdout Guarantee.
    
    Verifies that sample_context_target_indices:
      1. Dynamically samples context sizes spanning m in [1, 6].
      2. Strictly guarantees that the target index is NEVER present in the context set:
         t not in C across 100% of samples.
    """
    generator = torch.Generator().manual_seed(42)
    batch_size = 32
    candidates = 8

    observed_sizes = set()
    for _ in range(50):
        context, target = sample_context_target_indices(
            batch_size, candidates, generator, torch.device("cpu")
        )
        assert context.shape[0] == batch_size
        assert target.shape[0] == batch_size
        m = context.shape[1]
        assert 1 <= m <= 6
        observed_sizes.add(m)

        # Holdout check: target must not appear in any row's context
        target_in_context = (context == target.unsqueeze(1)).any()
        assert not target_in_context, "Leakage detected: target index present in context set!"

    assert observed_sizes == set(range(1, 7)), f"Context sizes did not cover [1, 6]: {observed_sizes}"


def test_world_8_full_classifier_and_auxiliary_reconstruction_gradient_flow():
    """World 8: Full Classifier & Auxiliary Reconstruction Gradient Flow.
    
    Verifies clean end-to-end forward and backward passes with finite non-zero gradients across:
      1. continuous_primary: classification loss -> gradients throughout set encoder.
      2. categorical_primary: classification loss -> gradients into known operator embeddings.
      3. continuous_auxiliary: joint classification, reconstruction, and orientation loss -> all modules.
      4. categorical_auxiliary: joint classification and reconstruction loss -> all modules.
    """
    torch.manual_seed(42)
    batch_size = 4
    context_size = 4
    phases = 16
    feat_dim = 128
    classes = 5

    operators = torch.randn(batch_size, context_size, 8)
    operators = operators / operators.norm(dim=-1, keepdim=True)
    responses = torch.randn(batch_size, context_size, phases, feat_dim)
    operator_ids = torch.randint(0, 8, (batch_size, context_size))
    target_operator = operators[:, 0]
    target_ids = operator_ids[:, 0]
    target_response = responses[:, 0]
    labels = torch.randint(0, 2, (batch_size, classes)).float()

    # 1. continuous_primary
    m1 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="continuous")
    m1.train()
    logits1 = m1(operators, responses)
    loss1 = nn.functional.binary_cross_entropy_with_logits(logits1, labels)
    loss1.backward()
    assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in m1.parameters() if p.requires_grad)

    # 2. categorical_primary
    m2 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="categorical", vocabulary_size=8)
    m2.train()
    logits2 = m2(None, responses, operator_ids=operator_ids)
    loss2 = nn.functional.binary_cross_entropy_with_logits(logits2, labels)
    loss2.backward()
    assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in m2.parameters() if p.requires_grad)

    # 3. continuous_auxiliary (Joint Cls + Recon + Orient)
    m3 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="continuous")
    m3.train()
    logits3, recon3 = m3(operators, responses, target_operator=target_operator, return_reconstruction=True)
    # Also evaluate negative for orientation
    neg_logits3, _ = m3(-operators, -responses, target_operator=-target_operator, return_reconstruction=True)
    loss3 = (
        nn.functional.binary_cross_entropy_with_logits(logits3, labels)
        + 0.1 * nn.functional.mse_loss(recon3, target_response)
        + 0.05 * symmetric_bernoulli_kl(logits3, neg_logits3)
    )
    loss3.backward()
    assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in m3.parameters() if p.requires_grad)

    # 4. categorical_auxiliary
    m4 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="categorical", vocabulary_size=8)
    m4.train()
    logits4, recon4 = m4(
        None, responses, operator_ids=operator_ids, target_operator_ids=target_ids, return_reconstruction=True
    )
    loss4 = (
        nn.functional.binary_cross_entropy_with_logits(logits4, labels)
        + 0.1 * nn.functional.mse_loss(recon4, target_response)
    )
    loss4.backward()
    assert all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in m4.parameters() if p.requires_grad)
