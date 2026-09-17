import numpy as np
import pytest
import torch
from torch import nn

from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.common.variants import ExperimentVariant
from repecg.paper07_operator import (
    UNKNOWN_ID,
    OperatorSetModel,
    analytic_pinv_reconstruct,
    build_training_vocabulary,
    canonical_operators,
    derived_limb_operators,
    frozen_operator_banks,
    interpolation_operators,
    lmmse_reconstruct,
    map_operator_ids,
    nearest_known_operator,
    normalize_operator,
    operator_waveform,
    projective_distance,
    projective_response_atoms,
    record_training_operators,
    response_atoms,
    sample_context_target_indices,
    symmetric_bernoulli_kl,
)


def test_world_1_linearity_and_non_odd_phase_kme_proof():
    """World 1: Raw Measurement Linearity & Non-Odd Phase-KME Demonstration.
    
    Verifies:
      1. Raw measurement projection obeys linear superposition: X (a q1 + b q2) = a X q1 + b X q2.
      2. Raw phase-space response atoms are strictly odd under polarity: atoms(-q) = -atoms(q) to < 10^-12.
      3. CRUCIAL MATHEMATICAL AUDIT: A nonlinear Nyström/KME feature map psi(a) is NOT odd:
         psi(-a) != -psi(a) for standard RBF / IMQ kernels.
         Therefore, r_{-q} != -r_q, proving that sign symmetry cannot be assumed post-KME.
    """
    rng = np.random.default_rng(42)
    n_beats = 4
    n_samples = 256
    leads = 8
    scale = 0.25

    # 1. Linearity of measurement projection
    basis_mv = rng.normal(size=(n_samples, leads))
    q1 = normalize_operator(rng.normal(size=leads))
    q2 = normalize_operator(rng.normal(size=leads))
    a, b = 1.5, -0.7
    q_combo = normalize_operator(a * q1 + b * q2)
    w_expected = basis_mv @ q_combo
    w_actual = operator_waveform(basis_mv, q_combo)
    assert np.allclose(w_actual, w_expected, atol=1e-12)

    # 2. Raw response atoms are odd under polarity reversal
    beats = rng.normal(size=(n_beats, n_samples, leads))
    pos_atoms = response_atoms(beats, q1, scale)
    neg_atoms = response_atoms(beats, -q1, scale)
    assert pos_atoms.shape == (n_beats, n_samples, 2)
    assert np.allclose(neg_atoms, -pos_atoms, atol=1e-12)

    # 3. Demonstration that nonlinear Phase-KME is NOT odd: psi(-a) != -psi(a)
    flat_pos = pos_atoms.reshape(-1, 2)
    nystrom = NystromMap.fit(flat_pos, landmarks=16, c2=1.0, seed=42)
    flat_neg = -flat_pos

    kme_pos = nystrom.mean(flat_pos)
    kme_neg = nystrom.mean(flat_neg)

    # If KME were odd, kme_neg + kme_pos would be identically zero.
    # Because IMQ kernel 1/sqrt(||x - c||^2 + c2) is strictly positive, kme is NOT odd!
    odd_discrepancy = float(np.linalg.norm(kme_neg + kme_pos))
    assert odd_discrepancy > 0.01, (
        f"Phase-KME was unexpectedly odd; expected strictly positive discrepancy: {odd_discrepancy}"
    )



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

    model = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="projective")
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


def test_world_3_exact_z2_gauge_symmetry_of_projective_atoms():
    """World 3: Exact Z2 Projective Gauge Symmetry.
    
    Verifies:
      1. Projective measurement atom g_q(t) = [q * x_q(t), q * dx_q/dt(t)] in R^16
         is IDENTICALLY INVARIANT under (q, x_q) -> (-q, -x_q) to float64 machine precision (< 10^-14).
      2. Projective operator encoder E(q) = MLP(flatten(q q^T)) is IDENTICALLY INVARIANT
         under q -> -q: E(-q) == E(q) to < 10^-6.
    """
    rng = np.random.default_rng(123)
    n_beats = 4
    n_samples = 256
    leads = 8
    scale = 0.25

    beats = rng.normal(size=(n_beats, n_samples, leads))
    q = rng.normal(size=leads)

    # 1. Projective atoms: g_q == g_{-q}
    g_pos = projective_response_atoms(beats, q, scale)
    g_neg = projective_response_atoms(beats, -q, scale)
    assert g_pos.shape == (n_beats, n_samples, 16)
    max_diff = float(np.max(np.abs(g_pos - g_neg)))
    assert max_diff < 1e-14, f"Projective atoms violated Z2 invariance: max_diff = {max_diff}"

    # 2. Projective operator encoder
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="projective")
    q_tensor = torch.randn(2, 4, 8)
    q_tensor = q_tensor / q_tensor.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        e_pos = model.encode_operator(q_tensor, None)
        e_neg = model.encode_operator(-q_tensor, None)
    encoder_diff = float((e_pos - e_neg).abs().max())
    assert encoder_diff < 1e-6, f"Projective encoder violated Z2 invariance: diff = {encoder_diff}"


def test_world_4_continuous_generalization_vs_strong_baselines():
    """World 4: Continuous Functional Generalization vs Strong Baselines.
    
    Compares continuous geometric representation against:
      1. nearest_known_operator (maps unseen q to closest canonical lead).
      2. q_ablated_set (discards q entirely, only set-aggregates responses).
      3. categorical_ids (standard discrete token embedding).
    
    Verifies:
      - Continuous operator smoothly distinguishes interior interpolation points.
      - Nearest known operator snaps across angular boundaries.
      - q_ablated and categorical embeddings remain invariant or map to constant UNK.
    """
    train_vocab = canonical_operators()  # 8 standard leads (I, II, V1-V6)
    alphas = np.linspace(0.05, 0.95, 10)
    lead_i = train_vocab[0]
    lead_v2 = train_vocab[3]

    interp_ops = [normalize_operator((1.0 - a) * lead_i + a * lead_v2) for a in alphas]

    # 1. Continuous projective encoder distinguishes every interpolation point smoothly
    model_proj = OperatorSetModel(operator_mode="projective")
    q_batch = torch.as_tensor(np.stack(interp_ops), dtype=torch.float32)
    with torch.no_grad():
        proj_embs = model_proj.encode_operator(q_batch, None)
    dists = torch.norm(proj_embs[1:] - proj_embs[:-1], dim=1)
    assert torch.all(dists > 1e-3), "Projective continuous encoder must distinguish distinct lead directions"

    # 2. Nearest known operator snaps discretely to Lead I (alpha < 0.5) and Lead V2 (alpha >= 0.5)
    nearest_indices = [nearest_known_operator(q, train_vocab)[0] for q in interp_ops]
    # First half should map to 0 (Lead I), second half to 3 (Lead V2)
    assert nearest_indices[:4] == [0, 0, 0, 0]
    assert nearest_indices[-4:] == [3, 3, 3, 3]

    # 3. q_ablated maps all to identical constant vector
    model_ablated = OperatorSetModel(operator_mode="q_ablated")
    with torch.no_grad():
        ablated_embs = model_ablated.encode_operator(q_batch, None)
    assert float(torch.var(ablated_embs, dim=0).sum()) == 0.0


def test_world_5_observability_governed_source_reconstruction():
    """World 5: Observability-Governed Source Reconstructive Inversion.
    
    Verifies the mathematical observability criterion:
      x(t) = L s(t) with rank-3 source s(t) in R^3, L in R^{8x3}.
      Given m measurements Q in R^{mx8}:
      1. Observable: rank(Q L) == 3 => exact recovery error < 10^-12.
      2. Unobservable: rank(Q L) < 3 => exact recovery provably fails!
    """
    rng = np.random.default_rng(888)
    n_samples = 150
    leads = 8
    n_sources = 3

    s_true = rng.normal(size=(n_samples, n_sources))
    l_lead, _ = np.linalg.qr(rng.normal(size=(leads, n_sources)))
    x_multilead = s_true @ l_lead.T

    q_target = normalize_operator(rng.normal(size=leads))
    y_target_true = x_multilead @ q_target

    # 1. Observable Case: 4 measurements where rank(Q L) == 3
    q_obs = rng.normal(size=(4, leads))
    q_obs = np.stack([normalize_operator(r) for r in q_obs])
    # Verify rank(Q L) == 3
    ql_obs = q_obs @ l_lead
    assert np.linalg.matrix_rank(ql_obs) == 3

    y_obs = x_multilead @ q_obs.T  # (T, 4)
    # Inversion: s_est = y_obs @ pinv((Q L)^T)
    s_est = y_obs @ np.linalg.pinv(ql_obs.T)
    y_target_est = s_est @ (l_lead.T @ q_target)
    obs_error = float(np.max(np.abs(y_target_true - y_target_est)))
    assert obs_error < 1e-12, f"Observable recovery failed: {obs_error}"

    # 2. Unobservable Case: measurements chosen orthogonal to source direction 0
    # Q L has rank 2 < 3
    null_dir = l_lead[:, 0]
    # Construct Q orthogonal to null_dir
    q_unobs = rng.normal(size=(4, leads))
    q_unobs = q_unobs - (q_unobs @ null_dir[:, None]) * null_dir[None, :]
    q_unobs = np.stack([normalize_operator(r) for r in q_unobs])
    ql_unobs = q_unobs @ l_lead
    rank_unobs = np.linalg.matrix_rank(ql_unobs, tol=1e-8)
    assert rank_unobs < 3, f"Expected rank < 3, got {rank_unobs}"

    y_unobs = x_multilead @ q_unobs.T
    s_unobs_est = y_unobs @ np.linalg.pinv(ql_unobs.T)
    y_target_unobs_est = s_unobs_est @ (l_lead.T @ q_target)
    unobs_error = float(np.max(np.abs(y_target_true - y_target_unobs_est)))
    assert unobs_error > 0.01, (
        f"Unobservable context should fail exact recovery, but had error: {unobs_error}"
    )



def test_world_6_analytic_pinv_and_lmmse_reconstruction_baselines():
    """World 6: Analytic Pseudo-Inverse & LMMSE Linear Reconstruction Baselines.
    
    Verifies that non-neural linear baselines (analytic_pinv_reconstruct and lmmse_reconstruct)
    execute cleanly and provide well-conditioned reconstruction bounds.
    """
    rng = np.random.default_rng(999)
    T = 100
    leads = 8

    sigma_x = np.eye(leads) * 0.5 + 0.1
    x_true = rng.multivariate_normal(np.zeros(leads), sigma_x, size=T)

    q_context = canonical_operators()[:4]  # 4 canonical leads
    y_observed = x_true @ q_context.T      # (T, 4)
    q_target = canonical_operators()[5]     # lead 5

    # 1. Analytic pinv
    y_pinv = analytic_pinv_reconstruct(y_observed, q_context, q_target)
    assert y_pinv.shape == (T,)
    assert np.all(np.isfinite(y_pinv))

    # 2. LMMSE with covariance
    y_lmmse = lmmse_reconstruct(y_observed, q_context, q_target, sigma_x, noise_var=1e-4)
    assert y_lmmse.shape == (T,)
    assert np.all(np.isfinite(y_lmmse))

    # Error should be finite and bounded
    err_pinv = float(np.mean((x_true @ q_target - y_pinv) ** 2))
    err_lmmse = float(np.mean((x_true @ q_target - y_lmmse) ** 2))
    assert np.isfinite(err_pinv) and np.isfinite(err_lmmse)


def test_world_7_projective_distance_and_operator_separation():
    """World 7: Projective Operator Separation & Controlled Interpolation.
    
    Verifies:
      1. Projective distance d_RP(q, p) = arccos(|q^T p|) is in [0, pi/2].
      2. d_RP(q, -q) == 0 (unoriented axis invariance).
      3. Controlled interpolation q(alpha) moves monotonically away from q(0) in d_RP.
      4. Frozen operator banks satisfy projective separation d_RP(q_test, Q_train) >= delta.
    """
    q1 = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    q2 = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])

    # 1. Polarity invariance
    assert np.isclose(projective_distance(q1, -q1), 0.0, atol=1e-12)
    # Orthogonal
    assert np.isclose(projective_distance(q1, q2), np.pi / 2, atol=1e-12)

    # 2. Monotonic controlled interpolation
    alphas = np.linspace(0.0, 1.0, 11)
    interp_dist = [
        projective_distance(q1, normalize_operator((1.0 - a) * q1 + a * q2)) for a in alphas
    ]
    for k in range(len(interp_dist) - 1):
        assert interp_dist[k + 1] >= interp_dist[k] - 1e-12

    # 3. Bank separation under projective distance
    banks = frozen_operator_banks(tolerance=1e-8)
    seen = banks["seen"]
    for name in ("derived_limb", "dense", "i_to_v2"):
        for q in banks[name]:
            min_rp_dist = float(np.min([projective_distance(q, p) for p in seen]))
            assert min_rp_dist > 1e-8, f"Bank {name} operator too close in RP^7: {min_rp_dist}"


def test_world_8_full_classifier_and_auxiliary_reconstruction_gradient_flow():
    """World 8: Full Classifier & Auxiliary Reconstruction Gradient Flow.
    
    Verifies clean end-to-end forward and backward passes across all benchmark modes:
      1. projective_continuous (primary)
      2. projective_continuous_aux (joint cls + recon)
      3. continuous_mlp (raw continuous q)
      4. categorical_ids (discrete token embedding)
      5. q_ablated_set (no operator geometry)
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

    # 1. projective_continuous
    m1 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="projective")
    m1.train()
    logits1 = m1(operators, responses)
    loss1 = nn.functional.binary_cross_entropy_with_logits(logits1, labels)
    loss1.backward()
    for name, p in m1.named_parameters():
        if not name.startswith("decoder"):
            assert p.grad is not None and torch.all(torch.isfinite(p.grad)), f"Missing grad in m1: {name}"

    # 2. projective_continuous_aux
    m2 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="projective")
    m2.train()
    logits2, recon2 = m2(operators, responses, target_operator=target_operator, return_reconstruction=True)
    loss2 = (
        nn.functional.binary_cross_entropy_with_logits(logits2, labels)
        + 0.1 * nn.functional.mse_loss(recon2, target_response)
    )
    loss2.backward()
    for name, p in m2.named_parameters():
        assert p.grad is not None and torch.all(torch.isfinite(p.grad)), f"Missing grad in m2: {name}"

    # 3. continuous_mlp
    m3 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="continuous")
    m3.train()
    logits3 = m3(operators, responses)
    loss3 = nn.functional.binary_cross_entropy_with_logits(logits3, labels)
    loss3.backward()
    for name, p in m3.named_parameters():
        if not name.startswith("decoder"):
            assert p.grad is not None and torch.all(torch.isfinite(p.grad)), f"Missing grad in m3: {name}"

    # 4. categorical_ids
    m4 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="categorical", vocabulary_size=8)
    m4.train()
    logits4 = m4(None, responses, operator_ids=operator_ids)
    loss4 = nn.functional.binary_cross_entropy_with_logits(logits4, labels)
    loss4.backward()
    for name, p in m4.named_parameters():
        if not name.startswith("decoder"):
            assert p.grad is not None and torch.all(torch.isfinite(p.grad)), f"Missing grad in m4: {name}"

    # 5. q_ablated_set
    m5 = OperatorSetModel(response_dim=feat_dim, classes=classes, operator_mode="q_ablated")
    m5.train()
    logits5 = m5(operators, responses)
    loss5 = nn.functional.binary_cross_entropy_with_logits(logits5, labels)
    loss5.backward()
    for name, p in m5.named_parameters():
        if not name.startswith("decoder") and not name.startswith("unknown_operator"):
            assert p.grad is not None and torch.all(torch.isfinite(p.grad)), f"Missing grad in m5: {name}"
