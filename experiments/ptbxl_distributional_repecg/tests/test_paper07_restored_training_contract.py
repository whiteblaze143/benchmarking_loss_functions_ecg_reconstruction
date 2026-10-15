import pytest
import numpy as np
import torch
import torch.nn as nn

from repecg.paper07_operator.model import OperatorSetModel


def test_class_weight_computation():
    """Verify class weights computed from prevalence match the original formula."""
    # Mock labels with known class distribution
    labels = np.array([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    prevalence = torch.as_tensor(labels.mean(axis=0))
    expected_prev = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.1], dtype=torch.float32)
    assert torch.allclose(prevalence, expected_prev)

    pos_weight = ((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    expected_weight = torch.tensor([4.0, 4.0, 4.0, 4.0, 9.0], dtype=torch.float32)
    assert torch.allclose(pos_weight, expected_weight)

    # Test clamping at 10 for very rare classes (< 9.09% prevalence)
    rare_prev = torch.tensor([0.01])
    rare_weight = ((1 - rare_prev) / rare_prev.clamp_min(1e-8)).clamp_max(10)
    assert rare_weight.item() == 10.0


def test_weighted_vs_unweighted_bce_loss():
    """Verify BCEWithLogitsLoss with pos_weight penalizes false negatives on rare classes."""
    logits = torch.tensor([[-2.0, -2.0]])  # Predict negative for both
    targets = torch.tensor([[1.0, 1.0]])   # True positive for both

    unweighted_loss = nn.BCEWithLogitsLoss()(logits, targets)
    pos_weight = torch.tensor([1.0, 5.0])
    weighted_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, targets)

    # Weighted loss must be strictly greater than unweighted loss due to pos_weight > 1 on class 1
    assert weighted_loss > unweighted_loss


def test_context_sampling_law():
    """Verify context size m is strictly in [1, 7], and m=8 NEVER occurs."""
    rng = np.random.default_rng(42)
    n_candidates = 8
    sampled_m = []
    for _ in range(1000):
        m = int(rng.integers(1, min(8, n_candidates)))
        sampled_m.append(m)

    assert min(sampled_m) == 1
    assert max(sampled_m) == 7
    assert 8 not in sampled_m
    # Check all cardinalities 1..7 appear
    for k in range(1, 8):
        assert k in sampled_m


def test_model_forward_logits_reproducibility():
    """Verify OperatorSetModel forward pass produces identical logits given fixed seed."""
    torch.manual_seed(42)
    m1 = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    m1.eval()

    torch.manual_seed(42)
    m2 = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    m2.eval()

    torch.manual_seed(999)
    ops = torch.randn(2, 5, 8)
    res = torch.randn(2, 5, 16, 128)

    with torch.no_grad():
        out1 = m1(ops, res)
        out2 = m2(ops, res)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_one_optimizer_step_and_gradient_clip():
    """Verify training gradients, clip_grad_norm_, and AdamW update."""
    torch.manual_seed(42)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0, 3.0, 3.0, 4.0, 7.0]))

    ops = torch.randn(4, 4, 8)
    res = torch.randn(4, 4, 16, 128)
    targets = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32)

    optimizer.zero_grad(set_to_none=True)
    logits = model(ops, res)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)

    loss.backward()

    # Check active classification parameters have finite gradients
    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm)

    # Initial param snapshot
    p_before = model.head.weight.clone()
    optimizer.step()
    p_after = model.head.weight

    # Ensure parameter updated
    assert not torch.equal(p_before, p_after)


def test_cosine_annealing_lr_progression():
    """Verify CosineAnnealingLR progression over max_epochs."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    lrs = []
    for _ in range(80):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == 1e-3
    assert lrs[-1] < 1e-4
    # LR must be monotonically non-increasing
    assert all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1))


def test_strict_checkpoint_save_and_load(tmp_path):
    """Verify checkpoint saves strict state dict and loads without missing/unexpected keys."""
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    ckpt_path = tmp_path / "test_ckpt.pt"

    torch.save({
        "state_dict": model.state_dict(),
        "seed": 42,
        "intervention": "basis",
        "best_epoch": 10,
        "best_macro_auroc": 0.85,
        "hyperparameters": {"lr": 3e-4, "wd": 1e-3},
    }, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=False)
    new_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    incompatible = new_model.load_state_dict(loaded["state_dict"], strict=True)
    assert len(incompatible.missing_keys) == 0
    assert len(incompatible.unexpected_keys) == 0


def test_end_to_end_smoke_training_loop(tmp_path):
    """End-to-end smoke test verifying train_single_seed on paired Basis and Span arms."""
    from scripts.paper07.train_final_5seeds import train_single_seed

    device = torch.device("cpu")
    n_records = 64
    val_records = 32

    # Construct paired initial state
    torch.manual_seed(42)
    template = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    initial_state = {k: v.clone() for k, v in template.state_dict().items()}

    # Labels with non-trivial prevalence
    rng = np.random.default_rng(42)
    labels = (rng.uniform(size=(n_records, 5)) < 0.25).astype(np.float32)
    val_labels = (rng.uniform(size=(val_records, 5)) < 0.25).astype(np.float32)

    val_data = {
        "operators": rng.standard_normal((val_records, 8, 8)).astype(np.float32),
        "responses": rng.standard_normal((val_records, 8, 16, 128)).astype(np.float32),
        "labels": val_labels,
    }

    # Test Basis Arm (8 candidates)
    basis_data = {
        "operators": rng.standard_normal((n_records, 8, 8)).astype(np.float32),
        "responses": rng.standard_normal((n_records, 8, 16, 128)).astype(np.float32),
        "labels": labels,
    }
    basis_ckpt = train_single_seed(
        seed=42,
        intervention="basis",
        initial_state=initial_state,
        train_data=basis_data,
        val_data=val_data,
        output_dir=tmp_path,
        epochs=2,
        patience=2,
        batch_size=16,
        lr=3e-4,
        wd=1e-3,
        device_str="cpu",
    )
    assert basis_ckpt.exists()
    chk_b = torch.load(basis_ckpt, weights_only=False)
    assert chk_b["intervention"] == "basis"
    assert chk_b["seed"] == 42
    assert "positive_weights" in chk_b["hyperparameters"]

    # Test Span Arm (16 candidates)
    span_data = {
        "operators": rng.standard_normal((n_records, 16, 8)).astype(np.float32),
        "responses": rng.standard_normal((n_records, 16, 16, 128)).astype(np.float32),
        "labels": labels,
    }
    span_ckpt = train_single_seed(
        seed=42,
        intervention="span",
        initial_state=initial_state,
        train_data=span_data,
        val_data=val_data,
        output_dir=tmp_path,
        epochs=2,
        patience=2,
        batch_size=16,
        lr=3e-4,
        wd=1e-3,
        device_str="cpu",
    )
    assert span_ckpt.exists()
    chk_s = torch.load(span_ckpt, weights_only=False)
    assert chk_s["intervention"] == "span"
    assert chk_s["seed"] == 42

    # Verify strict state dict loading for both
    m_test = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    m_test.load_state_dict(chk_b["state_dict"], strict=True)
    m_test.load_state_dict(chk_s["state_dict"], strict=True)

