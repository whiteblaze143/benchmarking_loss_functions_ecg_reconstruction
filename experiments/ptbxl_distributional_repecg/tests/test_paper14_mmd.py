import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from repecg.common.models import (
    InvariantMechanismDiscoveryModel,
    _pairwise_sq_dists,
    _imq_kernel,
    biased_imq_mmd2,
    multi_environment_mmd,
)
from repecg.common.variants import ExperimentVariant, PAPER14_VARIANTS


def test_mmd_identical_samples_is_zero():
    torch.manual_seed(42)
    x = torch.randn(50, 16)
    val = biased_imq_mmd2(x, x, c2=1.0)
    assert torch.isclose(val, torch.tensor(0.0), atol=1e-7), f"Expected 0, got {val}"


def test_mmd_shifted_samples_is_positive():
    torch.manual_seed(42)
    x = torch.randn(50, 16)
    y = x + 2.0
    val = biased_imq_mmd2(x, y, c2=1.0)
    assert val.item() > 0.05, f"Expected positive MMD for shifted samples, got {val}"


def test_mmd_is_symmetric():
    torch.manual_seed(42)
    x = torch.randn(40, 16)
    y = torch.randn(40, 16) + 0.5
    mmd_xy = biased_imq_mmd2(x, y, c2=1.0)
    mmd_yx = biased_imq_mmd2(y, x, c2=1.0)
    assert torch.isclose(mmd_xy, mmd_yx, atol=1e-7), f"Asymmetric MMD: {mmd_xy} vs {mmd_yx}"


def test_mmd_has_nonzero_gradient():
    torch.manual_seed(42)
    x = torch.randn(30, 8, requires_grad=True)
    y = torch.randn(30, 8) + 1.0
    penalty = biased_imq_mmd2(x, y, c2=1.0)
    penalty.backward()
    assert x.grad is not None, "Gradient should exist"
    assert torch.isfinite(x.grad).all(), "Gradient contains non-finite values"
    assert x.grad.abs().sum().item() > 0.0, "Gradient should be non-zero"


def test_mmd_requires_multiple_environments():
    torch.manual_seed(42)
    z = torch.randn(20, 16)
    single_env = torch.zeros(20, dtype=torch.long)
    with pytest.raises(RuntimeError, match="requires at least two environments"):
        InvariantMechanismDiscoveryModel.compute_mmd_penalty(z, single_env)

    # Test single-sample environment
    unbalanced_env = torch.cat([torch.zeros(19, dtype=torch.long), torch.ones(1, dtype=torch.long)])
    with pytest.raises(RuntimeError, match="at least two samples"):
        InvariantMechanismDiscoveryModel.compute_mmd_penalty(z, unbalanced_env)


def test_mmd_variant_changes_total_loss():
    torch.manual_seed(42)
    model = InvariantMechanismDiscoveryModel(input_dim=8, classes=1, variant=PAPER14_VARIANTS["mmd"])
    x = torch.randn(20, 16, 8)
    y = torch.randint(0, 2, (20, 1)).float()
    env = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    
    # Make representations differ by shifting environment 1
    x[10:] = x[10:] + 3.0

    logits, z = model.forward_with_representation(x)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    mmd = model.compute_mmd_penalty(z, env, c2=1.0)

    total_erm = bce
    total_mmd = bce + 1.0 * mmd

    assert mmd.item() > 0.0
    assert total_mmd.item() > total_erm.item()


def test_mmd_variant_changes_encoder_gradient():
    torch.manual_seed(42)
    # Model 1: ERM
    model_erm = InvariantMechanismDiscoveryModel(input_dim=8, width=32, classes=1, variant=PAPER14_VARIANTS["erm"])
    # Model 2: MMD with identical initial weights
    model_mmd = InvariantMechanismDiscoveryModel(input_dim=8, width=32, classes=1, variant=PAPER14_VARIANTS["mmd"])
    model_mmd.load_state_dict(model_erm.state_dict())

    x = torch.randn(20, 16, 8)
    x[10:] = x[10:] + 2.0  # environment shift
    y = torch.randint(0, 2, (20, 1)).float()
    env = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

    # ERM backward
    logits_erm, _ = model_erm.forward_with_representation(x)
    loss_erm = F.binary_cross_entropy_with_logits(logits_erm, y)
    loss_erm.backward()

    # MMD backward
    logits_mmd, z_mmd = model_mmd.forward_with_representation(x)
    bce_mmd = F.binary_cross_entropy_with_logits(logits_mmd, y)
    penalty = model_mmd.compute_mmd_penalty(z_mmd, env, c2=1.0)
    loss_mmd = bce_mmd + 5.0 * penalty
    loss_mmd.backward()

    # Compare encoder conv1 gradients
    grad_erm = model_erm.encoder[0].weight.grad
    grad_mmd = model_mmd.encoder[0].weight.grad

    diff = torch.norm(grad_erm - grad_mmd).item()
    assert diff > 1e-4, f"Expected distinct gradients between ERM and MMD, got diff={diff}"


def test_erm_variant_has_zero_mmd_contribution():
    variant_erm = PAPER14_VARIANTS["erm"]
    assert not variant_erm.use_mmd

    variant_probe = PAPER14_VARIANTS["linear_probe"]
    assert not variant_probe.use_mmd

    variant_mmd = PAPER14_VARIANTS["mmd"]
    assert variant_mmd.use_mmd


def test_environment_balanced_batch_contract():
    # Helper simulating balanced batch partitioning
    def make_balanced_batches(x, y, env, batch_size=20):
        envs = torch.unique(env)
        if len(envs) < 2:
            raise RuntimeError("Paper 14 MMD batch contains fewer than two environments")
        
        per_env = batch_size // len(envs)
        indices = []
        for e in envs:
            e_idx = (env == e).nonzero(as_tuple=True)[0]
            indices.append(e_idx[:per_env])
        batch_idx = torch.cat(indices)
        return x[batch_idx], y[batch_idx], env[batch_idx]

    env = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    x = torch.randn(100, 8)
    y = torch.randint(0, 2, (100, 1)).float()

    bx, by, benv = make_balanced_batches(x, y, env, batch_size=20)
    assert len(bx) == 20
    assert (benv == 0).sum().item() == 10
    assert (benv == 1).sum().item() == 10

    # Fail closed on single environment
    single_env = torch.zeros(100, dtype=torch.long)
    with pytest.raises(RuntimeError, match="fewer than two environments"):
        make_balanced_batches(x, y, single_env, batch_size=20)


def test_mmd_only_can_collapse():
    """G3 collapse demonstration: MMD-only training drives Var(Z) -> 0."""
    torch.manual_seed(42)
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
        def forward(self, x):
            return self.fc(x)

    model = SimpleEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Two environments with different means
    env0 = torch.randn(50, 2) * 0.1 - 2.0
    env1 = torch.randn(50, 2) * 0.1 + 2.0
    x = torch.cat([env0, env1], dim=0)
    env = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])

    initial_var = model(x).var(dim=0).mean().item()
    assert initial_var > 0.05

    # Train only with MMD penalty
    for _ in range(200):
        optimizer.zero_grad()
        z = model(x)
        loss = multi_environment_mmd(z, env, c2=1.0)
        loss.backward()
        optimizer.step()

    final_var = model(x).var(dim=0).mean().item()
    # Var(Z) should collapse towards 0
    assert final_var < 0.01, f"Expected Var(Z) to collapse under MMD-only, got {final_var}"


def test_bce_plus_mmd_preserves_predictive_signal():
    """G1 positive alignment: BCE + lambda*MMD preserves Var(Z) > v_min and maintains accuracy while aligning."""
    torch.manual_seed(42)
    B = 200
    # S is label-informative: Y = 1[S > 0]
    S = torch.randn(B)
    Y = (S > 0.0).float().unsqueeze(1)
    
    # Nuisance shifts across environments
    # Env 0: nuisance has mean -2
    # Env 1: nuisance has mean +2
    nuisance0 = torch.randn(B // 2) - 2.0
    nuisance1 = torch.randn(B // 2) + 2.0
    nuisance = torch.cat([nuisance0, nuisance1])

    # Combined input [B, 2] where feature 0 is S, feature 1 is nuisance
    X = torch.stack([S, nuisance], dim=1)
    env = torch.cat([torch.zeros(B // 2, dtype=torch.long), torch.ones(B // 2, dtype=torch.long)])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(2, 2)
            self.head = nn.Linear(2, 1)
        def forward(self, x):
            z = self.enc(x)
            logits = self.head(z)
            return logits, z

    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(250):
        optimizer.zero_grad()
        logits, z = model(X)
        bce = criterion(logits, Y)
        mmd = multi_environment_mmd(z, env, c2=1.0)
        loss = bce + 2.0 * mmd
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits, z = model(X)
        var_z = z.var(dim=0).mean().item()
        preds = (logits > 0.0).float()
        acc = (preds == Y).float().mean().item()

    assert var_z > 0.1, f"Representation collapsed: Var(Z) = {var_z}"
    assert acc > 0.85, f"Predictive signal destroyed: Acc = {acc}"


def test_matched_marginal_world_exposes_mmd_limitation():
    """G2 matched-marginal failure world:
    Spurious feature has identical marginal distribution across envs,
    but opposite label association:
      Y in {-1, +1}
      S = Y + eps (E=0)
      S = -Y + eps (E=1)
    Because marginal P(S|E=0) == P(S|E=1), MMD cannot detect mechanism reversal.
    Assert that MMD is small despite mechanism reversal.
    """
    torch.manual_seed(42)
    N = 1000
    # Balanced Y in {-1, 1}
    Y0 = (torch.randint(0, 2, (N,)) * 2 - 1).float()
    Y1 = (torch.randint(0, 2, (N,)) * 2 - 1).float()

    # S has same marginal distribution in both envs!
    S0 = Y0 + torch.randn(N) * 0.1
    S1 = -Y1 + torch.randn(N) * 0.1

    z0 = S0.unsqueeze(1)
    z1 = S1.unsqueeze(1)

    mmd2 = biased_imq_mmd2(z0, z1, c2=1.0).item()
    # MMD is near zero because marginals match
    assert mmd2 < 0.01, f"Expected small MMD on matched marginals, got {mmd2}"


def test_scale_escape_approaches_diagonal_floor():
    """G5 scale escape audit:
    Demonstrates that fixed-c^2 biased IMQ MMD on unconstrained representations
    approaches its finite-sample diagonal floor (1/n + 1/m) as a -> infinity.
    Also verifies that LayerNorm(Z) strictly eliminates this escape route.
    """
    torch.manual_seed(42)
    N = 100
    D = 16
    # Fixed non-aligned distributions
    z0 = torch.randn(N, D) - 1.0
    z1 = torch.randn(N, D) + 1.0

    diagonal_floor = 1.0 / N + 1.0 / N  # 0.02

    # Unconstrained: MMD drops towards diagonal floor (0.02)
    mmd_1 = biased_imq_mmd2(1.0 * z0, 1.0 * z1, c2=1.0).item()
    mmd_10 = biased_imq_mmd2(10.0 * z0, 10.0 * z1, c2=1.0).item()
    mmd_1000 = biased_imq_mmd2(1000.0 * z0, 1000.0 * z1, c2=1.0).item()
    assert mmd_1 > mmd_10 > mmd_1000, f"Expected monotonic drop: {mmd_1}, {mmd_10}, {mmd_1000}"
    assert abs(mmd_1000 - diagonal_floor) < 0.005, f"Expected MMD to approach diagonal floor {diagonal_floor}, got {mmd_1000}"

    # With LayerNorm (elementwise_affine=False), representation scale is constrained
    ln = nn.LayerNorm(D, elementwise_affine=False)
    mmd_ln_1 = biased_imq_mmd2(ln(1.0 * z0), ln(1.0 * z1), c2=1.0).item()
    mmd_ln_10 = biased_imq_mmd2(ln(10.0 * z0), ln(10.0 * z1), c2=1.0).item()
    mmd_ln_100 = biased_imq_mmd2(ln(100.0 * z0), ln(100.0 * z1), c2=1.0).item()
    assert abs(mmd_ln_1 - mmd_ln_10) < 1e-5
    assert abs(mmd_ln_1 - mmd_ln_100) < 1e-5


def test_trainable_normalizer_escape_p14_g5b():
    """G5b trainable normalizer escape:
    Verifies that InvariantMechanismDiscoveryModel has no trainable affine parameters
    in its normalization layer (elementwise_affine=False) and is invariant to H -> aH + b.
    """
    model = InvariantMechanismDiscoveryModel(input_dim=8, width=32, classes=1, variant=PAPER14_VARIANTS["mmd"])
    assert len(list(model.norm.parameters())) == 0, "Model norm contains trainable affine parameters"

    h = torch.randn(50, 32)
    z0 = model.norm(h)
    for a in [0.5, 1.0, 2.0, 10.0, 100.0]:
        for b in [-5.0, 0.0, 5.0, 10.0]:
            z_scaled = model.norm(a * h + b)
            diff = torch.max(torch.abs(z_scaled - z0)).item()
            assert diff < 2e-4, f"Failed scale/shift invariance: a={a}, b={b}, diff={diff}"


def test_label_shift_boundary_penalizes_unequal_priors():
    """G6 label-shift boundary:
    When class-conditionals P(X|Y,E) are identical across environments,
    but label prevalence P(Y|E) differs (e.g., 0.9 vs 0.1),
    marginal MMD is strictly positive (> 0.5) despite zero domain nuisance.
    """
    torch.manual_seed(42)
    N = 500
    # Env 0: 90% Y=1
    # Env 1: 10% Y=1
    y0 = (torch.rand(N) < 0.9).float()
    y1 = (torch.rand(N) < 0.1).float()

    x0 = (y0 * 2.0 - 1.0).unsqueeze(1) + torch.randn(N, 1) * 0.1
    x1 = (y1 * 2.0 - 1.0).unsqueeze(1) + torch.randn(N, 1) * 0.1

    mmd_shift = biased_imq_mmd2(x0, x1, c2=1.0).item()
    assert mmd_shift > 0.5, f"Expected substantial marginal MMD under pure label shift, got {mmd_shift}"


def test_intervention_survival_p14_g7b():
    """G7b intervention survival:
    Verifies that acquisition perturbations (gain 0.8x, gain 1.2x, noise 20dB, resampling)
    survive preprocessing and untrained model encoding with non-zero relative distance.
    """
    torch.manual_seed(42)
    B, T, C = 32, 16, 8
    X = torch.randn(B, T, C)
    model = InvariantMechanismDiscoveryModel(input_dim=C, width=32, classes=1, variant=PAPER14_VARIANTS["mmd"])

    # Transforms
    x_g8 = X * 0.8
    x_g12 = X * 1.2
    x_noise = X + torch.randn_like(X) * (X.std() * 0.1)
    down = F.interpolate(X.transpose(1, 2), scale_factor=0.5, mode="linear")
    x_res = F.interpolate(down, size=T, mode="linear").transpose(1, 2)

    for name, x_pert in [("gain_0.8", x_g8), ("gain_1.2", x_g12), ("noise_20db", x_noise), ("resample", x_res)]:
        d_raw = (torch.norm(X - x_pert) / torch.norm(X)).item()
        d_input = (torch.norm(model.encode(X) - model.encode(x_pert)) / torch.norm(model.encode(X))).item()
        assert d_raw > 0.01, f"{name} failed raw survival: {d_raw}"
        assert d_input > 0.01, f"{name} failed input representation survival: {d_input}"


def test_relative_effective_rank_p14_g8():
    """G8 relative effective rank:
    Verifies r_rel = r_eff / (d - 1) computation and ensures non-collapsed rank.
    """
    from repecg.common.models import effective_rank
    torch.manual_seed(42)
    d = 16
    max_d = d - 1
    # Full-rank random features
    z = torch.randn(100, d)
    ln = nn.LayerNorm(d, elementwise_affine=False)
    z_norm = ln(z)
    r_eff = effective_rank(z_norm)
    r_rel = r_eff / max_d
    assert r_rel > 0.2, f"Expected non-degenerate relative rank, got {r_rel}"



