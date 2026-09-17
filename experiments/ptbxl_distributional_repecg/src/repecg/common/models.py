from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================================
# Foundational Building Blocks
# ============================================================================

class CircularResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=0)
        self.conv2 = nn.Conv1d(width, width, 3, padding=0)
        self.activation = nn.GELU()

    def _conv(self, x: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
        return layer(F.pad(x, (1, 1), mode="circular"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self._conv(x, self.conv1))
        x = self._conv(x, self.conv2)
        return self.activation(x + residual)


# ============================================================================
# Paper 01: Distributional Recurrence Operator
# ============================================================================

class RecurrenceCNN(nn.Module):
    """
    Paper 01: Evaluates non-linear recurrence by operating on the N x N 
    Maximum Mean Discrepancy (MMD) pairwise recurrence matrix.
    """
    def __init__(self, classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(32, 128)
        self.head = nn.Linear(128, classes)

    def forward(self, operator: torch.Tensor) -> torch.Tensor:
        if operator.ndim == 3:
            operator = operator[:, None]
        x = self.encoder(operator).flatten(1)
        return self.head(F.gelu(self.projection(x)))


# ============================================================================
# Paper 02: Local Phase Kernel Mean Embeddings
# ============================================================================

class PhaseCNN(nn.Module):
    """
    Paper 02: 1D circular residual convolutional network processing the 
    temporal sequence of 16 phase-cell KME representations.
    """
    def __init__(self, input_dim: int, width: int = 128, blocks: int = 3, classes: int = 5):
        super().__init__()
        self.input = nn.Conv1d(input_dim, width, 1)
        self.blocks = nn.Sequential(*(CircularResidualBlock(width) for _ in range(blocks)))
        self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if phase_features.ndim != 3:
            raise ValueError("expected (batch, phase, feature)")
        x = phase_features.transpose(1, 2)
        x = self.blocks(self.input(x)).mean(dim=-1)
        return self.head(x)


# ============================================================================
# Paper 03: Phase-Cell Signature Path
# ============================================================================

class PathSignatureClassifier(nn.Module):
    """
    Paper 03: Rough Path Theory iterated integrals (signatures) providing
    strict invariance to arbitrary continuous monotonic time-warping.
    Computes Level-1 (displacement) and Level-2 (iterated tensor area) features.
    """
    def __init__(self, input_dim: int, proj_dim: int = 16, classes: int = 5):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        # Signature dimension: level 1 (proj_dim) + level 2 (proj_dim * proj_dim)
        sig_dim = proj_dim + (proj_dim * proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(sig_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, classes)
        )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, T=16, feature_dim)
        x = self.proj(phase_features)  # (batch, T, proj_dim)
        dx = x[:, 1:] - x[:, :-1]      # increments: (batch, T-1, proj_dim)

        # Level 1 signature: cumulative displacement
        sig1 = dx.sum(dim=1)  # (batch, proj_dim)

        # Level 2 signature: iterated integrals sum_{s < t} dx_s tensor dx_t
        # Cumulative sum of previous increments:
        cumsum_dx = torch.cumsum(dx, dim=1)  # (batch, T-1, proj_dim)
        # Shift cumsum by 1 step to represent sum_{s < t} dx_s
        shifted_cumsum = torch.cat([torch.zeros_like(dx[:, :1]), cumsum_dx[:, :-1]], dim=1)
        # Outer product sum:
        # (batch, T-1, proj_dim, 1) * (batch, T-1, 1, proj_dim)
        level2_terms = shifted_cumsum.unsqueeze(-1) * dx.unsqueeze(-2)
        sig2 = level2_terms.sum(dim=1).flatten(1)  # (batch, proj_dim * proj_dim)

        signature = torch.cat([sig1, sig2], dim=1)
        return self.classifier(signature)


# ============================================================================
# Paper 04: Hankel Dynamics & Dynamic Mode Decomposition
# ============================================================================

class HankelDynamicsModel(nn.Module):
    """
    Paper 04: Dynamic Mode Decomposition (DMD) over multi-lag block Hankel matrices.
    Extracts complex dynamic modes and decay eigenvalues from local phase transitions.
    """
    def __init__(self, input_dim: int, proj_dim: int = 16, lag: int = 4, classes: int = 5):
        super().__init__()
        self.lag = lag
        self.proj = nn.Linear(input_dim, proj_dim)
        hankel_dim = proj_dim * lag
        self.head = nn.Sequential(
            nn.Linear(hankel_dim * hankel_dim, 128),
            nn.GELU(),
            nn.Linear(128, classes)
        )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, T=16, feature_dim)
        x = self.proj(phase_features)  # (batch, T, proj_dim)
        B, T, D = x.shape
        M = T - self.lag + 1

        # Construct block Hankel matrix: (batch, lag * D, M)
        hankel_blocks = []
        for i in range(self.lag):
            hankel_blocks.append(x[:, i : i + M, :].transpose(1, 2))  # (B, D, M)
        H = torch.cat(hankel_blocks, dim=1)  # (B, lag * D, M)

        # Time-shifted snapshot matrices:
        # H1: first M-1 columns, H2: second to M columns
        H1 = H[:, :, :-1]  # (B, lag*D, M-1)
        H2 = H[:, :, 1:]   # (B, lag*D, M-1)

        # Compute operator A = H2 @ H1^T @ (H1 @ H1^T + reg * I)^-1
        H1T = H1.transpose(1, 2)
        C11 = torch.bmm(H1, H1T)
        reg = 1e-4 * torch.eye(C11.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
        C11_reg = C11 + reg
        C21 = torch.bmm(H2, H1T)

        A = torch.linalg.solve(C11_reg, C21)  # (B, lag*D, lag*D)
        features = A.flatten(1)
        return self.head(features)


# ============================================================================
# Paper 05: The Koopman Operator in Lifted RKHS
# ============================================================================

class KoopmanOperatorModel(nn.Module):
    """
    Paper 05: Extended Dynamic Mode Decomposition (EDMD) exploiting the property
    that non-linear dynamics become linear when lifted into the RKHS: Z_{t+1} = K Z_t.
    """
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width)
        )
        # Learnable global Koopman transition operator
        self.koopman_k = nn.Parameter(torch.eye(width) + 0.01 * torch.randn(width, width))
        self.head = nn.Sequential(
            nn.Linear(width * 2, 128),
            nn.GELU(),
            nn.Linear(128, classes)
        )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, T=16, feature_dim)
        z = self.lift(phase_features)  # (batch, T, width)
        
        # Static invariant representation: mean over phases
        z_mean = z.mean(dim=1)  # (batch, width)
        
        # Dynamical transition representation: step-ahead prediction error / spectral feature
        z_pred = torch.matmul(z[:, :-1], self.koopman_k)  # (batch, T-1, width)
        dyn_residual = (z[:, 1:] - z_pred).mean(dim=1)    # (batch, width)
        
        features = torch.cat([z_mean, dyn_residual], dim=-1)
        return self.head(features)

    def forward_koopman_loss(self, phase_features: torch.Tensor) -> torch.Tensor:
        """Computes auxiliary linearity reconstruction loss for the Koopman operator."""
        z = self.lift(phase_features)
        z_pred = torch.matmul(z[:, :-1], self.koopman_k)
        return F.mse_loss(z_pred, z[:, 1:])


# ============================================================================
# Paper 06: Conditional RepStat
# ============================================================================

class ConditionalRepStatModel(nn.Module):
    """
    Paper 06: Conditional KME with sequential orthogonalization.
    Projects each phase representation onto the orthogonal complement of the
    prior phase subspace to strictly eliminate collinear diagnostic redundancy.
    """
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU()
        )
        self.circular_blocks = nn.Sequential(
            CircularResidualBlock(width),
            CircularResidualBlock(width)
        )
        self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, T=16, feature_dim)
        z = self.encoder(phase_features)  # (batch, T, width)

        # Causal orthogonalization: project z_t onto orthogonal complement of z_{<t}
        # Orthogonal innovations Z_perp
        ortho_innovations = [z[:, 0]]
        subspace_basis = z[:, 0:1]  # (B, 1, width)

        for t in range(1, z.shape[1]):
            zt = z[:, t : t + 1]  # (B, 1, width)
            # Projection onto subspace_basis: P = B (B^T B + eps)^-1 B^T
            # Gram matrix: (B, k, k)
            gram = torch.bmm(subspace_basis, subspace_basis.transpose(1, 2))
            reg = 1e-4 * torch.eye(gram.shape[1], device=z.device, dtype=z.dtype).unsqueeze(0)
            proj_weights = torch.linalg.solve(gram + reg, torch.bmm(subspace_basis, zt.transpose(1, 2)))
            proj_zt = torch.bmm(proj_weights.transpose(1, 2), subspace_basis)
            zt_perp = zt - proj_zt
            ortho_innovations.append(zt_perp.squeeze(1))
            subspace_basis = torch.cat([subspace_basis, zt_perp], dim=1)

        z_ortho = torch.stack(ortho_innovations, dim=1)  # (B, T, width)
        # Apply circular residual convolutions over orthogonal innovations
        x = z_ortho.transpose(1, 2)
        feat = self.circular_blocks(x).mean(dim=-1)
        return self.head(feat)


# ============================================================================
# Paper 07: Operator Reconstruction Auxiliary
# ============================================================================

class OperatorReconstructionAuxiliary(nn.Module):
    """
    Paper 07: Dual-head diagnostic + physical 8-lead waveform reconstruction.
    Guarantees biophysical retention and prevents shortcut learning via a generative bottleneck.
    """
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, width, 1),
            CircularResidualBlock(width),
            CircularResidualBlock(width)
        )
        self.head = nn.Linear(width, classes)
        
        # Generative physical reconstruction decoder:
        # Maps latent vector (width) to (8 leads x 256 physical samples)
        self.decoder = nn.Sequential(
            nn.Linear(width, 256),
            nn.GELU(),
            nn.Linear(256, 8 * 256)
        )

    def forward(self, phase_features: torch.Tensor, return_reconstruction: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # phase_features: (batch, T=16, feature_dim)
        x = phase_features.transpose(1, 2)
        latent = self.encoder(x).mean(dim=-1)  # (batch, width)
        logits = self.head(latent)

        if return_reconstruction:
            recon = self.decoder(latent).view(-1, 8, 256)
            return logits, recon
        return logits


# ============================================================================
# Paper 08: Local Token Cross-Attention
# ============================================================================

class LocalTokenCrossAttention(nn.Module):
    """
    Paper 08: Multi-head self- and cross-attention Transformer across 16 phase tokens.
    Dynamically routes information between cardiac intervals (P-R vs ST-T) on a patient-specific basis.
    """
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, classes: int = 5):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, 16, feature_dim)
        B = phase_features.shape[0]
        x = self.input_proj(phase_features) + self.pos_embedding
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, x], dim=1)  # (B, 17, d_model)
        
        out = self.transformer(tokens)
        cls_out = out[:, 0]  # (B, d_model)
        return self.head(cls_out)


# ============================================================================
# Paper 09: Counterfactual Measurement-Operator
# ============================================================================

class CounterfactualMeasurementOperator(nn.Module):
    """
    Paper 09: Disentangles intrinsic 3D dipole state Z_S from spatial measurement
    operator q, enabling synthesis of counterfactual lead angles.
    """
    def __init__(self, input_dim: int, dipole_dim: int = 16, classes: int = 5):
        super().__init__()
        # Maps phase features to intrinsic dipole representation across 16 phases
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, dipole_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(dipole_dim * 16, 128),
            nn.GELU(),
            nn.Linear(128, classes)
        )
        # Learnable measurement projection operator Q (8 leads x dipole_dim)
        self.measurement_operator = nn.Parameter(torch.randn(8, dipole_dim) / math.sqrt(dipole_dim))

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, 16, feature_dim)
        dipole_states = self.encoder(phase_features)  # (batch, 16, dipole_dim)
        # Diagnostic head relies strictly on the invariant biological dipole state
        return self.head(dipole_states.flatten(1))

    def counterfactual_synthesis(self, phase_features: torch.Tensor, lead_operator: torch.Tensor | None = None) -> torch.Tensor:
        """Projects biological dipole onto an arbitrary query measurement operator."""
        dipole_states = self.encoder(phase_features)  # (batch, 16, dipole_dim)
        Q = lead_operator if lead_operator is not None else self.measurement_operator
        return torch.matmul(dipole_states, Q.t())  # (batch, 16, num_leads)


# ============================================================================
# Paper 10: Interventional RepStat
# ============================================================================

class InterventionalRepStatModel(nn.Module):
    """
    Paper 10: Multi-hospital acquisition invariance via representation factorization
    into acquisition latents Z_A and diagnostic invariant latents Z_S.
    """
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, width, 1),
            CircularResidualBlock(width),
            CircularResidualBlock(width)
        )
        # Factorization heads
        self.zs_proj = nn.Linear(width, width // 2)
        self.za_proj = nn.Linear(width, width // 2)
        self.diag_head = nn.Linear(width // 2, classes)
        self.acq_head = nn.Linear(width // 2, 9)  # 9 hospital domains

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        x = phase_features.transpose(1, 2)
        h = self.backbone(x).mean(dim=-1)
        z_s = self.zs_proj(h)
        return self.diag_head(z_s)

    def factorize(self, phase_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = phase_features.transpose(1, 2)
        h = self.backbone(x).mean(dim=-1)
        return self.zs_proj(h), self.za_proj(h)


# ============================================================================
# Paper 11: Causal-State ECG via epsilon-Machines
# ============================================================================

class CausalStateECGModel(nn.Module):
    """
    Paper 11: Computational Mechanics epsilon-machine architecture.
    Discovers minimal sufficient causal states based on statistical equivalence of futures.
    """
    def __init__(self, input_dim: int, num_states: int = 8, state_dim: int = 64, classes: int = 5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, state_dim)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        # Soft clustering into causal states
        self.state_classifier = nn.Linear(state_dim, num_states)
        self.head = nn.Linear(num_states * 16, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, 16, feature_dim)
        z = F.gelu(self.encoder(phase_features))
        h, _ = self.gru(z)  # (batch, 16, state_dim)
        # Soft causal state assignment probabilities
        state_probs = F.softmax(self.state_classifier(h), dim=-1)  # (batch, 16, num_states)
        return self.head(state_probs.flatten(1))


# ============================================================================
# Paper 12: Structural-Innovation repECG
# ============================================================================

class StructuralInnovationModel(nn.Module):
    """
    Paper 12: Isolates unexpected structural innovation U_g = Z_g - mu(Z_{<g})
    via causal autoregressive filtering, focusing strictly on pathological deviations.
    """
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5):
        super().__init__()
        self.encoder = nn.Linear(input_dim, width)
        self.ar_gru = nn.GRU(width, width, batch_first=True)
        self.pred_mean = nn.Linear(width, width)
        self.head = nn.Sequential(
            nn.Linear(width * 16, 128),
            nn.GELU(),
            nn.Linear(128, classes)
        )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.encoder(phase_features))  # (batch, 16, width)
        
        # Autoregressive past context
        # Shift inputs right by 1 step so z_t only receives z_{<t}
        zeros = torch.zeros_like(z[:, :1])
        z_past = torch.cat([zeros, z[:, :-1]], dim=1)
        h_past, _ = self.ar_gru(z_past)
        z_expected = self.pred_mean(h_past)
        
        # Structural innovation (unpredicted residual)
        u = z - z_expected
        return self.head(u.flatten(1))


# ============================================================================
# Paper 13: Counterfactual Distribution Surgery
# ============================================================================

class CounterfactualSurgeryModel(nn.Module):
    """
    Paper 13: Implements Pearl's do() calculus: do(P_g = P_g^{ref}).
    Permits surgical phase substitution to mathematically verify necessity and sufficiency.
    """
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, width, 1),
            CircularResidualBlock(width),
            CircularResidualBlock(width)
        )
        self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        x = phase_features.transpose(1, 2)
        h = self.backbone(x).mean(dim=-1)
        return self.head(h)

    def do_surgery(self, phase_features: torch.Tensor, target_phase: int, reference_dist: torch.Tensor) -> torch.Tensor:
        """Surgically replaces target_phase with a reference healthy distribution."""
        features_surg = phase_features.clone()
        features_surg[:, target_phase] = reference_dist
        return self.forward(features_surg)


# ============================================================================
# Paper 14: Invariant Mechanism Discovery with repStat
# ============================================================================

class InvariantMechanismDiscoveryModel(nn.Module):
    """
    Paper 14: Invariant Risk Minimization (IRMv1) across multi-hospital domains.
    Discovers representations where the optimal diagnostic classifier is invariant across environments.
    """
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Conv1d(input_dim, width, 1),
            CircularResidualBlock(width),
            CircularResidualBlock(width)
        )
        self.linear_head = nn.Linear(width, classes, bias=False)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        x = phase_features.transpose(1, 2)
        rep = self.phi(x).mean(dim=-1)
        return self.linear_head(rep)

    def compute_irm_penalty(self, phase_features: torch.Tensor, labels: torch.Tensor, dummy_w: torch.Tensor) -> torch.Tensor:
        """Computes IRM gradient norm penalty: ||grad_{w=1.0} Loss_e(w * Phi(X))||^2."""
        x = phase_features.transpose(1, 2)
        rep = self.phi(x).mean(dim=-1)
        logits = self.linear_head(rep) * dummy_w
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        grad = torch.autograd.grad(loss, [dummy_w], create_graph=True)[0]
        return grad.pow(2).sum()


# ============================================================================
# Paper 15: Causal Mechanism Factorization
# ============================================================================

class CausalMechanismFactorizationModel(nn.Module):
    """
    Paper 15: Independent Causal Mechanisms (ICM).
    Factorizes the cardiac cycle into 15 autonomous, decoupled transition operators M_g.
    """
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5):
        super().__init__()
        self.init_proj = nn.Linear(input_dim, width)
        # 15 distinct, autonomous transition operators M_1 through M_15
        self.mechanisms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width)
            ) for _ in range(15)
        ])
        self.head = nn.Linear(width * 16, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        # phase_features: (batch, 16, feature_dim)
        z0 = F.gelu(self.init_proj(phase_features[:, 0]))  # (batch, width)
        trajectory = [z0]
        
        current_z = z0
        for i, mech in enumerate(self.mechanisms):
            next_z = current_z + mech(current_z)  # residual autonomous mechanism
            trajectory.append(next_z)
            current_z = next_z

        traj_tensor = torch.stack(trajectory, dim=1)  # (batch, 16, width)
        return self.head(traj_tensor.flatten(1))
