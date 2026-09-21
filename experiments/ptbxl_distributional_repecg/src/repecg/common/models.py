from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .variants import ExperimentVariant

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

class StandardResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=1)
        self.conv2 = nn.Conv1d(width, width, 3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        return self.activation(x + residual)


# ============================================================================
# Paper 01: Distributional Recurrence Operator
# ============================================================================

class RecurrenceCNN(nn.Module):
    def __init__(self, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
            
        self.projection = nn.Linear(32, 128)
        self.head = nn.Linear(128, classes)
        if self.variant.head == "linear":
            self.linear_probe = nn.Linear(120, classes)

    def forward(self, operator: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            if operator.shape[1:] != (16, 16):
                raise ValueError("Paper 1 linear probe expects a 16x16 recurrence operator")
            upper = torch.triu_indices(16, 16, offset=1, device=operator.device)
            return self.linear_probe(operator[:, upper[0], upper[1]])
            
        if operator.ndim == 3 and operator.shape[1:] == (16, 16):
            operator = operator[:, None]
        if operator.ndim != 4 or operator.shape[1:] != (1, 16, 16):
            raise ValueError("Paper 1 expects a 16x16 recurrence operator")
        x = self.encoder(operator).flatten(1)
        return self.head(F.gelu(self.projection(x)))


# ============================================================================
# Paper 02: Local Phase Kernel Mean Embeddings
# ============================================================================

class PhaseCNN(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, blocks: int = 3, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.input = nn.Conv1d(input_dim, width, 1)
        
        if self.variant.mechanism == "no_circular":
            self.blocks = nn.Sequential(*(StandardResidualBlock(width) for _ in range(blocks)))
        else:
            self.blocks = nn.Sequential(*(CircularResidualBlock(width) for _ in range(blocks)))
            
        if self.variant.head == "phase_kme_linear":
            self.head = nn.Linear(16 * input_dim, classes)
        elif self.variant.head in ("linear", "global_kme_linear"):
            self.head = nn.Linear(input_dim, classes)
        elif self.variant.head == "phase_aware":
            self.head = nn.Linear(16 * width, classes)
        else:
            self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if phase_features.ndim != 3:
            raise ValueError("expected (batch, phase, feature)")
            
        if self.variant.mechanism == "global_kme":
            # Pool across all 16 phase cells to remove phase conditioning, then broadcast back
            phase_features = phase_features.mean(dim=1, keepdim=True).expand(-1, phase_features.shape[1], -1)
            
        if self.variant.head == "phase_kme_linear":
            return self.head(phase_features.flatten(1))
        if self.variant.head in ("linear", "global_kme_linear"):
            return self.head(phase_features.mean(dim=1))
        
        x = phase_features.transpose(1, 2)
        features = self.blocks(self.input(x))
        if self.variant.head == "phase_aware":
            return self.head(features.flatten(1))
        return self.head(features.mean(dim=-1))


# ============================================================================
# Paper 03: Phase-Cell Signature Path
# ============================================================================

class PathSignatureClassifier(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 16, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.proj = nn.Linear(input_dim, proj_dim)
        
        sig_dim = proj_dim + (proj_dim * proj_dim)
        if self.variant.mechanism == "signature_level1":
            sig_dim = proj_dim
            
        if self.variant.head == "linear":
            sig_dim = input_dim
            
        self.classifier = nn.Sequential(
            nn.Linear(sig_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, classes)
        )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.classifier(phase_features.mean(dim=1))
            
        if self.variant.mechanism == "scramble_local_order" or self.variant.control == "time_reverse":
            # Scrambling/reversing happens in the dataloader or feature extractor, 
            # but we can optionally handle reversing here for simplicity
            if self.variant.control == "time_reverse":
                phase_features = torch.flip(phase_features, dims=[1])
            elif self.variant.mechanism == "scramble_local_order":
                idx = torch.randperm(phase_features.shape[1], device=phase_features.device)
                phase_features = phase_features[:, idx, :]
                
        x = self.proj(phase_features)
        dx = x[:, 1:] - x[:, :-1]
        sig1 = dx.sum(dim=1)
        
        if self.variant.mechanism == "signature_level1":
            return self.classifier(sig1)
            
        cumsum_dx = torch.cumsum(dx, dim=1)
        shifted_cumsum = torch.cat([torch.zeros_like(dx[:, :1]), cumsum_dx[:, :-1]], dim=1)
        level2_terms = shifted_cumsum.unsqueeze(-1) * dx.unsqueeze(-2)
        sig2 = level2_terms.sum(dim=1).flatten(1)
        signature = torch.cat([sig1, sig2], dim=1)
        
        return self.classifier(signature)


# ============================================================================
# Paper 04: Hankel Dynamics & Dynamic Mode Decomposition
# ============================================================================

class HankelDynamicsModel(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 4, lag: int = 4, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        if self.variant.mechanism == "paper02_phasecnn":
            self.phase_cnn = PhaseCNN(input_dim=input_dim, classes=classes)
        elif self.variant.head == "flat_phase_mlp":
            self.mlp_head = nn.Sequential(nn.Linear(16 * input_dim, 128), nn.GELU(), nn.Linear(128, classes))
        elif self.variant.head == "phase_aware_linear":
            self.linear_head = nn.Linear(16 * input_dim, classes)
        elif self.variant.head == "linear":
            self.linear_head = nn.Linear(input_dim, classes)
        else:
            if self.variant.mechanism in ("lag1", "markovian_lag1"):
                self.lag = 1
            elif self.variant.mechanism == "lag2":
                self.lag = 2
            else:
                self.lag = lag
            self.proj_dim = proj_dim
            self.proj = nn.Linear(input_dim, self.proj_dim)
            hankel_dim = self.proj_dim * self.lag
            if self.variant.head == "operator_summary_probe":
                self.head = nn.Sequential(nn.Linear(8, 64), nn.GELU(), nn.Linear(64, classes))
            else:
                self.head = nn.Sequential(
                    nn.Linear(hankel_dim * hankel_dim, 128),
                    nn.GELU(),
                    nn.Linear(128, classes),
                )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.mechanism == "paper02_phasecnn":
            return self.phase_cnn(phase_features)
        if self.variant.head == "flat_phase_mlp":
            return self.mlp_head(phase_features.flatten(1))
        if self.variant.head == "phase_aware_linear":
            return self.linear_head(phase_features.flatten(1))
        if self.variant.head == "linear":
            return self.linear_head(phase_features.mean(dim=1))

        if self.variant.mechanism == "time_shuffle_local":
            idx = torch.randperm(phase_features.shape[1], device=phase_features.device)
            phase_features = phase_features[:, idx, :]
        elif self.variant.mechanism == "time_reversed":
            phase_features = phase_features.flip(dims=[1])

        x = self.proj(phase_features)
        B, T, D = x.shape
        if self.variant.mechanism == "open_chain":
            M = T - self.lag + 1
            blocks = [x[:, i : i + M, :] for i in range(self.lag)]
            H = torch.cat(blocks, dim=-1).transpose(1, 2)
            X = H[:, :, :-1]
            Y = H[:, :, 1:]
        else:
            blocks = [torch.roll(x, shifts=-l, dims=1) for l in range(self.lag)]
            H = torch.cat(blocks, dim=-1)
            X = H.transpose(1, 2)
            Y = torch.roll(H, shifts=-1, dims=1).transpose(1, 2)

        alpha = 1e-3 if self.variant.mechanism == "ridge_strength_sensitivity" else 1e-4
        DH = X.shape[1]
        M_col = X.shape[2]
        if DH <= M_col:
            C11 = torch.bmm(X, X.transpose(-1, -2)).float()
            C21 = torch.bmm(Y, X.transpose(-1, -2)).float()
            scale = C11.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            reg_lambda = torch.clamp(alpha * scale, min=1e-6)
            identity = torch.eye(DH, device=x.device, dtype=torch.float32).unsqueeze(0)
            C11_reg = C11 + reg_lambda[:, None, None] * identity
            A = torch.linalg.solve(C11_reg, C21.transpose(-1, -2)).transpose(-1, -2).to(x.dtype)
        else:
            G = torch.bmm(X.transpose(-1, -2), X).float()
            scale = G.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
            reg_lambda = torch.clamp(alpha * scale, min=1e-6)
            identity = torch.eye(M_col, device=x.device, dtype=torch.float32).unsqueeze(0)
            G_reg = G + reg_lambda[:, None, None] * identity
            Z = torch.linalg.solve(G_reg, X.transpose(-1, -2).float())
            A = torch.bmm(Y.float(), Z).to(x.dtype)
            C11_reg = torch.bmm(X, X.transpose(-1, -2)).float() + reg_lambda[:, None, None] * torch.eye(DH, device=x.device, dtype=torch.float32).unsqueeze(0)

        if self.variant.head == "operator_summary_probe":
            A_f = A.float()
            X_f = X.float()
            Y_f = Y.float()
            frob = torch.norm(A_f, p="fro", dim=(-2, -1))
            spec = torch.linalg.matrix_norm(A_f, ord=2)
            tr = torch.diagonal(A_f, dim1=-2, dim2=-1).sum(dim=-1)
            A2 = torch.bmm(A_f, A_f)
            tr2 = torch.diagonal(A2, dim1=-2, dim2=-1).sum(dim=-1)
            A3 = torch.bmm(A2, A_f)
            tr3 = torch.diagonal(A3, dim1=-2, dim2=-1).sum(dim=-1)
            res = torch.norm(Y_f - torch.bmm(A_f, X_f), p="fro", dim=(-2, -1)) / torch.norm(Y_f, p="fro", dim=(-2, -1)).clamp_min(1e-6)
            comm = torch.bmm(A_f.transpose(-1, -2), A_f) - torch.bmm(A_f, A_f.transpose(-1, -2))
            non_norm = torch.norm(comm, p="fro", dim=(-2, -1))
            cond = torch.linalg.cond(C11_reg).clamp_max(1e6)
            log_cond = torch.log(cond.clamp_min(1.0))
            summary_vec = torch.stack([frob, spec, tr, tr2, tr3, res, non_norm, log_cond], dim=-1).to(x.dtype)
            return self.head(summary_vec)

        return self.head(A.flatten(1))


# ============================================================================
# Paper 05: The Koopman Operator in Lifted RKHS
# ============================================================================

class KoopmanOperatorModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, classes),
        )
        self.linear_probe = nn.Linear(input_dim, classes)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.ndim != 2:
            raise ValueError("Paper 5 expects fixed record descriptors with shape (batch, feature)")
        x = descriptor
        if self.variant.mechanism == "operator_only":
            mask = torch.ones_like(x)
            mask[:, :min(32, x.shape[1])] = 0.0
            x = x * mask
        elif self.variant.mechanism == "occupancy_only":
            mask = torch.zeros_like(x)
            mask[:, :min(32, x.shape[1])] = 1.0
            x = x * mask
        elif self.variant.mechanism == "spectral_only":
            mask = torch.zeros_like(x)
            if x.shape[1] >= 96:
                mask[:, 96:] = 1.0
            x = x * mask

        if self.variant.head in ("linear", "phase_aware_linear"):
            return self.linear_probe(x)
        return self.network(x)


# ============================================================================
# Paper 06: Conditional RepStat
# ============================================================================

class ConditionalRepStatModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 32, classes: int = 5, variant: ExperimentVariant | None = None, in_channels: int = 2):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        in_c = getattr(self.variant, "in_channels", in_channels)
        self.in_channels = in_c
        self.encoder = nn.Sequential(
            nn.Conv2d(in_c, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, width, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        if self.variant.head in ("linear", "phase_aware_linear"):
            self.linear_probe = nn.Linear(120 * in_c, classes)
        else:
            self.head = nn.Linear(width, classes)

    def forward(self, recurrence: torch.Tensor) -> torch.Tensor:
        if recurrence.ndim != 4 or recurrence.shape[2:] != (16, 16):
            raise ValueError("Paper 6 expects (batch, channels, 16, 16) recurrence tensors")
        if recurrence.shape[1] != self.in_channels:
            raise ValueError(f"Paper 6 expected {self.in_channels} channels, got {recurrence.shape[1]}")
        if self.variant.head in ("linear", "phase_aware_linear"):
            upper = torch.triu_indices(16, 16, offset=1, device=recurrence.device)
            values = recurrence[:, :, upper[0], upper[1]].flatten(1)
            return self.linear_probe(values)
        return self.head(self.encoder(recurrence).flatten(1))


# ============================================================================
# Paper 07: Operator Reconstruction Auxiliary
# ============================================================================

class OperatorReconstructionAuxiliary(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.encoder = nn.Sequential(nn.Conv1d(input_dim, width, 1), CircularResidualBlock(width), CircularResidualBlock(width))
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
            self.decoder = nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 8 * 256))
        else:
            self.head = nn.Linear(width, classes)
            self.decoder = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 8 * 256))
            
        if self.variant.mechanism == "categorical_lead_id":
            self.lead_embeddings = nn.Parameter(torch.randn(8, input_dim))

    def forward(self, phase_features: torch.Tensor, return_reconstruction: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.variant.mechanism == "categorical_lead_id":
            # Replace continuous q with learned lead embeddings of matched dimension
            B, T, _ = phase_features.shape
            phase_features = self.lead_embeddings.unsqueeze(0).expand(B, -1, -1)
            
        if self.variant.head == "linear":
            latent = phase_features.mean(dim=1)
        else:
            x = phase_features.transpose(1, 2)
            latent = self.encoder(x).mean(dim=-1)
            
        logits = self.head(latent)
        if return_reconstruction:
            recon = self.decoder(latent).view(-1, 8, 256)
            return logits, recon
        return logits


# ============================================================================
# Paper 08: Local Token Cross-Attention / Phase-Token Transformer
# ============================================================================

from repecg.paper08_tokens.model import PhaseTokenTransformer


class LocalTokenCrossAttention(PhaseTokenTransformer):
    """
    Backwards-compatible wrapper delegating to PhaseTokenTransformer.
    Supports global, local banded, static, uniform, and phase-agnostic attention modes.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        classes: int = 5,
        variant: ExperimentVariant | None = None,
        bandwidth: int = 2,
    ):
        super().__init__(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            classes=classes,
            variant=variant,
            bandwidth=bandwidth,
        )



# ============================================================================
# Paper 09: Counterfactual Measurement-Operator
# ============================================================================

class CounterfactualMeasurementOperator(nn.Module):
    def __init__(self, input_dim: int, dipole_dim: int = 16, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, dipole_dim))
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Sequential(nn.Linear(dipole_dim * 16, 128), nn.GELU(), nn.Linear(128, classes))
            
        self.measurement_operator = nn.Parameter(torch.randn(8, dipole_dim) / math.sqrt(dipole_dim))

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
        dipole_states = self.encoder(phase_features)
        return self.head(dipole_states.flatten(1))

    def counterfactual_synthesis(self, phase_features: torch.Tensor, lead_operator: torch.Tensor | None = None) -> torch.Tensor:
        dipole_states = self.encoder(phase_features)
        
        Q = lead_operator if lead_operator is not None else self.measurement_operator
        
        if self.variant.control == "mismatch_q_waveform":
            # Deliberately mismatch (q_i, X_{q_j}) by shuffling Q
            idx = torch.randperm(Q.shape[0], device=Q.device)
            Q = Q[idx]
            
        return torch.matmul(dipole_states, Q.t())


# ============================================================================
# Paper 10: Interventional RepStat
# ============================================================================

class InterventionalRepStatModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5, environments: int = 9, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.backbone = nn.Sequential(nn.Conv1d(input_dim, width, 1), CircularResidualBlock(width), CircularResidualBlock(width))
        self.zs_proj = nn.Linear(width, width // 2)
        self.za_proj = nn.Linear(width, width // 2)
        
        if self.variant.head == "linear":
            self.diag_head = nn.Linear(input_dim, classes)
        else:
            self.diag_head = nn.Linear(width // 2, classes)
        self.acq_head = nn.Linear(width // 2, environments)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.diag_head(phase_features.mean(dim=1))
        x = phase_features.transpose(1, 2)
        h = self.backbone(x).mean(dim=-1)
        z_s = self.zs_proj(h)
        return self.diag_head(F.normalize(z_s, dim=-1))

    def factorize(self, phase_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = phase_features.transpose(1, 2)
        h = self.backbone(x).mean(dim=-1)
        return self.zs_proj(h), self.za_proj(h)


# ============================================================================
# Paper 11: Causal-State ECG via epsilon-Machines
# ============================================================================

class CausalStateECGModel(nn.Module):
    def __init__(self, input_dim: int, num_states: int = 8, state_dim: int = 64, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.encoder = nn.Linear(input_dim, state_dim)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.state_classifier = nn.Linear(state_dim, num_states)
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Linear(num_states * 16, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
            
        if self.variant.mechanism == "shuffle_history_futures":
            # Shuffle features along time to destroy history-future relation before state construction
            idx = torch.randperm(phase_features.shape[1], device=phase_features.device)
            phase_features = phase_features[:, idx, :]
            
        z = F.gelu(self.encoder(phase_features))
        h, _ = self.gru(z)
        state_probs = F.softmax(self.state_classifier(h), dim=-1)
        return self.head(state_probs.flatten(1))


# ============================================================================
# Paper 12: Structural-Innovation repECG
# ============================================================================

class StructuralInnovationModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.encoder = nn.Linear(input_dim, width)
        self.ar_gru = nn.GRU(width, width, batch_first=True)
        self.pred_mean = nn.Linear(width, width)
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Sequential(nn.Linear(width * 16, 128), nn.GELU(), nn.Linear(128, classes))

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
            
        z = F.gelu(self.encoder(phase_features))
        
        if self.variant.mechanism == "unconditional_state":
            # Replace p(Z_g | Z_<g) with p(Z_g), passing raw states instead of innovations
            return self.head(z.flatten(1))
            
        zeros = torch.zeros_like(z[:, :1])
        z_past = torch.cat([zeros, z[:, :-1]], dim=1)
        h_past, _ = self.ar_gru(z_past)
        z_expected = self.pred_mean(h_past)
        u = z - z_expected
        return self.head(u.flatten(1))


# ============================================================================
# Paper 13: Counterfactual Distribution Surgery
# ============================================================================

class CounterfactualSurgeryModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.backbone = nn.Sequential(nn.Conv1d(input_dim, width, 1), CircularResidualBlock(width), CircularResidualBlock(width))
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Linear(width, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
        x = phase_features.transpose(1, 2)
        h = self.backbone(x).mean(dim=-1)
        return self.head(h)

    def do_surgery(self, phase_features: torch.Tensor, target_phase: int, reference_dist: torch.Tensor) -> torch.Tensor:
        features_surg = phase_features.clone()
        features_surg[:, target_phase] = reference_dist
        
        if self.variant.mechanism == "no_structural_propagation":
            # Replace distribution g, but do not propagate effect through network
            # To do this, we just extract features and replace after backbone.
            # But the backbone is convolutional so it mixes across phases.
            # To simulate no-propagation, we forward both, and splice the replaced phase output directly.
            out_base = self.backbone(phase_features.transpose(1, 2))
            out_surg = self.backbone(features_surg.transpose(1, 2))
            # Just mix the mean states
            mixed = out_base.mean(dim=-1) * 0.9 + out_surg.mean(dim=-1) * 0.1
            return self.head(mixed)
            
        return self.forward(features_surg)


# ============================================================================
# Paper 14: Invariant Mechanism Discovery with repStat
# ============================================================================

def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD expects 2D tensors [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD feature dimensions must match")

    x2 = x.square().sum(dim=1, keepdim=True)
    y2 = y.square().sum(dim=1, keepdim=True).T
    d2 = x2 + y2 - 2.0 * (x @ y.T)
    return d2.clamp_min(0.0)


def _imq_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    c2: float = 1.0,
) -> torch.Tensor:
    if c2 <= 0:
        raise ValueError("c2 must be positive")
    d2 = _pairwise_sq_dists(x, y)
    return c2 / (c2 + d2)


def biased_imq_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    c2: float = 1.0,
) -> torch.Tensor:
    """Exact biased MMD^2. Includes diagonal kernel entries."""
    if len(x) == 0 or len(y) == 0:
        raise ValueError("cannot compute MMD with an empty environment")

    k_xx = _imq_kernel(x, x, c2=c2)
    k_yy = _imq_kernel(y, y, c2=c2)
    k_xy = _imq_kernel(x, y, c2=c2)

    mmd2 = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

    # Only numerical protection. Analytic biased MMD^2 is nonnegative.
    return mmd2.clamp_min(0.0)


def multi_environment_mmd(
    representation: torch.Tensor,
    environment: torch.Tensor,
    c2: float = 1.0,
) -> torch.Tensor:
    """
    Mean pairwise MMD^2 across all distinct environments.

    representation: [B, D]
    environment:    [B]
    """
    if representation.ndim != 2:
        raise ValueError("representation must have shape [batch, features]")
    if environment.ndim != 1 or len(environment) != len(representation):
        raise ValueError("environment must have one id per sample")

    envs = torch.unique(environment, sorted=True)

    if len(envs) < 2:
        raise RuntimeError(
            "MMD-enabled training requires at least two environments per batch"
        )

    groups = [representation[environment == env] for env in envs]

    if any(len(group) < 2 for group in groups):
        raise RuntimeError(
            "each environment must contribute at least two samples to an MMD batch"
        )

    penalties = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            penalties.append(
                biased_imq_mmd2(groups[i], groups[j], c2=c2)
            )

    return torch.stack(penalties).mean()


def effective_rank(z: torch.Tensor) -> float:
    """Computes participation ratio / effective rank of Cov(Z):
    r_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2 = (Tr(C))^2 / Tr(C^2)
    """
    if z.ndim != 2 or z.shape[0] < 2:
        return 0.0
    centered = z - z.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / (z.shape[0] - 1)
    tr_cov = torch.trace(cov)
    tr_cov2 = torch.trace(cov @ cov)
    if tr_cov2.item() <= 1e-12:
        return 0.0
    return float(((tr_cov.square()) / tr_cov2).item())


class InvariantMechanismDiscoveryModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, width, 1),
            CircularResidualBlock(width),
            CircularResidualBlock(width),
        )
        self.phi = self.encoder
        
        rep_dim = input_dim if self.variant.head == "linear" else width
        self.norm = nn.LayerNorm(rep_dim, elementwise_affine=False)
        self.classifier = nn.Linear(rep_dim, classes, bias=False)
        self.linear_head = self.classifier

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the normalized latent representation Z = Norm(H) that feeds both classifier and MMD."""
        if self.variant.head == "linear":
            if x.ndim == 2:
                h = x
            else:
                h = x.mean(dim=1)
        else:
            if x.ndim == 2:
                x_in = x.unsqueeze(-1)
            else:
                x_in = x.transpose(1, 2)
            h = self.encoder(x_in).mean(dim=-1)
        return self.norm(h)

    def forward_with_representation(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_representation(x)
        return logits

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    @staticmethod
    def compute_mmd_penalty(
        z: torch.Tensor,
        environment: torch.Tensor,
        c2: float = 1.0,
    ) -> torch.Tensor:
        return multi_environment_mmd(z, environment, c2=c2)

    def compute_irm_penalty(self, phase_features: torch.Tensor, labels: torch.Tensor, dummy_w: torch.Tensor) -> torch.Tensor:
        """Retained for legacy/shared compatibility only; not called by Paper 14."""
        z = self.encode(phase_features)
        logits = self.classifier(z) * dummy_w
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        grad = torch.autograd.grad(loss, [dummy_w], create_graph=True)[0]
        return grad.pow(2).sum()


# ============================================================================
# Paper 15: Causal Mechanism Factorization
# ============================================================================

class CausalMechanismFactorizationModel(nn.Module):
    """Paper 15: Modular Phase-Transition Dynamics in ECG Representation Space (P15-V2).
    
    Models the 16 cyclic transitions of the periodic cardiac cycle:
        Z_hat_{(g+1)%16} = Z_g + M_g(Z_g)
    in a fixed coordinate system.
    """
    def __init__(
        self,
        input_dim: int,
        width: int = 64,
        classes: int = 5,
        num_phases: int = 16,
        variant: ExperimentVariant | None = None,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.width = width
        self.num_phases = num_phases
        self.input_dim = input_dim
        
        # Latent projection / state grounding
        self.init_proj = nn.Linear(input_dim, width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        if freeze_encoder:
            for p in self.init_proj.parameters():
                p.requires_grad = False
                
        # 16 Cyclic Transition Mechanisms (g -> (g+1)%16)
        mech_type = self.variant.mechanism
        if mech_type in ("shared_mechanism", "shared_capacity_matched"):
            # Capacity-matched shared: matches total parameters of 16 width->width->width modules
            target_parameters = num_phases * (2 * width * width + 2 * width)
            matched_width = round((target_parameters - width) / (2 * width + 1))
            shared_mech = nn.Sequential(
                nn.Linear(width, matched_width),
                nn.GELU(),
                nn.Linear(matched_width, width),
            )
            self.mechanisms = nn.ModuleList([shared_mech for _ in range(num_phases)])
            self.is_phase_conditioned = False
        elif mech_type == "shared_same_width":
            # Same-width shared: 1 shared width->width->width module
            shared_mech = nn.Sequential(
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.mechanisms = nn.ModuleList([shared_mech for _ in range(num_phases)])
            self.is_phase_conditioned = False
        elif mech_type == "shared_phase_conditioned":
            # Phase-conditioned shared: 1 shared module taking [Z_g, e_g] where e_g is one-hot phase indicator
            shared_mech = nn.Sequential(
                nn.Linear(width + num_phases, width),
                nn.GELU(),
                nn.Linear(width, width),
            )
            self.mechanisms = nn.ModuleList([shared_mech for _ in range(num_phases)])
            self.is_phase_conditioned = True
        elif mech_type == "linear_modular":
            # Phase-specific linear transition maps: A_g Z_g + b_g
            self.mechanisms = nn.ModuleList([
                nn.Linear(width, width) for _ in range(num_phases)
            ])
            self.is_phase_conditioned = False
        else:
            # Default: Modular (16 distinct non-linear MLPs)
            self.mechanisms = nn.ModuleList([
                nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
                for _ in range(num_phases)
            ])
            self.is_phase_conditioned = False
            
        # Diagnostic head (for probing / legacy compatibility)
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Linear(width * num_phases, classes)

    def encode_states(self, phase_features: torch.Tensor) -> torch.Tensor:
        """Ground states Z_g across all 16 observed cardiac phases."""
        if phase_features.ndim == 2:
            phase_features = phase_features.unsqueeze(1).expand(-1, self.num_phases, -1)
        h = F.gelu(self.init_proj(phase_features))
        return self.norm(h)

    def predict_transitions(self, z: torch.Tensor) -> torch.Tensor:
        """Apply transition mechanisms to predict Z_{(g+1)%16} from Z_g across all 16 cyclic transitions.
        
        Args:
            z: Latent states of shape (B, 16, width)
        Returns:
            z_pred: Predicted next states of shape (B, 16, width)
        """
        B, T, D = z.shape
        z_preds = []
        device = z.device
        for g in range(self.num_phases):
            mech = self.mechanisms[g]
            current_zg = z[:, g]
            if getattr(self, "is_phase_conditioned", False):
                e_g = torch.zeros(B, self.num_phases, device=device, dtype=current_zg.dtype)
                e_g[:, g] = 1.0
                inp = torch.cat([current_zg, e_g], dim=-1)
                delta = mech(inp)
            else:
                delta = mech(current_zg)
            next_zg_pred = current_zg + delta
            z_preds.append(next_zg_pred)
        return torch.stack(z_preds, dim=1)

    def evaluate_permuted_transitions(
        self,
        z: torch.Tensor,
        derangement: list[int] | None = None,
    ) -> torch.Tensor:
        """Post-training frozen Kill Test operator.
        Evaluates transition prediction when mechanism assignments are permuted across mismatched phases.
        
        Args:
            z: Latent states of shape (B, 16, width)
            derangement: List of length 16 where derangement[g] != g.
                         Defaults to cyclic half-shift pi(g) = (g + 8) % 16.
        Returns:
            z_pred_perm: Predicted next states of shape (B, 16, width) under permuted mechanisms
        """
        if derangement is None:
            derangement = [(g + self.num_phases // 2) % self.num_phases for g in range(self.num_phases)]
            
        B, T, D = z.shape
        z_preds = []
        device = z.device
        for g in range(self.num_phases):
            perm_g = derangement[g]
            mech = self.mechanisms[perm_g]
            current_zg = z[:, g]
            if getattr(self, "is_phase_conditioned", False):
                e_g = torch.zeros(B, self.num_phases, device=device, dtype=current_zg.dtype)
                e_g[:, perm_g] = 1.0
                inp = torch.cat([current_zg, e_g], dim=-1)
                delta = mech(inp)
            else:
                delta = mech(current_zg)
            next_zg_pred = current_zg + delta
            z_preds.append(next_zg_pred)
        return torch.stack(z_preds, dim=1)

    def compute_transition_loss(self, z: torch.Tensor, z_pred: torch.Tensor) -> torch.Tensor:
        """Transition prediction fidelity loss on 16-phase cyclic ring:
        L_trans = (1/16) * sum_{g=0}^{15} || Z_{(g+1)%16} - Z_pred_g ||^2.
        Target states are detached to prevent degenerative moving target.
        """
        z_target = torch.roll(z, shifts=-1, dims=1).detach()
        return F.mse_loss(z_pred, z_target)

    def extract_representation_hierarchy(
        self,
        phase_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Extracts the 4-tier diagnostic representation hierarchy:
        1. State: Z
        2. Observed Dynamics: Delta = Z_{(g+1)%16} - Z_g
        3. Predicted Transitions: Delta_hat = M_g(Z_g)
        4. Transition Innovations: R = Delta - Delta_hat
        """
        z = self.encode_states(phase_features)
        z_pred = self.predict_transitions(z)
        z_target = torch.roll(z, shifts=-1, dims=1)
        
        delta_obs = z_target - z
        delta_pred = z_pred - z
        innovations = delta_obs - delta_pred
        
        return {
            "state": z,
            "observed_dynamics": delta_obs,
            "predicted_transitions": delta_pred,
            "innovations": innovations,
        }

    @staticmethod
    def evaluate_identity_baseline(z: torch.Tensor) -> float:
        """Identity baseline: Z_hat_{(g+1)%16} = Z_g."""
        z_target = torch.roll(z, shifts=-1, dims=1)
        return F.mse_loss(z, z_target).item()

    @staticmethod
    def evaluate_mean_residual_baseline(z_train: torch.Tensor, z_eval: torch.Tensor) -> float:
        """Mean residual baseline: Z_hat_{(g+1)%16} = Z_g + E_train[Z_{(g+1)%16} - Z_g]."""
        z_train_target = torch.roll(z_train, shifts=-1, dims=1)
        mean_delta = (z_train_target - z_train).mean(dim=0, keepdim=True)
        z_eval_target = torch.roll(z_eval, shifts=-1, dims=1)
        pred = z_eval + mean_delta
        return F.mse_loss(pred, z_eval_target).item()

    @staticmethod
    def evaluate_phase_mean_baseline(z_train: torch.Tensor, z_eval: torch.Tensor) -> float:
        """Phase mean baseline: Z_hat_{(g+1)%16} = mu_{(g+1)%16}^{train}."""
        z_train_target = torch.roll(z_train, shifts=-1, dims=1)
        target_means = z_train_target.mean(dim=0, keepdim=True)
        z_eval_target = torch.roll(z_eval, shifts=-1, dims=1)
        pred = target_means.expand_as(z_eval_target)
        return F.mse_loss(pred, z_eval_target).item()

    def forward_with_transitions(
        self,
        phase_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_states(phase_features)
        z_pred = self.predict_transitions(z)
        if self.variant.head == "linear":
            mean_feats = phase_features.mean(dim=1) if phase_features.ndim == 3 else phase_features
            logits = self.head(mean_feats)
        else:
            logits = self.head(z.flatten(1))
        return logits, z, z_pred

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            mean_feats = phase_features.mean(dim=1) if phase_features.ndim == 3 else phase_features
            return self.head(mean_feats)
        z = self.encode_states(phase_features)
        return self.head(z.flatten(1))

