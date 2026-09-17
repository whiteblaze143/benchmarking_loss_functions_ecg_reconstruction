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
    def __init__(self, input_dim: int, proj_dim: int = 16, lag: int = 4, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.lag = lag
        self.proj = nn.Linear(input_dim, proj_dim)
        hankel_dim = proj_dim * lag
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(hankel_dim * hankel_dim, 128),
                nn.GELU(),
                nn.Linear(128, classes)
            )

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
            
        if self.variant.mechanism == "time_shuffle_local":
            idx = torch.randperm(phase_features.shape[1], device=phase_features.device)
            phase_features = phase_features[:, idx, :]
            
        x = self.proj(phase_features)
        B, T, D = x.shape
        if self.lag == 1:
            M = T
            H1 = x[:, :-1, :].transpose(1, 2)
            H2 = x[:, 1:, :].transpose(1, 2)
        else:
            M = T - self.lag + 1
            hankel_blocks = []
            for i in range(self.lag):
                hankel_blocks.append(x[:, i : i + M, :].transpose(1, 2))
            H = torch.cat(hankel_blocks, dim=1)
            H1 = H[:, :, :-1]
            H2 = H[:, :, 1:]
        H1T = H1.transpose(1, 2)
        C11 = torch.bmm(H1, H1T).float()
        C21 = torch.bmm(H2, H1T).float()
        scale = C11.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
        identity = torch.eye(C11.shape[1], device=x.device, dtype=torch.float32).unsqueeze(0)
        C11_reg = C11 + (1e-4 * scale)[:, None, None] * identity
        
        # torch.linalg.solve doesn't support bfloat16 on CUDA
        A = torch.linalg.solve(C11_reg, C21).to(x.dtype)
        
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
        return self.linear_probe(descriptor) if self.variant.head == "linear" else self.network(descriptor)


# ============================================================================
# Paper 06: Conditional RepStat
# ============================================================================

class ConditionalRepStatModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 32, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, width, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        if self.variant.head == "linear":
            self.linear_probe = nn.Linear(240, classes)
        else:
            self.head = nn.Linear(width, classes)

    def forward(self, recurrence: torch.Tensor) -> torch.Tensor:
        if recurrence.ndim != 4 or recurrence.shape[1:] != (2, 16, 16):
            raise ValueError("Paper 6 expects (batch,2,16,16) recurrence tensors")
        if self.variant.head == "linear":
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
# Paper 08: Local Token Cross-Attention
# ============================================================================

class LocalTokenCrossAttention(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("codebook", torch.zeros(64, input_dim), persistent=True)
        self.register_buffer("codebook_ready", torch.tensor(False), persistent=True)
        
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Linear(d_model, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
            
        if self.variant.mechanism == "kmeans_dictionary":
            if not bool(self.codebook_ready):
                raise RuntimeError("kmeans token variant requires a fitted train-only codebook")
            distances = torch.cdist(phase_features.float(), self.codebook.float())
            phase_features = self.codebook[distances.argmin(dim=-1)].to(phase_features.dtype)
        B = phase_features.shape[0]
        x = self.input_proj(phase_features) + self.pos_embedding
        
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        out = self.transformer(tokens)
        return self.head(out[:, 0])

    def set_codebook(self, centers: torch.Tensor) -> None:
        values = torch.as_tensor(centers, dtype=self.codebook.dtype, device=self.codebook.device)
        if values.shape != self.codebook.shape or not torch.isfinite(values).all():
            raise ValueError(f"codebook must be finite with shape {tuple(self.codebook.shape)}")
        self.codebook.copy_(values)
        self.codebook_ready.fill_(True)


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
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.backbone = nn.Sequential(nn.Conv1d(input_dim, width, 1), CircularResidualBlock(width), CircularResidualBlock(width))
        self.zs_proj = nn.Linear(width, width // 2)
        self.za_proj = nn.Linear(width, width // 2)
        
        if self.variant.head == "linear":
            self.diag_head = nn.Linear(input_dim, classes)
        else:
            self.diag_head = nn.Linear(width // 2, classes)
        self.acq_head = nn.Linear(width // 2, 9)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.diag_head(phase_features.mean(dim=1))
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

class InvariantMechanismDiscoveryModel(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.phi = nn.Sequential(nn.Conv1d(input_dim, width, 1), CircularResidualBlock(width), CircularResidualBlock(width))
        
        if self.variant.head == "linear":
            self.linear_head = nn.Linear(input_dim, classes, bias=False)
        else:
            self.linear_head = nn.Linear(width, classes, bias=False)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.linear_head(phase_features.mean(dim=1))
        x = phase_features.transpose(1, 2)
        rep = self.phi(x).mean(dim=-1)
        return self.linear_head(rep)

    def compute_irm_penalty(self, phase_features: torch.Tensor, labels: torch.Tensor, dummy_w: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, input_dim: int, width: int = 64, classes: int = 5, variant: ExperimentVariant | None = None):
        super().__init__()
        self.variant = variant if variant is not None else ExperimentVariant()
        self.init_proj = nn.Linear(input_dim, width)
        
        if self.variant.mechanism == "shared_mechanism":
            # Match the unique parameter count of fifteen width->width->width
            # mechanisms with one wider mechanism reused at all fifteen phases.
            target_parameters = 15 * (2 * width * width + 2 * width)
            matched_width = round((target_parameters - width) / (2 * width + 1))
            shared_mech = nn.Sequential(
                nn.Linear(width, matched_width), 
                nn.GELU(), 
                nn.Linear(matched_width, width)
            )
            self.mechanisms = nn.ModuleList([shared_mech for _ in range(15)])
        else:
            self.mechanisms = nn.ModuleList([
                nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width)) for _ in range(15)
            ])
            
        if self.variant.head == "linear":
            self.head = nn.Linear(input_dim, classes)
        else:
            self.head = nn.Linear(width * 16, classes)

    def forward(self, phase_features: torch.Tensor) -> torch.Tensor:
        if self.variant.head == "linear":
            return self.head(phase_features.mean(dim=1))
        z0 = F.gelu(self.init_proj(phase_features[:, 0]))
        trajectory = [z0]
        current_z = z0
        for i, mech in enumerate(self.mechanisms):
            next_z = current_z + mech(current_z)
            trajectory.append(next_z)
            current_z = next_z
        traj_tensor = torch.stack(trajectory, dim=1)
        return self.head(traj_tensor.flatten(1))
