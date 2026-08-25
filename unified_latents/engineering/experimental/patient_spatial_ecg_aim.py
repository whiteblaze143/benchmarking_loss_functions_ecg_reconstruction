#!/usr/bin/env python3
"""Patient-conditioned spatial architecture ECG-AIM for strict 1->12 reconstruction."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

class PatientMetadataEncoder(nn.Module):
    """Encodes standardized patient metadata and missingness indicators into latent representations."""
    def __init__(self, in_dim: int = 8, hidden_dim: int = 256, out_dim: int = 768, num_layers: int = 2):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            curr_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        # m: [B, in_dim]
        return self.net(m)  # [B, out_dim]


class TargetQuerySpatialDecoder(nn.Module):
    """Target-lead queries actively attend to source morphology and communicate inter-lead."""
    def __init__(
        self,
        dim: int = 768,
        heads: int = 8,
        num_layers: int = 2,
        use_self_attn: bool = True,
        gated: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_self_attn = use_self_attn
        self.gated = gated
        
        self.cross_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_layers)
        ])
        self.cross_norms_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.cross_norms_kv = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        
        if use_self_attn:
            self.self_layers = nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_layers)
            ])
            self.self_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        else:
            self.self_layers = self.self_norms = None

        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Linear(4 * dim, dim)
            ) for _ in range(num_layers)
        ])
        
        if gated:
            self.gains = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        else:
            self.gains = None

    def forward(self, q_target: torch.Tensor, h_source: torch.Tensor) -> torch.Tensor:
        # q_target: [B, 12, P=200, D]
        # h_source: [B, P=200, D]
        B, L, P, D = q_target.shape
        
        # Flatten spatial grid into sequence for cross-attention or iterate over layers
        h_src_rep = h_source.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, P, D)
        q = q_target.reshape(B * L, P, D)
        
        for i in range(self.num_layers):
            # 1. Cross-Attention: target queries retrieve from source
            q_norm = self.cross_norms_q[i](q)
            kv_norm = self.cross_norms_kv[i](h_src_rep)
            attn_out, _ = self.cross_layers[i](q_norm, kv_norm, kv_norm, need_weights=False)
            
            if self.gated:
                q = q + self.gains[i] * attn_out
            else:
                q = q + attn_out
                
            # 2. Inter-Lead Communication (Self-Attention across the 12 leads at each patch)
            if self.use_self_attn:
                # Reshape to [B * P, 12, D]
                q_lead = q.reshape(B, L, P, D).permute(0, 2, 1, 3).reshape(B * P, L, D)
                q_lead_norm = self.self_norms[i](q_lead)
                self_out, _ = self.self_layers[i](q_lead_norm, q_lead_norm, q_lead_norm, need_weights=False)
                q_lead = q_lead + self_out
                q = q_lead.reshape(B, P, L, D).permute(0, 2, 1, 3).reshape(B * L, P, D)
                
            # 3. MLP
            q = q + self.mlps[i](q)
            
        return q.reshape(B, L, P, D)


class SpatialGraphDecoder(nn.Module):
    """Message passing graph neural network over anatomical lead connections."""
    def __init__(self, dim: int = 768, num_layers: int = 2, learnable_adj: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.learnable_adj = learnable_adj
        
        # Base anatomical adjacency matrix (12x12)
        # Leads: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
        A = torch.zeros(12, 12)
        # Einthoven / Limb triangle connections
        limb = [0, 1, 2, 3, 4, 5]
        for i in limb:
            for j in limb:
                if i != j: A[i, j] = 0.5
        # Precordial ladder connections (V1-V2-V3-V4-V5-V6)
        precordial = [6, 7, 8, 9, 10, 11]
        for idx, i in enumerate(precordial):
            if idx > 0: A[i, precordial[idx-1]] = 1.0
            if idx < len(precordial) - 1: A[i, precordial[idx+1]] = 1.0
        # Self-loops
        A = A + torch.eye(12)
        # Symmetric degree normalization
        deg = A.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        A_norm = A / deg
        
        if learnable_adj:
            self.adj = nn.Parameter(A_norm)
        else:
            self.register_buffer("adj", A_norm)
            
        self.gcn_weights = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
            for _ in range(num_layers)
        ])

    def forward(self, h: torch.Tensor, patient_mod: Optional[torch.Tensor] = None) -> torch.Tensor:
        # h: [B, 12, P, D]
        B, L, P, D = h.shape
        # Permute to [B, P, 12, D]
        x = h.permute(0, 2, 1, 3)
        adj = F.softmax(self.adj, dim=-1) if self.learnable_adj else self.adj
        
        for i in range(self.num_layers):
            z = self.norms[i](x)
            # Message passing: A * X * W
            msg = torch.matmul(adj, z)  # [B, P, 12, D]
            msg = self.gcn_weights[i](msg)
            x = x + F.gelu(msg)
            x = x + self.mlps[i](x)
            
        return x.permute(0, 2, 1, 3)  # [B, 12, P, D]


class CardiacBasisDecoder(nn.Module):
    """Generates K global latent temporal basis components + lead-specific mixing."""
    def __init__(self, dim: int = 768, num_bases: int = 6, with_residual: bool = True):
        super().__init__()
        self.num_bases = num_bases
        self.with_residual = with_residual
        
        self.basis_proj = nn.Linear(dim, num_bases * dim)
        self.mixing_net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, num_bases)
        )
        if with_residual:
            self.res_proj = nn.Linear(dim, dim)
        else:
            self.res_proj = None

    def forward(self, h_source: torch.Tensor, q_leads: torch.Tensor) -> torch.Tensor:
        # h_source: [B, P, D]
        # q_leads: [B, 12, D] (lead + patient embeddings)
        B, P, D = h_source.shape
        K = self.num_bases
        
        # 1. Extract K global temporal basis tracks: [B, P, K, D]
        bases = self.basis_proj(h_source).reshape(B, P, K, D).permute(0, 2, 1, 3) # [B, K, P, D]
        
        # 2. Extract mixing coefficients per lead: [B, 12, K]
        weights = self.mixing_net(q_leads) # [B, 12, K]
        
        # 3. Mix: [B, 12, P, D]
        # weights: [B, 12, K, 1, 1], bases: [B, 1, K, P, D]
        out = (weights[:, :, :, None, None] * bases[:, None, :, :, :]).sum(dim=2) # [B, 12, P, D]
        
        if self.with_residual:
            out = out + self.res_proj(q_leads)[:, :, None, :] * 0.1
            
        return out


class PatientSpatialECGAIM(nn.Module):
    """Master Patient-Conditioned Spatial ECG-AIM Architecture."""
    def __init__(
        self,
        mode: str = "A00_base_frozen",
        patch_size: int = 25,
        width: int = 768,
        encoder_depth: int = 8,
        decoder_depth: int = 4,
        heads: int = 12,
        meta_dim: int = 8,
        meta_hidden: int = 256,
        predict_delineation: bool = True
    ):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.width = width
        self.num_patches = 5000 // patch_size
        
        # 1. Source Lead Morphology Encoder
        self.patch_proj = nn.Linear(patch_size, width)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, width) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4*width,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_depth)
        
        # 2. Target Lead Embeddings
        self.lead_embed = nn.Parameter(torch.randn(12, width) * 0.02)
        
        # 3. Patient Metadata Encoder
        self.use_metadata = "meta" in mode or "patient" in mode or mode in {
            "A02_meta_age_sex_concat", "A03_meta_all4_concat", "A04_meta_patient_token",
            "A05_meta_lead_embedding", "A06_meta_film_grid", "A07_meta_film_leadwise",
            "A08_meta_target_query", "A09_meta_hyperdecoder", "A10_meta_age_sex_film",
            "A11_meta_all4_bmi_film", "A12_meta_shuffled_control", "D24_graph_patient"
        }
        if self.use_metadata:
            in_dim = 4 if "age_sex" in mode else 9 if "bmi" in mode else meta_dim
            self.meta_encoder = PatientMetadataEncoder(in_dim=in_dim, hidden_dim=meta_hidden, out_dim=width)
        else:
            self.meta_encoder = None
            
        # 4. Mode-Specific Spatial Modules
        self.film_leadwise = None
        self.target_decoder = None
        self.graph_decoder = None
        self.basis_decoder = None
        self.per_lead_heads = None
        
        if "film" in mode:
            if "leadwise" in mode or mode in {"A07_meta_film_leadwise", "A10_meta_age_sex_film", "A11_meta_all4_bmi_film", "A12_meta_shuffled_control"}:
                self.film_leadwise = nn.Linear(width, 12 * 2 * width)
            else:
                self.film_leadwise = nn.Linear(width, 2 * width)
                
        elif "target_query" in mode or mode.startswith("B"):
            self.target_decoder = TargetQuerySpatialDecoder(
                dim=width, heads=heads,
                num_layers=2 if "2x" in mode else 1,
                use_self_attn="selfattn" in mode,
                gated="gated" in mode
            )
            
        elif "graph" in mode or mode.startswith("D"):
            self.graph_decoder = SpatialGraphDecoder(
                dim=width, num_layers=2, learnable_adj="learnable" in mode
            )
            
        elif "basis" in mode or mode.startswith("F"):
            k = 3 if "k3" in mode else 8 if "k8" in mode else 6
            self.basis_decoder = CardiacBasisDecoder(
                dim=width, num_bases=k, with_residual="residual" in mode
            )
            
        elif mode in {"E26_per_lead_heads", "E27_shared_plus_refiner"}:
            self.per_lead_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(width, width // 2, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(width // 2, patch_size, kernel_size=1)
                ) for _ in range(12)
            ])

        # 5. Shared Waveform Synthesis Head
        self.waveform_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, patch_size)
        )
        
        # 6. Delineation Head
        if predict_delineation:
            self.delineation_head = nn.Sequential(
                nn.Conv1d(width, 96, kernel_size=15, padding=7),
                nn.GELU(),
                nn.Conv1d(96, 4 * patch_size, kernel_size=1)
            )
        else:
            self.delineation_head = None

    def forward(
        self,
        x_source: torch.Tensor,
        obs_lead_idx: int = 0,
        metadata: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # x_source: [B, 1, 5000]
        B = x_source.shape[0]
        device = x_source.device
        
        # 1. Patchify source lead: [B, 200, 25] -> [B, 200, width]
        patches = x_source.reshape(B, self.num_patches, self.patch_size)
        h_source = self.patch_proj(patches) + self.pos_embed
        h_source = self.source_encoder(h_source) # [B, 200, width]
        
        # 2. Encode metadata if present
        e_patient = None
        if self.use_metadata and metadata is not None:
            e_patient = self.meta_encoder(metadata) # [B, width]
            
        # 3. Construct spatial representation
        lead_emb = self.lead_embed[None, :, None, :] # [1, 12, 1, width]
        
        if self.basis_decoder is not None:
            q_leads = self.lead_embed[None, :, :].expand(B, -1, -1)
            if e_patient is not None:
                q_leads = q_leads + e_patient[:, None, :]
            h_grid = self.basis_decoder(h_source, q_leads) # [B, 12, 200, width]
            
        elif self.target_decoder is not None:
            q_target = lead_emb.expand(B, -1, self.num_patches, -1) + self.pos_embed[:, None, :, :]
            if e_patient is not None:
                q_target = q_target + e_patient[:, None, None, :]
            h_grid = self.target_decoder(q_target, h_source) # [B, 12, 200, width]
            
        else:
            # Standard additive spatial broadcast
            h_grid = h_source[:, None, :, :] + lead_emb # [B, 12, 200, width]
            
            # Apply FiLM conditioning if active
            if self.film_leadwise is not None and e_patient is not None:
                film_params = self.film_leadwise(e_patient) # [B, 12 * 2 * width] or [B, 2 * width]
                if film_params.shape[-1] == 12 * 2 * self.width:
                    film_params = film_params.reshape(B, 12, 2, self.width)
                    gamma = 1.0 + film_params[:, :, 0, None, :]
                    beta = film_params[:, :, 1, None, :]
                else:
                    gamma = 1.0 + film_params[:, None, 0:1, None, :]
                    beta = film_params[:, None, 1:2, None, :]
                h_grid = gamma * h_grid + beta
                
            # Apply Graph message passing if active
            if self.graph_decoder is not None:
                h_grid = self.graph_decoder(h_grid, e_patient)
                
        # 4. Waveform Synthesis: [B, 12, 200, width] -> [B, 12, 5000]
        synth_patches = self.waveform_head(h_grid) # [B, 12, 200, 25]
        waveform_pred = synth_patches.reshape(B, 12, 5000)
        
        # Apply per-lead refiner if active
        if self.per_lead_heads is not None:
            for l in range(12):
                h_l = h_grid[:, l].permute(0, 2, 1) # [B, width, 200]
                res = self.per_lead_heads[l](h_l).permute(0, 2, 1).reshape(B, 5000)
                if self.mode == "E27_shared_plus_refiner":
                    waveform_pred[:, l] = waveform_pred[:, l] + 0.1 * res
                else:
                    waveform_pred[:, l] = res

        # 5. Delineation Head (optional multi-task output)
        seg_logits = None
        if self.delineation_head is not None:
            # Flatten to [B * 12, width, 200]
            h_flat = h_grid.reshape(B * 12, self.num_patches, self.width).permute(0, 2, 1)
            del_out = self.delineation_head(h_flat) # [B * 12, 4 * 25, 200]
            # Reshape to [B, 12, 4, 5000]
            seg_logits = del_out.reshape(B, 12, 4, self.patch_size, self.num_patches).permute(0, 1, 4, 3, 2).reshape(B, 12, 5000, 4)

        return {
            "reconstruction": waveform_pred,
            "segmentation_logits": seg_logits,
            "h_grid": h_grid
        }
