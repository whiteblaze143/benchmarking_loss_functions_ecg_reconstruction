#!/usr/bin/env python3
"""
FAM-ECG: Foundation-Agnostic Multi-lead ECG Reconstruction Components.

This file defines the standard "FAM-ECG" modules that can be attached to 
any foundation model backbone (ECG-FM, HuBERT, ECGFounder, etc.) 
to enable unified multi-lead reconstruction.

Components:
1. UniversalSpatialFusionAdapter: Cross-attention fusion of multi-lead embeddings.
2. ResDecoder: High-fidelity residual CNN decoder for morphology synthesis.
3. PhysicsProjection: Constraint-preserving layer for Einthoven/Goldberger laws.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. SPATIAL FUSION ADAPTER
class UniversalSpatialFusionAdapter(nn.Module):
    """
    High-fidelity, backbone-agnostic adapter using Cross-Attention.
    Signal features (Q) attend to active Lead Embeddings (K, V).
    """
    def __init__(self, dim=768, n_heads=8, n_leads=12, dropout=0.1, init_gate=0.1):
        super().__init__()
        self.dim = dim
        self.lead_embeddings = nn.Embedding(n_leads, dim)
        
        # Cross-Attention: Backbone features attend to Lead Signatures
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        
        # Self-Attention: Temporal dependency modeling within fused space
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.tensor([init_gate])) 

    def forward(self, z, lead_indices):
        # z: [B, T, dim]
        device = z.device
        indices = torch.tensor(lead_indices, device=device)
        
        # Retrieve lead embeddings: [N, dim] -> [1, N, dim] -> [B, N, dim]
        leads = self.lead_embeddings(indices).unsqueeze(0).expand(z.size(0), -1, -1)
        
        # Cross-Attention: Query=Signal, Key/Value=Leads
        attn_out, _ = self.cross_attn(z, leads, leads)
        
        # Gated Residual Connection 1 (Spatial Fusion)
        z = self.ln1(z + self.gate * attn_out)
        
        # Temporal Self-Attention refinement
        self_attn_out, _ = self.self_attn(z, z, z)
        z = self.ln3(z + self_attn_out)
        
        # Feed Forward
        z = self.ln2(z + self.mlp(z))
        
        return z

# 2. MORPHOLOGY DECODER (MasonDecoder)
class MasonBlock1d(nn.Module):
    """
    Aligned with Mason et al.'s ConvolutionalBlock:
    - Kernel 17
    - ReLU activation
    - No Normalization
    - Residual connection
    """
    def __init__(self, channels, kernel_size=17, padding=8):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.act2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        return self.act2(out + residual)

class MasonDecoder(nn.Module):
    """
    High-fidelity decoder matching Mason et al. (2024) specifics.
    Optimized for R2 maximization.
    """
    def __init__(self, embed_dim=768, hidden_dim=256, output_len=5000, n_leads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_len = output_len
        
        self.proj = nn.Linear(embed_dim, hidden_dim)
        
        # All residual blocks now operate at full 500Hz resolution
        self.res1 = MasonBlock1d(hidden_dim)
        self.res2 = MasonBlock1d(hidden_dim)
        self.res3 = MasonBlock1d(hidden_dim) # Added depth for 500Hz complexity
        
        self.to_leads = nn.Sequential(
            nn.Conv1d(hidden_dim, n_leads * 2, kernel_size=17, padding=8),
            nn.ReLU(),
            nn.Conv1d(n_leads * 2, n_leads, kernel_size=3, padding=1),
        )

    def forward(self, x, target_len=None):
        # x: [B, T, embed_dim]
        # 1. Project and expand to 500Hz resolution immediately
        x = self.proj(x).transpose(1, 2) # [B, hidden_dim, T]
        
        # 2. Native interpolation to target length (no learned weights)
        L = target_len if target_len is not None else self.output_len
        x = F.interpolate(x, size=L, mode='linear', align_corners=False)
        
        # 3. Mason Blocks at full resolution
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # 4. Final lead synthesis
        x = self.to_leads(x)
        return x

# 3. PHYSICS PROJECTION
class PhysicsProjection(nn.Module):
    """
    Project 12-lead ECG output to the nearest valid ECG on constraint manifold.
    Enforces Einthoven's and Goldberger's laws.
    """
    def __init__(self):
        super().__init__()
        A = torch.tensor([
            [1.0, 0.0],      # I
            [0.0, 1.0],      # II
            [-1.0, 1.0],     # III = II - I
            [-0.5, -0.5],    # aVR = -(I + II)/2
            [1.0, -0.5],     # aVL = I - II/2
            [-0.5, 1.0],     # aVF = II - I/2
        ], dtype=torch.float32)
        
        AtA = A.T @ A
        AtA_inv = torch.linalg.inv(AtA)
        proj_matrix = AtA_inv @ A.T
        
        self.register_buffer('A', A)
        self.register_buffer('proj_matrix', proj_matrix)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 12, T)
        B, L, T = x.shape
        if L != 12: return x
        
        limb_in = x[:, :6, :].permute(0, 2, 1) # (B, T, 6)
        params = limb_in @ self.proj_matrix.T # (B, T, 2)
        limb_proj = (params @ self.A.T).permute(0, 2, 1) # (B, 6, T)
        
        out = x.clone()
        out[:, :6, :] = limb_proj
        return out
