#!/usr/bin/env python3
"""
HuBERT-ECG Bridge for 12-Lead Reconstruction

Uses the actual HuBERT-ECG implementation from:
https://github.com/Edoar-do/HuBERT-ECG

License: CC BY-NC 4.0 (Non-Commercial)
"""

import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import torch
import torch.nn as nn
import numpy as np
from scipy.signal import firwin, filtfilt
from pathlib import Path
from transformers import AutoModel
from src.models.fam_ecg import MasonDecoder, PhysicsProjection


class HuBERTBridge(nn.Module):
    """
    Bridge module that uses HuBERT-ECG backbone for 12-lead ECG reconstruction.
    
    Similar interface to ECGFMBridge for drop-in compatibility.
    """
    def __init__(
        self,
        model_name: str = "Edoardo-BS/hubert-ecg-base",  # or small/large
        freeze_encoder: bool = True,
        physics_projection: bool = True,
        use_res_decoder: bool = True,
        target_len: int = 5000
    ):
        super().__init__()
        self.target_len = target_len
        
        # Load HuBERT-ECG via transformers AutoModel
        print(f"Loading HuBERT-ECG from {model_name}...")
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embed_dim = self.encoder.config.hidden_size
        print(f"  ✓ Loaded HuBERT-ECG: embed_dim={self.embed_dim}")
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("  Encoder: FROZEN")
        
        # Reconstruction decoder (FAM-ECG ResDecoder)
        self.use_res_decoder = use_res_decoder
        if use_res_decoder:
            self.decoder = MasonDecoder(
                embed_dim=self.embed_dim,
                hidden_dim=256,
                output_len=target_len,
                n_leads=12
            )
        else:
            self.decoder = nn.Linear(self.embed_dim, 12)
        
        # Physics projection (FAM-ECG Physics)
        self.physics_projection = physics_projection
        if physics_projection:
            self.calibrator = PhysicsProjection()
        else:
            self.calibrator = None
        
        # Lead embeddings for multi-lead adapter
        self.lead_embeddings = nn.Embedding(12, self.embed_dim)
        
        # HuBERT Preprocessing: FIR Filter coefficients
        # Bandpass [0.05, 47] Hz as per paper
        self.fs = 500
        numtaps = int(0.3 * self.fs)
        if numtaps % 2 == 0: numtaps += 1
        self.fir_coeffs = firwin(numtaps, [0.05, 47], pass_zero=False, fs=self.fs)
    
    def _preprocess(self, x_torch):
        """
        HuBERT-specific preprocessing:
        1. FIR Filter [0.05, 47] Hz
        2. Lead-wise Min-Max scaling to [-1, 1]
        """
        x = x_torch.cpu().numpy() # (B, L, T)
        B, L, T = x.shape
        
        # 1. FIR Filtering
        x_filt = np.zeros_like(x)
        for b in range(B):
            for l in range(L):
                # Only filter if not a "dead lead" (all zeros)
                if np.abs(x[b, l]).max() > 1e-6:
                    x_filt[b, l] = filtfilt(self.fir_coeffs, 1.0, x[b, l])
        
        # 2. Lead-wise Min-Max Scaling to [-1, 1]
        # formula: 2 * (x - x_min) / (x_max - x_min + eps) - 1
        x_norm = np.zeros_like(x_filt)
        eps = 1e-8
        for b in range(B):
            for l in range(L):
                x_min = np.min(x_filt[b, l])
                x_max = np.max(x_filt[b, l])
                if (x_max - x_min) > eps:
                    x_norm[b, l] = 2 * (x_filt[b, l] - x_min) / (x_max - x_min + eps) - 1
                else:
                    x_norm[b, l] = 0.0 # Dead lead stays zero
        
        return torch.from_numpy(x_norm).to(x_torch.device).float()
    
    def forward(self, x, lead_indices=None, adapter=None, return_embedding=False):
        """
        Forward pass for 12-lead reconstruction.
        
        Args:
            x: Input ECG tensor (B, N_leads, T) or (B, 12, T)
            lead_indices: List of input lead indices (e.g., [0, 1, 8] for Mason)
            adapter: Optional UniversalSpatialFusionAdapter for multi-lead fusion
            return_embedding: If True, returns tuple (recon, embedding)
        
        Returns:
            Reconstructed 12-lead ECG (B, 12, T)
            OR (recon, embedding) if return_embedding=True
        """
        # HuBERT-ECG expects specific preprocessing aligned with pre-training.
        # 1. FIR Filter and Lead-wise Min-Max to [-1, 1]
        x = self._preprocess(x)
        
        B, N_leads, T = x.shape
        
        # If sparse input, expand to 12 leads with zeros
        if N_leads < 12:
            x_full = torch.zeros(B, 12, T, device=x.device, dtype=x.dtype)
            if lead_indices is not None:
                for i, idx in enumerate(lead_indices):
                    x_full[:, idx, :] = x[:, i, :]
            else:
                x_full[:, :N_leads, :] = x
            x = x_full
        
        B, L, T = x.shape
        x_concatenated = x.reshape(B, L * T)
        
        # Optimization: Run encoder in no_grad/eval mode if frozen to save massive memory
        # This allows 2x-4x larger batch sizes
        context = torch.no_grad() if self.encoder.training == False or not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad()
        
        with context:
            if not any(p.requires_grad for p in self.encoder.parameters()):
                self.encoder.eval() # Force eval mode for frozen encoder (disable dropout)
                
            # Run through encoder
            outputs = self.encoder(x_concatenated, return_dict=True)
            hidden_states = outputs.last_hidden_state  # (B, T_total, embed_dim)
        
        # Squeeze/Reshape out back to leads for spatial/temporal fusion
        T_total = hidden_states.shape[1]
        T_prime = T_total // L
        hidden_states = hidden_states[:, :T_prime * L, :]
        hidden_states = hidden_states.view(B, L, T_prime, self.embed_dim)
        
        # Aggregate across ACTIVE leads only for the adapter
        if lead_indices is not None:
            active_h = hidden_states[:, lead_indices, :, :]
            hidden_states = active_h.mean(dim=1)  # (B, T_prime, embed_dim)
        else:
            hidden_states = hidden_states.mean(dim=1)  # (B, T_prime, embed_dim)
        
        # Apply adapter if provided
        if adapter is not None and lead_indices is not None:
            hidden_states = adapter(hidden_states, lead_indices)

        # Store mean-pooled embedding for UMA/downstream (if requested)
        latent_embedding = hidden_states.mean(dim=1) 
        
        # Decode to 12-lead ECG (ResDecoder)
        if self.use_res_decoder:
            recon = self.decoder(hidden_states, target_len=T) # (B, 12, T)
        else:
            recon = self.decoder(hidden_states)  # (B, T', 12)
            recon = recon.transpose(1, 2)  # (B, 12, T')
            # Interpolate to match target length
            recon = nn.functional.interpolate(recon, size=T, mode='linear', align_corners=False)
        
        # Denormalize output? NO. 
        # We want the model to learn the target amplitude directly.
        # recon = recon * std + mean  <-- This was forcing input stats onto output, causing error.
        
        # Physics projection
        if self.physics_projection and self.calibrator is not None:
            recon = self.calibrator(recon)
        
        if return_embedding:
            return recon, latent_embedding
        return recon

    
    def get_embedding(self, x, pool=True):
        """Get embedding for downstream tasks."""
        outputs = self.encoder(x, return_dict=True)
        hidden_states = outputs.last_hidden_state
        if pool:
            return hidden_states.mean(dim=1)
        return hidden_states


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bridge = HuBERTBridge(
        model_name="Edoardo-BS/hubert-ecg-base",
        freeze_encoder=True,
        physics_projection=False
    ).to(device)
    
    x = torch.randn(2, 3, 2500).to(device)  # 3-lead input
    lead_indices = [0, 1, 8]  # Mason triad
    
    with torch.no_grad():
        y = bridge(x, lead_indices=lead_indices)
    
    print(f"Input: {x.shape}, Output: {y.shape}")
