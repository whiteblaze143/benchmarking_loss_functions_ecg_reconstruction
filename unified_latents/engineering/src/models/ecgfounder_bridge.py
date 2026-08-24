#!/usr/bin/env python3
"""
ECGFounder Bridge for 12-Lead Reconstruction

Uses the actual ECGFounder implementation from:
https://github.com/PKUDigitalHealth/ECGFounder

License: MIT
"""

import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt
from pathlib import Path
from src.models.fam_ecg import MasonDecoder, PhysicsProjection

# Add ECGFounder repo to path
FOUNDER_REPO = Path(__file__).parent.parent.parent / "ecg_fm_integration" / "ecgfounder_repo"
sys.path.insert(0, str(FOUNDER_REPO))

# Import their actual model
from net1d import Net1D


class ECGFounderBridge(nn.Module):
    """
    Bridge module that uses ECGFounder backbone for 12-lead ECG reconstruction.
    
    Similar interface to ECGFMBridge for drop-in compatibility.
    """
    def __init__(
        self,
        checkpoint_path: str = "ecg_fm_integration/checkpoints/ecgfounder/12_lead_ECGFounder.pth",
        freeze_encoder: bool = True,
        physics_projection: bool = True,
        use_res_decoder: bool = True,
        target_len: int = 5000
    ):
        super().__init__()
        self.target_len = target_len
        
        # ECGFounder architecture (from their paper)
        # Note: We use return_features=True to get embeddings
        self.encoder = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=150,  # Original task
            use_bn=True,
            use_do=True,
            return_features=True
        )
        
        # The feature dimension before final dense layer is 1024
        self.embed_dim = 1024
        
        # Load checkpoint
        print(f"Loading ECGFounder from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        # Load with strict=False since we may have architecture differences
        missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [WARN] Missing keys: {len(missing)}")
        if unexpected:
            print(f"  [WARN] Unexpected keys: {len(unexpected)}")
        
        print(f"  ✓ Loaded ECGFounder: embed_dim={self.embed_dim}")
        
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
            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Linear(1024, 12)
            )
        
        # Physics projection (FAM-ECG Physics)
        self.physics_projection = physics_projection
        if physics_projection:
            self.calibrator = PhysicsProjection()
        else:
            self.calibrator = None
        
        # Lead embeddings for multi-lead adapter
        self.lead_embeddings = nn.Embedding(12, self.embed_dim)
    
    def _preprocess(self, x):
        """Apply ECGFounder's robust preprocessing."""
        # x is (B, 12, T)
        device = x.device
        x_np = x.cpu().numpy()
        
        # 1. Bandpass filter 0.67-40Hz + Notch 50Hz
        fs = 500
        # Notch
        b_n, a_n = iirnotch(50, 30, fs)
        # Bandpass
        b_b, a_b = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
        
        for i in range(x_np.shape[0]):
            for c in range(x_np.shape[1]):
                sig = x_np[i, c]
                sig = filtfilt(b_n, a_n, sig)
                sig = filtfilt(b_b, a_b, sig)
                
                # Baseline removal (Medfilt)
                kernel_size = int(0.4 * fs) + 1
                if kernel_size % 2 == 0: kernel_size += 1
                baseline = medfilt(sig, kernel_size=kernel_size)
                x_np[i, c] = sig - baseline
        
        # 2. Z-score normalization
        mean = x_np.mean(axis=-1, keepdims=True)
        std = x_np.std(axis=-1, keepdims=True) + 1e-8
        x_np = (x_np - mean) / std
        
        return torch.from_numpy(x_np).to(device).float()

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
        B, N, T = x.shape
        # ECGFounder expects (B, 12, T)
        # Apply robust preprocessing (filtering + z-score)
        # FIXED Mason Align: Normalize ACTIVE leads before padding to prevent Zero-Pad Pollution.
        x = self._preprocess(x)
        
        # If sparse input, expand to 12 leads with zeros
        if N < 12:
            x_full = torch.zeros(B, 12, T, device=x.device, dtype=x.dtype)
            if lead_indices is not None:
                for i, idx in enumerate(lead_indices):
                    x_full[:, idx, :] = x[:, i, :]
            else:
                x_full[:, :N, :] = x
            x = x_full
        
        # We need the temporal features from the last stage.
        # Let's manually run the encoder stages to get 'out' before mean(-1)
        
        # Optimization: Run encoder in no_grad/eval mode if frozen
        context = torch.no_grad() if self.encoder.training == False or not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad()
        
        with context:
             if not any(p.requires_grad for p in self.encoder.parameters()):
                self.encoder.eval()
                
             out = self.encoder.first_activation(self.encoder.first_bn(self.encoder.first_conv(x)))
             for stage in self.encoder.stage_list:
                 out = stage(out)
        
        # Now 'out' has shape (B, 1024, T_reduced)
        features = out.transpose(1, 2)  # (B, T_reduced, 1024)
        
        # Apply adapter if provided
        if adapter is not None and lead_indices is not None:
            features = adapter(features, lead_indices)
        
        # Decode to 12-lead ECG (ResDecoder)
        if self.use_res_decoder:
            recon = self.decoder(features, target_len=T) # (B, 12, T)
        else:
            recon = self.decoder(features)  # (B, T_reduced, 12)
            recon = recon.transpose(1, 2)  # (B, 12, T_reduced)
            # Interpolate if needed
            recon = nn.functional.interpolate(recon, size=T, mode='linear', align_corners=False)
        
        # Physics projection
        if self.physics_projection and self.calibrator is not None:
            recon = self.calibrator(recon)
        
        if return_embedding:
            return recon, features.mean(dim=1)
            
        return recon

    
    def get_embedding(self, x, pool=True):
        """Get embedding for downstream tasks."""
        features = self.encoder(x)  # Already pooled
        return features


if __name__ == "__main__":
    # Quick test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    bridge = ECGFounderBridge(
        checkpoint_path="ecg_fm_integration/checkpoints/ecgfounder/12_lead_ECGFounder.pth",
        freeze_encoder=True
    ).to(device)
    
    x = torch.randn(2, 3, 5000).to(device)  # 3-lead input
    lead_indices = [0, 1, 8]  # Mason triad
    
    with torch.no_grad():
        y = bridge(x, lead_indices=lead_indices)
    
    print(f"Input: {x.shape}, Output: {y.shape}")
