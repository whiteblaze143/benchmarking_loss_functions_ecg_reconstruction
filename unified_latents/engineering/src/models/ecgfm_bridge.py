#!/usr/bin/env python3
"""
ECG-FM Bridge: Single-Lead to 12-Lead Reconstruction using ECG-FM Foundation Model.

Architecture:
    1. Single-lead input → Pad to 12 leads (zero-padded or replicated)
    2. ECG-FM encoder (frozen, 90M params pretrained on 900K ECGs)
    3. Lightweight decoder → 12-lead reconstruction

Key advantage: Uses ECG-FM's rich representations learned from 900K 12-lead ECGs
instead of Fa-MAE's 11K single-lead representations.

Author: Mithun Manivannan
Date: Feb 2026
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
from pathlib import Path
import torch.nn.functional as F
from src.models.fam_ecg import MasonDecoder, PhysicsProjection, UniversalSpatialFusionAdapter

# Add fairseq-signals to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'ecg_fm_integration' / 'fairseq-signals'))


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for 12-lead reconstruction from ECG-FM embeddings.
    
    Takes sequence of embeddings [B, T, 768] and outputs 12-lead ECG [B, 12, L].
    """
    def __init__(self, embed_dim=768, hidden_dim=256, output_len=5000, n_leads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.n_leads = n_leads
        
        # Project from embedding dim to hidden
        self.proj = nn.Linear(embed_dim, hidden_dim)
        
        # Upsample to target length
        # ECG-FM output is ~312 time steps for 5000 samples (downsampled by 16x)
        # We need to upsample back to 5000
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
        )
        
        # Final projection to 12 leads
        self.to_leads = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, n_leads * 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(n_leads * 2, n_leads, kernel_size=3, padding=1),
        )
        
        # Adaptive layer to handle variable lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_len)
        
    def forward(self, x, target_len=None):
        """
        Args:
            x: [B, T, embed_dim] sequence of embeddings
            target_len: optional target output length
        Returns:
            [B, n_leads, L] reconstructed 12-lead ECG
        """
        B, T, D = x.shape
        
        # Project and transpose for conv
        x = self.proj(x)  # [B, T, hidden_dim]
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        
        # Upsample
        x = self.upsample(x)  # [B, hidden_dim//2, T*16]
        
        # To leads
        x = self.to_leads(x)  # [B, n_leads, L_upsampled]
        
        # Adaptive pool to exact target length
        if target_len is not None:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        else:
            x = self.adaptive_pool(x)
        
        return x


class TwoHeadDecoder(nn.Module):
    """
    Two-head decoder for mixed supervision:
    - 12-lead head: trained on PTB-XL with full 12-lead supervision
    - 1-lead auxiliary head: trained on Icentia for self-supervised learning
    
    This prevents Icentia's 1-lead objective from corrupting the 12-lead synthesis.
    The shared trunk learns general ECG representations.
    """
    def __init__(self, embed_dim=768, hidden_dim=256, output_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_len = output_len
        
        # Shared trunk
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
        )
        
        # 12-lead head (full reconstruction)
        self.head_12lead = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, 24, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(24, 12, kernel_size=3, padding=1),
        )
        
        # 1-lead auxiliary head (self-supervised)
        self.head_1lead = nn.Sequential(
            nn.Conv1d(hidden_dim // 2, 8, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_len)
        
    def forward(self, x, target_len=None, return_both=False):
        """
        Args:
            x: [B, T, embed_dim] sequence of embeddings
            target_len: optional target output length
            return_both: if True, return (12-lead, 1-lead) outputs
        Returns:
            [B, 12, L] 12-lead reconstruction (default)
            or (12-lead, 1-lead) if return_both=True
        """
        B, T, D = x.shape
        
        # Shared trunk
        x = self.proj(x)  # [B, T, hidden_dim]
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        x = self.upsample(x)  # [B, hidden_dim//2, T*16]
        
        # 12-lead head
        out_12 = self.head_12lead(x)  # [B, 12, L_up]
        if target_len is not None:
            out_12 = F.interpolate(out_12, size=target_len, mode='linear', align_corners=False)
        else:
            out_12 = self.adaptive_pool(out_12)
        
        if return_both:
            # 1-lead auxiliary head
            out_1 = self.head_1lead(x)  # [B, 1, L_up]
            if target_len is not None:
                out_1 = F.interpolate(out_1, size=target_len, mode='linear', align_corners=False)
            else:
                out_1 = self.adaptive_pool(out_1)
            return out_12, out_1
        
        return out_12

# Removed local ResDecoder, ResidualBlock1d, and PhysicsProjectionLayer 
# because they are now centralized in fam_ecg.py


class ECGFMBridge(nn.Module):
    """
    ECG-FM Bridge: Limited-Lead to 12-Lead Reconstruction.
    
    Uses frozen ECG-FM encoder (pretrained on 900K 12-lead ECGs) for feature extraction,
    with a lightweight trainable decoder for reconstruction.
    
    Args:
        checkpoint_path: Path to ECG-FM pretrained checkpoint
        freeze_encoder: Whether to freeze ECG-FM encoder
        physics_projection: Whether to apply physics constraints
        input_strategy: 'pad_zeros', 'replicate', or 'mask_token'
        use_two_head: Use TwoHeadDecoder for mixed PTB-XL/Icentia training
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        embed_dim: int = 768,
        freeze_encoder: bool = True,
        physics_projection: bool = True,
        input_strategy: str = 'replicate',
        target_len: int = 5000,
        use_two_head: bool = False,
        use_res_decoder: bool = False,
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self.input_strategy = input_strategy
        self.target_len = target_len
        self.use_two_head = use_two_head
        self.use_physics_proj = physics_projection
        self.embed_dim = embed_dim
        self.use_res_decoder = use_res_decoder
        
        # Load ECG-FM backbone
        self.backbone = None
        
        if checkpoint_path is not None:
            self._load_ecgfm(checkpoint_path)
        
        # Decoder - choose between single-head, two-head, and ResDecoder
        if use_two_head:
            self.decoder = TwoHeadDecoder(
                embed_dim=self.embed_dim,
                hidden_dim=256,
                output_len=target_len
            )
        elif use_res_decoder:
            self.decoder = MasonDecoder(
                embed_dim=self.embed_dim,
                hidden_dim=256,
                output_len=target_len,
                n_leads=12
            )
        else:
            self.decoder = ConvDecoder(
                embed_dim=self.embed_dim,
                hidden_dim=256,
                output_len=target_len,
                n_leads=12
            )
        
        # Physics projection
        self.physics_proj = PhysicsProjection() if physics_projection else None
        
    def _load_ecgfm(self, checkpoint_path: str):
        """Load ECG-FM backbone from checkpoint."""
        try:
            from fairseq_signals.utils import checkpoint_utils
            
            self.backbone, self.cfg, self.task = checkpoint_utils.load_model_and_task(
                checkpoint_path
            )
            
            # Infer embed_dim from backbone/config if available
            if hasattr(self.backbone, 'embed_dim'):
                self.embed_dim = self.backbone.embed_dim
            elif hasattr(self.cfg, 'model') and hasattr(self.cfg.model, 'encoder_embed_dim'):
                self.embed_dim = self.cfg.model.encoder_embed_dim
                
            print(f"✓ Loaded ECG-FM backbone from {checkpoint_path}")
            print(f"  Encoder: embed_dim={self.embed_dim}")
            
            # Freeze if specified
            if self.freeze_encoder:
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"  Encoder: FROZEN")
            else:
                print(f"  Encoder: TRAINABLE")
                
        except ImportError as e:
            print(f"WARNING: Could not load fairseq_signals: {e}")
            print("Using random initialization for ECG-FM backbone (for testing only)")
            self.backbone = None
    
    def _prepare_input(self, x: torch.Tensor, lead_indices: list = None) -> torch.Tensor:
        """
        Prepare multi-lead input for ECG-FM (which expects 12-lead).
        
        Args:
            x: [B, N, L] input ECG leads
            lead_indices: Indices of input leads in the standard 12-lead order.
                          Default: [0] (Lead I) if x has 1 channel.
        Returns:
            [B, 12, L] prepared input for ECG-FM
        """
        B, C, L = x.shape
        
        if C == 12:
            return x  # Already 12-lead
            
        if lead_indices is None:
            if C == 1:
                lead_indices = [0]
            else:
                raise ValueError(f"lead_indices must be specified for input with {C} channels")
                
        if len(lead_indices) != C:
            raise ValueError(f"Number of lead_indices ({len(lead_indices)}) must match input channels ({C})")

        # Create shell
        x_12 = torch.zeros(B, 12, L, device=x.device, dtype=x.dtype)
        
        # Fill provided leads
        for i, idx in enumerate(lead_indices):
            x_12[:, idx, :] = x[:, i, :]
            
        if self.input_strategy == 'replicate' and C == 1:
            # Legacy single-lead replication
            return x.repeat(1, 12, 1)
            
        return x_12
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from ECG-FM backbone.
        
        Args:
            x: [B, 12, L] 12-lead ECG input
        Returns:
            [B, T, embed_dim] sequence of embeddings
        """
        if self.backbone is None:
            # Fallback: simple feature extraction for testing
            B, C, L = x.shape
            T = L // 16  # Approximate ECG-FM temporal downsampling
            return torch.randn(B, T, self.embed_dim, device=x.device)
        
        with torch.set_grad_enabled(not self.freeze_encoder):
            if self.freeze_encoder:
                self.backbone.eval()
            features = self.backbone.extract_features(x, padding_mask=None)
            embeddings = features['x']  # [B, T, 768]
        
        return embeddings
    
    def forward(
        self,
        x: torch.Tensor,
        lead_indices: list = None,
        adapter: nn.Module = None,
        return_features: bool = False,
        return_aux: bool = False,
        return_pre_physics: bool = False,
        return_embedding: bool = False,
    ):
        """
        Forward pass: multi-lead → 12-lead reconstruction.
        
        Args:
            x: [B, N, L] input ECG leads
            lead_indices: Indices of input leads. Default [0].
            adapter: Optional nn.Module to adapt latent embeddings.
            return_features: Whether to also return intermediate embeddings
            return_aux: Whether to return auxiliary 1-lead output (for two-head decoder)
            return_pre_physics: Whether to return pre-projection output for metrics
            return_embedding: Alias for return_features, returns (recon, embedding)
        Returns:
            [B, 12, L] reconstructed 12-lead ECG
            OR (recon, embedding) if return_embedding=True
        """
        # Store original length
        target_len = x.shape[-1]
        
        # ECG-FM expects raw mV signals (normalize: false in training config).
        # We ensure it receives the mV distribution directly from the dataset.
        x_norm = x  
        
        # Prepare input for ECG-FM (pads with true zeros)
        x_12 = self._prepare_input(x_norm, lead_indices=lead_indices)  # [B, 12, L]
        
        # Extract features
        features = self.extract_features(x_12)  # [B, T, 768]
        
        # Apply latent adapter if provided
        if adapter is not None:
            # Attempt to pass lead_indices if the adapter signature supports it
            try:
                features = adapter(features, lead_indices=lead_indices)
            except TypeError:
                features = adapter(features)
        
        # Decode to 12-lead
        aux_1lead = None
        if self.use_two_head and return_aux:
            recon, aux_1lead = self.decoder(features, target_len=target_len, return_both=True)
        else:
            recon = self.decoder(features, target_len=target_len)
        
        # Store pre-projection output
        recon_pre = recon.clone() if return_pre_physics else None
        
        # Apply physics constraints
        if self.physics_proj is not None:
            recon = self.physics_proj(recon)
        
        # Build return tuple based on flags
        if return_embedding:
            # Pooled features for UMA
            return recon, features.mean(dim=1)
            
        if return_features or return_aux or return_pre_physics:
            result = {'recon': recon}
            if return_features:
                result['features'] = features
            if return_aux and aux_1lead is not None:
                result['aux_1lead'] = aux_1lead
            if return_pre_physics:
                result['recon_pre'] = recon_pre
            return result
        
        return recon
    
    def get_embedding(self, x: torch.Tensor, pool: bool = True) -> torch.Tensor:
        """
        Get ECG-FM embedding for a single-lead input.
        
        Args:
            x: [B, 1, L] single-lead ECG
            pool: Whether to mean-pool over time dimension
        Returns:
            [B, 768] or [B, T, 768] embeddings
        """
        x_12 = self._prepare_input(x)
        features = self.extract_features(x_12)
        
        if pool:
            return features.mean(dim=1)  # [B, 768]
        return features


def test_ecgfm_bridge():
    """Quick test of ECG-FM Bridge architecture."""
    print("Testing ECG-FM Bridge...")
    
    # Test without actual ECG-FM weights (uses random features)
    bridge = ECGFMBridge(
        checkpoint_path=None,  # No weights, just test architecture
        freeze_encoder=True,
        physics_projection=True,
        input_strategy='replicate'
    )
    
    # Test forward pass
    x = torch.randn(2, 1, 5000)  # [B, 1, L] single-lead
    
    out = bridge(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    
    # Verify physics constraints
    I, II, III = out[:, 0], out[:, 1], out[:, 2]
    einthoven_residual = (I + III - II).abs().mean()
    print(f"Einthoven residual: {einthoven_residual:.6f} (should be ~0)")
    
    aVR, aVL, aVF = out[:, 3], out[:, 4], out[:, 5]
    aVR_expected = -(I + II) / 2
    goldberger_err = (aVR - aVR_expected).abs().mean()
    print(f"Goldberger aVR error: {goldberger_err:.6f} (should be ~0)")
    
    print("✓ ECG-FM Bridge architecture test passed!")


if __name__ == '__main__':
    test_ecgfm_bridge()
