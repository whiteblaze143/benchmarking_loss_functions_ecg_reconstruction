"""
Unified Latents ECG Engineering Track.

This module defines the reconstruction-first engineering branch:
masked sparse leads + FM tokens -> alignment head -> latent -> decoder -> 12-lead ECG.

The Foundation Model backbone is retained as frozen semantic conditioning.
The teacher encoder is retained only for latent supervision and decoder ceiling
checks. No prior, rollout, or stage2 generative path is active in this file.
The default optimized sparse-lead regime is `I, II, V2`, and the default
latent interface is the raw teacher/regressor latent rather than a diffusion-era
scaled boundary latent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint


def ckpt(fn, *args):
    """
    RNG-aware and autocast-consistent checkpoint helper.
    Ensures dropout masks are preserved during recomputation and 
    AMP context is maintained.
    
    Args:
        fn: The function or nn.Module to checkpoint.
        *args: Inputs to the function.
        
    Returns:
        The output of fn(*args) with gradient checkpointing.
    """
    enabled = torch.is_autocast_enabled()
    dtype = torch.get_autocast_gpu_dtype() if enabled else None

    def wrapped(*inputs):
        if enabled:
            with torch.amp.autocast("cuda", dtype=dtype):
                return fn(*inputs)
        return fn(*inputs)

    return checkpoint(
        wrapped,
        *args,
        use_reentrant=False,
        preserve_rng_state=True,   # Critical for dropout reproducibility
    )

# Import robust building blocks from local cNVAE_ECG clone
from src.reconstruction.unified_latents.cnvae_core.neural_operations_1d import BNSwishConv, FactorizedReduce, Conv1D, SE, UpSample


def _safe_group_count(channels: int, preferred: int = 32) -> int:
    """Pick the largest valid GroupNorm group count up to `preferred`."""
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class NVAECell(nn.Module):
    """
    The fundamental building block of the UL architecture, adapted from cNVAE.
    Consists of two BNSwishConv layers with a Squeeze-and-Excitation (SE) block
    and a residual connection scaled by 0.1 for stability in deep networks.
    
    Args:
        in_c (int): Input channel dimension.
        out_c (int): Output channel dimension.
        kernel_size (int): Size of the 1D convolution kernel (default: 17).
        use_se (bool): Whether to include a Squeeze-and-Excitation block.
    """
    def __init__(self, in_c, out_c, kernel_size=17, use_se=True):
        super().__init__()
        # Ensure residual path dimensions match output channels
        self.skip = nn.Identity() if in_c == out_c else Conv1D(in_c, out_c, kernel_size=1)
        self.op1 = BNSwishConv(in_c, out_c, kernel_size, padding=kernel_size//2)
        self.op2 = BNSwishConv(out_c, out_c, kernel_size, padding=kernel_size//2)
        self.use_se = use_se
        if use_se:
            self.se = SE(in_c, out_c)

    def forward(self, x):
        skip = self.skip(x)
        out = self.op1(x)
        out = self.op2(out)
        if self.use_se:
            out = self.se(out)
        # 0.1 skip connection scale is standard in original cNVAE to maintain signal variance
        return skip + 0.1 * out

class ViTBlock1D(nn.Module):
    """
    A standard Transformer-style Vision Transformer (ViT) block for 1D sequences.
    Supports both self-attention and cross-attention (for conditioning).
    
    Args:
        dim (int): The embedding dimension.
        num_heads (int): Number of attention heads (default: 16).
        mlp_ratio (float): Expansion ratio for the Feed-Forward Network (default: 4.0).
    """
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, cross_cond=None):
        """
        Forward pass.
        
        Args:
            x: Input query sequence (B, L, D).
            cross_cond: Optional key/value sequence for cross-attention.
        """
        # x: (B, L, D)
        if cross_cond is not None:
            # Cross attention mode for semantics (conditioning)
            attn_out, _ = self.attn(self.norm1(x), cross_cond, cross_cond, need_weights=False)
        else:
            # Self attention mode
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
            
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class WearECGResidualBlock1D(nn.Module):
    """
    WearECG-style waveform refiner block.
    Uses GroupNorm + SiLU + Conv1d with a residual projection when channel
    dimensions change. This is intentionally local and lightweight; we avoid
    full temporal self-attention at 5000 samples in the refinement branch.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(_safe_group_count(in_channels), in_channels)
        self.conv_1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(_safe_group_count(out_channels), out_channels)
        self.conv_2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residual)


class WearECGResidualRefiner(nn.Module):
    """
    Small waveform-domain residual correction head inspired by WearECG's VAE
    residual blocks. It predicts a full 12-lead correction from the masked
    conditioning tensor after the main decoder has produced a coarse output.
    """
    def __init__(self, in_channels=24, out_channels=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            WearECGResidualBlock1D(128, 128),
            WearECGResidualBlock1D(128, 128),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            WearECGResidualBlock1D(64, 64),
            nn.GroupNorm(_safe_group_count(64), 64),
            nn.SiLU(),
            nn.Conv1d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------------------
# UNIFIED LATENTS ARCHITECTURE: DeepMind Semantic Option 1
# -------------------------------------------------------------------------
class UL_Encoder(nn.Module):
    """
    Deterministic Teacher Encoder for Unified Latents.
    Maps a full 12-lead ECG (5000 samples) to a compact latent manifold (312 tokens).
    Architecture follows the UL paper macro-architecture with NVAECell blocks.
    
    Macro-Architecture:
    [Conv17] -> [Downsample scales: 128, 256, 512, 512] -> [3x NVAECell] -> [Proj 32] -> [Norm]
    
    Args:
        in_channels (int): Typically 12 for full ECG.
        hidden_dim (int): Base channel dimension (default: 128).
        latent_dim (int): Bottleneck dimension per token (default: 32).
    """
    def __init__(self, in_channels=12, hidden_dim=128, latent_dim=32):
        super().__init__()
        # E1: Initial feature extraction
        self.conv_in = Conv1D(in_channels, hidden_dim, kernel_size=17, padding=8)
        
        # 4 scales of downsampling to go from 5000 to 312
        # Scale 1: 5000 -> 2500 (128 channels)
        self.down1 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim),
            NVAECell(hidden_dim, hidden_dim, kernel_size=17) 
        )
        # Scale 2: 2500 -> 1250 (256 channels)
        self.down2 = nn.Sequential(
            FactorizedReduce(hidden_dim, hidden_dim * 2),
            NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17) 
        )
        # Scale 3: 1250 -> 625 (512 channels)
        self.down3 = nn.Sequential(
            FactorizedReduce(hidden_dim * 2, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17) 
        )
        # Scale 4: 625 -> 313 (Cap at 312) (512 channels)
        self.down4 = nn.Sequential(
            FactorizedReduce(hidden_dim * 4, hidden_dim * 4),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17) 
        )
        
        # Processing cells at the lowest resolution (512 channels)
        self.process = nn.Sequential(
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17)
        )
        
        # Final projection to latent dimension
        self.proj = Conv1D(hidden_dim * 4, latent_dim, kernel_size=1, weight_norm=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        """
        Args:
            x: Full 12-lead ECG (B, 12, 5000).
            
        Returns:
            z_clean: Clean teacher latent (B, latent_dim, 312).
        """
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        h = self.process(h)
        z_clean = self.proj(h)
        # Layer norm across the latent channel dimension
        z_clean = self.norm(z_clean.transpose(1, 2)).transpose(1, 2)
        
        # Standardize sequence length to 312 for consistency across inputs
        if z_clean.shape[-1] > 312:
            z_clean = z_clean[:, :, :312]
        return z_clean


class UL_AlignmentRegressor(nn.Module):
    """
    Deterministic Alignment Regressor.
    Guarantees latent inferability from limited leads by directly predicting 
    the teacher's latent manifold from observed leads and FM tokens.
    G(x_cond, e_fm) -> z_obs_hat
    
    Args:
        in_channels (int): Input masking channels (typically 24).
        hidden_dim (int): Base channel dimension (default: 128).
        latent_dim (int): Target latent dimension (default: 32).
        fm_embed_dim (int): Dimension of FM tokens (typically 768).
    """
    def __init__(self, in_channels=24, hidden_dim=128, latent_dim=32, fm_embed_dim=768):
        super().__init__()
        self.conv_in = Conv1D(in_channels, hidden_dim, kernel_size=17, padding=8)
        self.down1 = nn.Sequential(FactorizedReduce(hidden_dim, hidden_dim), NVAECell(hidden_dim, hidden_dim, kernel_size=17))
        self.down2 = nn.Sequential(FactorizedReduce(hidden_dim, hidden_dim * 2), NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17))
        self.down3 = nn.Sequential(FactorizedReduce(hidden_dim * 2, hidden_dim * 4), NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17))
        self.down4 = nn.Sequential(FactorizedReduce(hidden_dim * 4, hidden_dim * 4), NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17))
        
        self.fm_norm = nn.LayerNorm(fm_embed_dim)
        self.fm_proj = nn.Linear(fm_embed_dim, hidden_dim * 4)
        
        self.process = nn.Sequential(
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17),
            NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17)
        )
        self.proj = Conv1D(hidden_dim * 4, latent_dim, kernel_size=1, weight_norm=True)
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, x_obs, e_fm):
        """
        Forward pass.
        
        Args:
            x_obs: Masked observational leads (B, 24, 5000).
            e_fm: FM tokens (B, 312, 768).
            
        Returns:
            z_obs: Deterministically regressed latent (B, latent_dim, 312).
        """
        h = self.conv_in(x_obs)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.down4(h)
        
        if h.shape[-1] > 312:
            h = h[:, :, :312]
            
        # Combine waveform features with pre-trained semantic tokens
        e_fm_mapped = self.fm_proj(self.fm_norm(e_fm)).transpose(1, 2)
        h = h + e_fm_mapped
        
        h = self.process(h)
        z_obs = self.proj(h)
        # Final normalization to ensure the latent stays on the teacher manifold
        return self.norm(z_obs.transpose(1, 2)).transpose(1, 2)

class UL_DeterministicDecoder(nn.Module):
    """
    Deterministic Decoder D_phi(z_0, x_cond, e_fm).
    Predicts a full 12-lead reconstruction (5000 samples) conditioned on 
    the inferenced latent z_0, masked observed leads, and FM tokens.
    
    Architecture: U-ViT inspired hybrid.
    [Encoder Path (NVAECells)] -> [Bottleneck (ViT Blocks)] -> [Decoder Path (UpSample + Merge)]
    
    Args:
        in_channels (int): Masked input channels (typically 24).
        out_channels (int): Target channels (12 for full ECG).
        latent_dim (int): Bottleneck dimension per token (default: 32).
        fm_embed_dim (int): FM token dimension (default: 768).
        hidden_dim (int): Base channel dimension (default: 128).
    """
    def __init__(self, in_channels=24, out_channels=12, latent_dim=32, fm_embed_dim=768, hidden_dim=128):
        super().__init__()
        # Projections for conditioning signals
        self.z_proj = nn.Linear(latent_dim, hidden_dim * 8)
        self.fm_norm = nn.LayerNorm(fm_embed_dim)
        self.fm_proj = nn.Linear(fm_embed_dim, hidden_dim * 8)
        
        self.conv_in = Conv1D(in_channels, hidden_dim, kernel_size=17, padding=8)
        
        # Down Path (Encoder-style feature extraction) - Blocks [128, 256, 512, 512]
        self.down1 = nn.Sequential(FactorizedReduce(hidden_dim, hidden_dim), NVAECell(hidden_dim, hidden_dim, kernel_size=17))
        self.down2 = nn.Sequential(FactorizedReduce(hidden_dim, hidden_dim * 2), NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17))
        self.down3 = nn.Sequential(FactorizedReduce(hidden_dim * 2, hidden_dim * 4), NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17))
        self.down4 = nn.Sequential(FactorizedReduce(hidden_dim * 4, hidden_dim * 4), NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17))
        
        self.to_bot = nn.Linear(hidden_dim * 4, hidden_dim * 8)
        
        # Bottleneck (Transformer) - 8 blocks, 1024 dimension for high-capacity semantic merging
        depth = 8
        self.bot_blocks = nn.ModuleList([ViTBlock1D(hidden_dim * 8) for _ in range(depth)])
        self.bot_cross = nn.ModuleList([ViTBlock1D(hidden_dim * 8) for _ in range(depth)])
        self.from_bot = nn.Linear(hidden_dim * 8, hidden_dim * 4)
        
        self.seq_pos_emb = nn.Parameter(torch.randn(1, 312, hidden_dim * 8) * 0.02)
        
        # Up Path (Decoder path with skip-merging) - Blocks [512, 512, 256, 128]
        self.up4 = UpSample()
        self.merge4 = nn.Sequential(Conv1D(hidden_dim * 8, hidden_dim * 4, kernel_size=1), NVAECell(hidden_dim * 4, hidden_dim * 4, kernel_size=17))
        self.up3 = UpSample()
        self.merge3 = nn.Sequential(Conv1D(hidden_dim * 6, hidden_dim * 2, kernel_size=1), NVAECell(hidden_dim * 2, hidden_dim * 2, kernel_size=17))
        self.up2 = UpSample()
        self.merge2 = nn.Sequential(Conv1D(hidden_dim * 3, hidden_dim, kernel_size=1), NVAECell(hidden_dim, hidden_dim, kernel_size=17))
        self.up1 = UpSample()
        self.merge1 = nn.Sequential(Conv1D(hidden_dim * 2, hidden_dim, kernel_size=1), NVAECell(hidden_dim, hidden_dim, kernel_size=17))
        
        self.conv_out = Conv1D(hidden_dim, out_channels, kernel_size=17, padding=8)
        
    def forward(self, x_obs, z_0, e_fm):
        """
        Synthesizes the 12-lead ECG.
        
        Args:
            x_obs: Masked input signal (B, 24, 5000).
            z_0: Inferenced latent (B, latent_dim, 312).
            e_fm: FM tokens (B, 312, 768).
            
        Returns:
            out: Reconstructed 12-lead ECG (B, 12, 5000).
        """
        z_emb = self.z_proj(z_0.transpose(1, 2)) # (B, 312, 1024)
        e_fm_mapped = self.fm_proj(self.fm_norm(e_fm)) # (B, 312, 1024)
        
        # Down path
        h1 = self.conv_in(x_obs)
        h2 = self.down1(h1)
        h3 = self.down2(h2)
        h4 = self.down3(h3)
        h5 = self.down4(h4)
        if h5.shape[-1] > 312:
            h5 = h5[:, :, :312]
            
        h_bot = h5.transpose(1, 2)
        h_bot = self.to_bot(h_bot)
        
        # Inject position, latent, and FM semantic information
        h_bot = h_bot + z_emb + self.seq_pos_emb
        e_fm_mapped = e_fm_mapped + self.seq_pos_emb
        
        # Bottleneck processing (self + cross attention)
        for block, cross in zip(self.bot_blocks, self.bot_cross):
            h_bot = ckpt(block, h_bot)
            h_bot = ckpt(lambda x, c: cross(x, cross_cond=c), h_bot, e_fm_mapped)
            
        h_bot = self.from_bot(h_bot).transpose(1, 2)
        
        # Up path with hierarchical skip connections for signal detail reconstruction
        u4 = self.up4(h_bot)
        if u4.shape[-1] != h4.shape[-1]: u4 = F.pad(u4, (0, h4.shape[-1] - u4.shape[-1]))
        u4 = self.merge4(torch.cat([u4, h4], dim=1))
        
        u3 = self.up3(u4)
        if u3.shape[-1] != h3.shape[-1]: u3 = F.pad(u3, (0, h3.shape[-1] - u3.shape[-1]))
        u3 = self.merge3(torch.cat([u3, h3], dim=1))
        
        u2 = self.up2(u3)
        if u2.shape[-1] != h2.shape[-1]: u2 = F.pad(u2, (0, h2.shape[-1] - u2.shape[-1]))
        u2 = self.merge2(torch.cat([u2, h2], dim=1))
        
        u1 = self.up1(u2)
        if u1.shape[-1] != h1.shape[-1]: u1 = F.pad(u1, (0, h1.shape[-1] - u1.shape[-1]))
        u1 = self.merge1(torch.cat([u1, h1], dim=1))
        
        # Final reconstruction layer
        return self.conv_out(u1)


class UL_ConditionalBridge(nn.Module):
    """
    Engineering-track conditional reconstructor.

    This variant is explicitly reconstruction-first:
    x_obs + FM tokens -> alignment head -> latent -> decoder -> 12-lead ECG.

    The teacher encoder remains only to provide a latent target and a clean
    decoder ceiling. This branch is recon-only and tuned around the default
    sparse regime `I, II, V2`.
    """
    def __init__(
        self,
        checkpoint_path,
        embed_dim=768,
        freeze_backbone=True,
        target_len=5000,
        latent_dim=32,
        teacher_loss_weight=0.10,
        reg_loss_weight=1.0,
        align_loss_weight=0.5,
        no_align=False,
        enable_skip_branch=True,
        finetune_mode="anchored",
        latent_interface="raw",
    ):
        super().__init__()
        if finetune_mode not in {"anchored", "strict"}:
            raise ValueError("finetune_mode must be 'anchored' or 'strict'.")
        if latent_interface not in {"raw", "scaled", "scaled_noisy_teacher"}:
            raise ValueError("latent_interface must be 'raw', 'scaled', or 'scaled_noisy_teacher'.")
        self.target_len = target_len
        self.latent_dim = latent_dim
        self.use_engineering_losses = True
        self.enable_skip_branch = bool(enable_skip_branch)
        self.finetune_mode = finetune_mode
        self.latent_interface = latent_interface
        
        # Loss Weight Initialization
        self.teacher_loss_weight = float(teacher_loss_weight)
        self.reg_loss_weight = float(reg_loss_weight)
        self.align_loss_weight = float(align_loss_weight)
        self.freeze_teacher_encoder = False
        
        # Load pre-trained ECG-FM Backbone
        import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')
        import os
        ecg_fm_path = os.path.expanduser('~/ecg_fm_integration/fairseq-signals')
        if ecg_fm_path not in sys.path:
            sys.path.append(ecg_fm_path)
            
        from fairseq_signals.models.ecg_transformer import ECGTransformerModel
        from fairseq_signals.utils.checkpoint_utils import load_checkpoint_to_cpu
        from omegaconf import OmegaConf

        state = load_checkpoint_to_cpu(checkpoint_path)
        cfg = state['cfg']['model']
        OmegaConf.set_struct(cfg, False)
        # Saliency flag support for older fairseq-signals checkpoints
        if getattr(cfg, 'saliency', None) is None: cfg.saliency = False
        self.backbone = ECGTransformerModel.build_model(cfg)
        self.backbone.load_state_dict(state['model'], strict=False)
        
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
            self.backbone.eval()
            
        self.encoder = UL_Encoder(in_channels=12, hidden_dim=128, latent_dim=latent_dim)
        if not no_align:
            self.alignment_head = UL_AlignmentRegressor(in_channels=24, hidden_dim=128, latent_dim=latent_dim, fm_embed_dim=embed_dim)
        else:
            self.alignment_head = None
        self.decoder = UL_DeterministicDecoder(in_channels=24, out_channels=12, latent_dim=latent_dim, fm_embed_dim=embed_dim, hidden_dim=128)
        
        # WearECG-style waveform residual refiner (obs -> full residual correction)
        if self.enable_skip_branch:
            self.obs_to_miss = WearECGResidualRefiner(in_channels=24, out_channels=12)
            # Bound the residual branch to a small non-negative correction.
            self.skip_scale_logit = nn.Parameter(torch.tensor(-1.3862944))
        else:
            self.obs_to_miss = None
            self.skip_scale_logit = None

        # Retained only for explicit latent-interface ablations, not the default path.
        # `scaled_noisy_teacher` is kept as a compatibility alias for older runs,
        # but the engineering branch now evaluates it deterministically.
        lambda_z0 = 5.0
        self.alpha_0 = np.sqrt(1 / (1 + np.exp(-lambda_z0)))
        self.sigma_0 = np.sqrt(1 - self.alpha_0**2)
        
        # Morphology losses
        class STFTLoss(nn.Module):
            """Multi-resolution STFT loss for spectral fidelity."""
            def __init__(self, fft_sizes=[64, 128, 256], hop_sizes=[16, 32, 64], win_lengths=[64, 128, 256]):
                super().__init__()
                self.fft_sizes = fft_sizes
                self.hop_sizes = hop_sizes
                self.win_lengths = win_lengths
            
            def forward(self, pred, target, lead_weights=None):
                pred = pred.float()
                target = target.float()
                batch_size, num_leads, signal_len = pred.shape
                loss = pred.new_zeros((batch_size, num_leads))
                for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
                    window = torch.hann_window(win_length).to(pred.device)
                    # Compute STFT for each resolution
                    pred_stft = torch.stft(pred.reshape(-1, signal_len), n_fft=fft_size, hop_length=hop_size, 
                                           win_length=win_length, window=window, return_complex=True, center=True)
                    target_stft = torch.stft(target.reshape(-1, signal_len), n_fft=fft_size, hop_length=hop_size,
                                             win_length=win_length, window=window, return_complex=True, center=True)
                    
                    # Magnitude spectrogram loss
                    pred_mag = torch.abs(pred_stft).reshape(batch_size, num_leads, pred_stft.shape[-2], pred_stft.shape[-1])
                    target_mag = torch.abs(target_stft).reshape(batch_size, num_leads, target_stft.shape[-2], target_stft.shape[-1])
                    loss = loss + torch.abs(pred_mag - target_mag).mean(dim=(2, 3))
                    
                    # Log magnitude loss
                    pred_log = torch.log(pred_mag + 1e-7)
                    target_log = torch.log(target_mag + 1e-7)
                    loss = loss + torch.abs(pred_log - target_log).mean(dim=(2, 3))
                
                loss = loss / len(self.fft_sizes)
                if lead_weights is not None:
                    loss = loss * lead_weights.view(1, -1)
                return loss.mean()

        self.stft_loss_fn = STFTLoss()
        
        # Full lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6.
        if self.use_engineering_losses:
            # Heavily bias product optimization toward hard precordial targets.
            weights = torch.tensor([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.5, 2.0, 2.25, 2.5])
        else:
            weights = torch.ones(12)
        self.register_buffer('full_lead_weights', weights.view(1, 12, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.freeze_teacher_encoder:
            self.encoder.eval()
        return self

    def get_skip_scale(self):
        if self.skip_scale_logit is None:
            return None
        return torch.sigmoid(self.skip_scale_logit) * 0.25

    def _resolve_indices(self, lead_indices):
        """
        Resolves provided lead indices into observational and missing sets.
        Default engineering regime: [0, 1, 7] -> Lead I, II, V2.
        """
        if lead_indices is None:
            obs_indices = [0, 1, 7]
        else:
            # Enforce same pattern across batch for now
            obs_indices = lead_indices[0].tolist()
        miss_indices = [i for i in range(12) if i not in obs_indices]
        return obs_indices, miss_indices

    def _make_condition_inputs(self, x, obs_indices):
        x_masked = torch.zeros_like(x)
        x_masked[:, obs_indices, :] = x[:, obs_indices, :]
        obs_mask = torch.zeros_like(x_masked)
        obs_mask[:, obs_indices, :] = 1.0
        x_cond = torch.cat([x_masked, obs_mask], dim=1)
        return x_masked, x_cond

    def _slice_missing(self, x_full, miss_indices):
        return x_full[:, miss_indices, :]

    def _get_missing_lead_weights(self, miss_indices):
        weights = self.full_lead_weights[:, miss_indices, :]
        return weights

    def extract_condition(self, x, lead_indices=None):
        """
        Returns:
            x_cond: (B, 24, T)
            x_full_target: (B, 12, T)
            e_fm: (B, 312, 768)
            obs_indices, miss_indices
        """
        obs_indices, miss_indices = self._resolve_indices(lead_indices)
        x_masked, x_cond = self._make_condition_inputs(x, obs_indices)
    
        with torch.no_grad():
            self.backbone.eval()
            e_fm = self.backbone.extract_features(x_masked.float(), padding_mask=None)['x']
    
        return x_cond, x, e_fm, obs_indices, miss_indices

    def encode_teacher_latent(self, x):
        if self.freeze_teacher_encoder:
            with torch.no_grad():
                return self.encoder(x)
        return self.encoder(x)

    def _prepare_latent_for_decode(self, z_latent):
        """Map latent tensors onto the decoder interface.

        The active engineering path is deterministic. `scaled_noisy_teacher`
        remains as a compatibility alias for the old scaled interface rather
        than reintroducing stochastic teacher noise into evaluation.
        """
        if self.latent_interface == "raw":
            return z_latent
        if self.latent_interface in {"scaled", "scaled_noisy_teacher"}:
            return self.alpha_0 * z_latent
        return z_latent

    def _compute_corr_loss(self, x_pred, x_true):
        """L1 distance between lead-lead correlation matrices."""
        xp = x_pred.float()
        xp = xp - xp.mean(dim=-1, keepdim=True)
        xp = xp / (xp.std(dim=-1, keepdim=True) + 1e-6)
        xt = x_true.float()
        xt = xt - xt.mean(dim=-1, keepdim=True)
        xt = xt / (xt.std(dim=-1, keepdim=True) + 1e-6)
        corr_p = torch.matmul(xp, xp.transpose(1, 2)) / x_pred.shape[-1]
        corr_t = torch.matmul(xt, xt.transpose(1, 2)) / x_true.shape[-1]
        return F.l1_loss(corr_p, corr_t)

    def _compute_diff_loss(self, y_pred, y_true, weight_tensor=None):
        """Pointwise difference of signal derivatives."""
        dy_pred = y_pred[:, :, 1:] - y_pred[:, :, :-1]
        dy_true = y_true[:, :, 1:] - y_true[:, :, :-1]
        diff = torch.abs(dy_pred - dy_true)
        if weight_tensor is not None:
            diff = diff * weight_tensor
        return diff.mean()

    def _compute_weighted_recon_loss(self, x_pred, x_true, miss_weights):
        """Combined L1 and MSE reconstruction loss on missing leads."""
        l1_diff = torch.abs(x_pred - x_true)
        mse_diff = (x_pred - x_true) ** 2
        loss_l1 = (l1_diff * miss_weights).mean()
        loss_mse = (mse_diff * miss_weights).mean()
        return loss_l1 + loss_mse

    def decode_full(self, x_cond, z_latent, e_fm):
        """
        Deterministic decoder path returning a full 12-lead prediction.
        z_latent should be (B, latent_dim, 312)
        """
        base_pred = self.decoder(x_cond, z_latent, e_fm)
        if self.obs_to_miss is not None:
            skip_pred = self.obs_to_miss(x_cond)
            return base_pred + self.get_skip_scale() * skip_pred
        return base_pred

    def assemble_full_prediction(self, x, x_full_pred_raw, obs_indices, miss_indices):
        x_full_pred = x.clone()
        x_full_pred[:, miss_indices, :] = x_full_pred_raw[:, miss_indices, :].to(x.dtype)
        return x_full_pred

    def stage1_forward(self, x, lead_indices=None):
        """
        Reconstruction-first training objective for the engineering track.

        The deployable path is the regressor -> decoder path. The teacher path
        remains as a weak anchor or a strict diagnostic branch depending on
        finetune_mode. The default engineering path decodes raw teacher/regressor
        latents rather than diffusion-derived boundary-scaled latents.
        """
        device = x.device
    
        # Step 0: Context extraction
        x_cond, x_target, e_fm, obs_indices, miss_indices = self.extract_condition(x, lead_indices)
        x_miss_target = self._slice_missing(x_target, miss_indices)
        miss_weights = self._get_missing_lead_weights(miss_indices)
        z_clean = self.encode_teacher_latent(x)
        
        # Step 1: Deterministic Alignment Path (Z-space Mapping)
        if self.alignment_head is not None:
            z_obs_hat = self.alignment_head(x_cond, e_fm)
            loss_align = F.mse_loss(z_obs_hat, z_clean.detach())
        else:
            z_obs_hat = None
            loss_align = torch.tensor(0.0, device=device)

        # Teacher path: decoder ceiling and weak reconstruction anchor.
        z_0_teacher = self._prepare_latent_for_decode(z_clean)
        x_pred_teacher_full = self.decode_full(x_cond, z_0_teacher, e_fm)
        x_miss_pred_teacher = self._slice_missing(x_pred_teacher_full, miss_indices)
        loss_decoder_teacher = self._compute_weighted_recon_loss(
            x_miss_pred_teacher,
            x_miss_target,
            miss_weights,
        )

        # Regressor path: deployed reconstruction route.
        loss_decoder_reg = torch.tensor(0.0, device=device)
        loss_stft_reg = torch.tensor(0.0, device=device)
        loss_diff_reg = torch.tensor(0.0, device=device)
        loss_corr_reg = torch.tensor(0.0, device=device)
        
        if z_obs_hat is not None:
            z_0_reg = self._prepare_latent_for_decode(z_obs_hat)
            x_pred_reg_full = self.decode_full(x_cond, z_0_reg, e_fm)
            x_miss_pred_reg = self._slice_missing(x_pred_reg_full, miss_indices)
            x_full_pred_reg = self.assemble_full_prediction(x, x_pred_reg_full, obs_indices, miss_indices)
            
            loss_decoder_reg = self._compute_weighted_recon_loss(
                x_miss_pred_reg,
                x_miss_target,
                miss_weights,
            )
            
            if self.use_engineering_losses:
                lead_weights = miss_weights.squeeze(0).squeeze(-1)
                loss_stft_reg = self.stft_loss_fn(x_miss_pred_reg, x_miss_target, lead_weights=lead_weights)
                loss_diff_reg = self._compute_diff_loss(x_miss_pred_reg, x_miss_target, miss_weights)
                loss_corr_reg = self._compute_corr_loss(x_miss_pred_reg, x_miss_target)
        
        total_reg_loss = loss_decoder_reg
        if self.use_engineering_losses:
            total_reg_loss = total_reg_loss + 0.25 * loss_diff_reg + 0.10 * loss_stft_reg + 0.20 * loss_corr_reg
        
        if self.finetune_mode == "strict":
            total_loss = self.reg_loss_weight * total_reg_loss + self.align_loss_weight * loss_align
        else:
            total_loss = (
                self.teacher_loss_weight * loss_decoder_teacher +
                self.reg_loss_weight * total_reg_loss +
                self.align_loss_weight * loss_align
            )
    
        x_full_pred_teacher = self.assemble_full_prediction(x, x_pred_teacher_full, obs_indices, miss_indices)
        if z_obs_hat is not None:
            x_full_pred = x_full_pred_reg
        else:
            x_full_pred = x_full_pred_teacher
    
        return {
            "loss": total_loss,
            "decoder_loss": loss_decoder_reg,
            "teacher_loss": loss_decoder_teacher,
            "align_loss": loss_align,
            "stft_loss": loss_stft_reg,
            "diff_loss": loss_diff_reg,
            "corr_loss": loss_corr_reg,
            "y_target": x,
            "y_pred_teacher": x_full_pred_teacher,
            "y_pred_reg": x_full_pred if z_obs_hat is not None else None,
            "y_pred": x_full_pred,
            "z_teacher": z_clean.detach(),
            "z_regressed": z_obs_hat.detach() if z_obs_hat is not None else None,
        }

    @torch.no_grad()
    def impute_from_teacher(self, x, lead_indices=None):
        """
        Diagnostic Path: Reconstruction from the Ground-Truth teacher latent.
        Verifies if the decoder is fundamentally capable of high-fidelity signal
        synthesis when provided with the 'perfect' compressed representation.
        
        The engineering default decodes the raw teacher latent; scaled variants
        remain explicit latent-interface ablations.
        """
        x_cond, _, e_fm, obs_indices, miss_indices = self.extract_condition(x, lead_indices)
        z_clean = self.encode_teacher_latent(x)
        z_0 = self._prepare_latent_for_decode(z_clean)
        x_pred_full = self.decode_full(x_cond, z_0, e_fm)
        
        x_full_pred = self.assemble_full_prediction(x, x_pred_full, obs_indices, miss_indices)
        return {"y_pred": x_full_pred, "z_clean": z_clean}

    def impute_from_regressor(self, x, lead_indices=None):
        """
        Deterministic Baseline Path: Reconstruction from Regressed latents.
        x_obs -> alignment_head -> decoder.
        
        This is the preferred low-latency/low-compute product path.
        """
        if self.alignment_head is None:
            return {"available": False, "y_pred": None, "z_latent": None}

        x_cond, _, e_fm, obs_indices, miss_indices = self.extract_condition(x, lead_indices)
        z_obs_hat = self.alignment_head(x_cond, e_fm)
    
        z_0_hat = self._prepare_latent_for_decode(z_obs_hat)
    
        x_pred_full = self.decode_full(x_cond, z_0_hat, e_fm)
        x_full_pred = self.assemble_full_prediction(x, x_pred_full, obs_indices, miss_indices)
    
        return {
            "available": True,
            "y_pred": x_full_pred,
            "z_latent": z_obs_hat,
        }

    def forward(self, x, lead_indices=None, mode="stage1", **kwargs):
        """
        Main entrypoint for the reconstruction-first engineering branch.
        """
        if mode != "stage1":
            raise ValueError("Engineering track supports only mode='stage1'.")
        return self.stage1_forward(x, lead_indices=lead_indices, **kwargs)
