import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.reconstruction_functions import (
    mason_batch_r2_loss,
    calculate_lead_r2,
    calculate_pearson,
)
from src.reconstruction.learn_functions.losses_deep import MIDTMelSpectralLoss

class HeteroscedasticFiducialLoss(nn.Module):
    """
    Heteroscedastic Loss with Fiducial Weighting for ECG Reconstruction.
    
    Equation:
        L = sum_{t} w(t) * ( (y_t - mu_t)^2 / (2 * sigma_t^2) + 0.5 * log(sigma_t^2) )
    
    where w(t) is the fiducial weight at time t.
    """
    def __init__(self, fiducial_weight=5.0, baseline_weight=1.0):
        super().__init__()
        self.fiducial_weight = fiducial_weight
        self.baseline_weight = baseline_weight

    def forward(self, pred_mu, pred_logvar, target, fiducial_mask=None):
        """
        Args:
            pred_mu: Predicted mean signal (Batch, Leads, Time)
            pred_logvar: Predicted log-variance (Batch, Leads, Time) - learned uncertainty
            target: Ground truth signal (Batch, Leads, Time)
            fiducial_mask: Binary or soft mask (Batch, 1, Time) where 1 indicates fiducial region
                           If None, assumes uniform weighting (standard heteroscedastic loss).
        """
        # 1. Compute Heteroscedastic Loss Term
        # Precision = 1 / sigma^2 = exp(-logvar)
        precision = torch.exp(-pred_logvar)
        mse_term = (pred_mu - target) ** 2
        
        # NLL = 0.5 * (precision * MSE + logvar)
        # We omit the constant 0.5 * log(2pi)
        nll_elementwise = 0.5 * (precision * mse_term + pred_logvar)

        # 2. Apply Fiducial Weighting
        if fiducial_mask is not None:
            # Broadcast mask if necessary
            weights = torch.ones_like(nll_elementwise) * self.baseline_weight
            # Apply higher weight where mask is active
            weights = weights + (fiducial_mask * (self.fiducial_weight - self.baseline_weight))
            
            weighted_loss = weights * nll_elementwise
            return weighted_loss.mean()
        else:
            return nll_elementwise.mean()


import numpy as np
from scipy.signal import find_peaks

def get_fiducial_mask(ecg_signal: torch.Tensor, sampling_rate: int = 500) -> torch.Tensor:
    """
    Generate a binary mask for fiducial regions (P-QRS-T) using peak detection.
    
    Args:
        ecg_signal: (Batch, Leads, Time) tensor. We use the first lead (usually Lead I or II) for detection.
        sampling_rate: Sampling rate in Hz.
        
    Returns:
        mask: (Batch, 1, Time) binary tensor (1.0 in fiducial regions, 0.0 otherwise).
    """
    batch_size, num_leads, seq_len = ecg_signal.shape
    device = ecg_signal.device
    mask = torch.zeros((batch_size, 1, seq_len), device=device, dtype=torch.float32)
    
    # Move to CPU for scipy processing
    signals_np = ecg_signal.detach().cpu().numpy()
    
    # Define window around R-peak: [R-200ms, R+400ms] covers P-QRS-T roughly
    # 200ms = 0.1s * 500 = 50 samples? No, 0.2 * 500 = 100 samples.
    pre_window = int(0.2 * sampling_rate)
    post_window = int(0.4 * sampling_rate)
    
    for b in range(batch_size):
        # Use the first lead (index 0) for peak detection
        # If 3-lead input, Lead I is index 0.
        sig = signals_np[b, 0, :]
        
        # Simple peak detection for R-waves
        # Height threshold: 0.5 * max (heuristic)
        # Distance: 0.4s * 500 = 200 samples (assuming max HR 150 bpm)
        peaks, _ = find_peaks(sig, height=np.max(sig)*0.5, distance=int(0.4*sampling_rate))
        
        for peak in peaks:
            start = max(0, peak - pre_window)
            end = min(seq_len, peak + post_window)
            mask[b, 0, start:end] = 1.0
            
    return mask


class DerivativeL1Loss(nn.Module):
    """
    Temporal Derivative Penalty (L1).
    Encourages preservation of high-frequency morphological features (QRS sharpness).
    """
    def __init__(self):
        super().__init__()
        derivative_kernel = torch.tensor([-0.5, 0, 0.5], dtype=torch.float32).view(1, 1, 3)
        self.register_buffer('kernel', derivative_kernel)

    def forward(self, pred, target):
        # pred, target: (B, C, T)
        b, c, t = pred.shape
        p_flat = pred.view(b * c, 1, t)
        t_flat = target.view(b * c, 1, t)
        
        dv_p = F.conv1d(F.pad(p_flat, (1, 1), mode='replicate'), self.kernel)
        dv_t = F.conv1d(F.pad(t_flat, (1, 1), mode='replicate'), self.kernel)
        
        return F.l1_loss(dv_p, dv_t)

class PearsonCorrelationLoss(nn.Module):
    """
    Pearson Correlation Loss (1 - rho).
    Enforces scale-invariant morphological topology.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return 1.0 - calculate_pearson(pred, target)

class DistanceCorrelationLoss(nn.Module):
    """
    Distance Correlation (dCor) Loss.
    
    A robust, non-linear dependency measure that vanishes if and only if 
    variables are independent. Unlike Pearson, dCor captures non-Gaussian 
    and non-linear relationships, making it suitable for heavy-tailed 
    ECG distributions (QRS outliers).
    
    Formula: dCor(X, Y) = dCov(X, Y) / sqrt(dCov(X, X) * dCov(Y, Y))
    where dCov is the distance covariance.
    """
    def __init__(self):
        super().__init__()

    def _distance_matrix(self, x):
        # x: (B, D)
        n = x.size(0)
        dist = torch.pow(x, 2).sum(1, keepdim=True) + torch.pow(x, 2).sum(1, keepdim=True).t() - 2 * torch.matmul(x, x.t())
        return torch.sqrt(torch.relu(dist) + 1e-8)

    def _double_center(self, a):
        # a: (B, B)
        return a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()

    def forward(self, pred, target):
        # pred, target: (B, C, T)
        # Flatten leads and time to treat as samples in the batch or multivariate vectors?
        # Typically x, y are (B, D) where D is features.
        x = pred.view(pred.size(0), -1)
        y = target.view(target.size(0), -1)
        
        A = self._double_center(self._distance_matrix(x))
        B = self._double_center(self._distance_matrix(y))
        
        n = A.size(0)
        dcov2_xy = (A * B).sum() / (n * n)
        dcov2_xx = (A * A).sum() / (n * n)
        dcov2_yy = (B * B).sum() / (n * n)
        
        dcor = torch.sqrt(dcov2_xy + 1e-8) / torch.sqrt(torch.sqrt(dcov2_xx * dcov2_yy + 1e-8) + 1e-8)
        return 1 - dcor

class SpearmanCorrelationLoss(nn.Module):
    """
    Differentiable Spearman Correlation Approximation.
    
    True Spearman rank correlation is non-differentiable.
    Here we use "Soft Spearman" by applying a Tanh transformation to the
    inputs before computing Pearson correlation. This squashes outliers
    and creates a "pseudo-rank" space that is robust to scale and outliers,
    similar to rank correlation, but differentiable.
    """
    def __init__(self):
        super().__init__()
        self.pearson = PearsonCorrelationLoss()

    def forward(self, pred, target):
        # Apply Soft Rank Transform (Tanh)
        # Scale inputs to standard range first?
        # Ideally, rank approx: x_soft_rank = sum(sigmoid(x - x_j))
        # But that's O(N^2). Tanh is a good cheap robustifier.
        
        # Normalize to roughly unit range before Tanh to effective use non-linearity
        p_std = pred.std(dim=-1, keepdim=True) + 1e-8
        t_std = target.std(dim=-1, keepdim=True) + 1e-8
        
        p_norm = (pred - pred.mean(dim=-1, keepdim=True)) / p_std
        t_norm = (target - target.mean(dim=-1, keepdim=True)) / t_std
        
        pred_robust = torch.tanh(p_norm)
        target_robust = torch.tanh(t_norm)
        
        return self.pearson(pred_robust, target_robust)

class MultiScaleSpectralLoss(nn.Module):
    """
    DEPRECATED: Use MIDTMelSpectralLoss from src.reconstruction.learn_functions.losses instead.
    Multi-Scale Spectral (MSS) Loss with "Smooth MSS" improvements.
    Ref: Schwär & Müller, "Multi-Scale Spectral Loss Revisited", IEEE SPL 2023.
    
    Features:
    - Flat-Top Windows: Broader mainlobe for smoother frequency gradients.
    - Prime Window Sizes: Reduces grid-alignment artifacts.
    - Log1p Compression: Stable, non-negative magnitude comparison.
    """
    def __init__(self, scales=[127, 257, 521, 1021], alpha=1.0, overlap=0.5):
        super().__init__()
        self.scales = scales
        self.alpha = alpha
        self.overlap = overlap
        
        # Prepare windows
        self.windows = nn.ParameterList()
        for n in scales:
            # Generate Flat-Top window
            # w(n) = a0 - a1*cos(2pi*n/(N-1)) + a2*cos(4pi*n/(N-1)) - ...
            # Standard coefficients (SFT3F)
            a = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
            n_arr = torch.arange(n).float()
            window = (a[0] 
                      - a[1] * torch.cos(2 * torch.pi * n_arr / (n - 1))
                      + a[2] * torch.cos(4 * torch.pi * n_arr / (n - 1))
                      - a[3] * torch.cos(6 * torch.pi * n_arr / (n - 1))
                      + a[4] * torch.cos(8 * torch.pi * n_arr / (n - 1)))
            self.windows.append(nn.Parameter(window, requires_grad=False))

    def forward(self, pred, target):
        # pred, target: (B, C, T)
        b, c, t = pred.shape
        p_flat = pred.view(-1, t)
        t_flat = target.view(-1, t)
        
        total_loss = 0.0
        
        for i, n in enumerate(self.scales):
            hop = int(n * (1 - self.overlap))
            window = self.windows[i]
            
            p_stft = torch.stft(p_flat, n_fft=n, hop_length=hop, 
                                win_length=n, window=window, 
                                return_complex=True, center=True)
            t_stft = torch.stft(t_flat, n_fft=n, hop_length=hop, 
                                win_length=n, window=window, 
                                return_complex=True, center=True)
            
            p_mag = torch.abs(p_stft)
            t_mag = torch.abs(t_stft)
            
            # Spectral Convergence
            sc_loss = torch.norm(t_mag - p_mag, p='fro') / (torch.norm(t_mag, p='fro') + 1e-8)
            
            # Log Magnitude Loss (Log1p for smoothness)
            # IEEE SPL 2023 recommends log(1 + gamma * mag)
            p_log = torch.log1p(p_mag)
            t_log = torch.log1p(t_mag)
            mag_loss = F.l1_loss(p_log, t_log)
            
            total_loss += sc_loss + self.alpha * mag_loss
            
        return total_loss / len(self.scales)


class MorphologyWeightedMSE(nn.Module):
    """
    Principled QRS-Aware Morphology Loss.
    Uses a bandpass-energy-threshold detector (5-15Hz) to localize QRS regions.
    Upweights MSE in these regions to preserve conduction-sensitive fidelity.
    Ref: Gradowski et al. (2025), Pan-Tompkins principles.
    """
    def __init__(self, qrs_boost=3.0, fs=500):
        super().__init__()
        self.qrs_boost = qrs_boost
        self.fs = fs
        
        # FIR Bandpass filter coefficients (approximation for 5-15Hz at 500Hz)
        # Using a simple moving average difference as a surrogate for bandpass
        # High-pass (vaguely 5Hz) and Low-pass (vaguely 15Hz)
        self.lp_size = int(fs / 15) # ~33 taps
        self.hp_size = int(fs / 5)  # ~100 taps
        
    def forward(self, pred, target):
        # target: (B, 12, T)
        b, c, t = target.shape
        device = target.device
        
        # 1. Bandpass Approximation (Difference of Gaussians/Means)
        # 1D Convolution with uniform kernels
        lp_kernel = torch.ones(1, 1, self.lp_size, device=device) / self.lp_size
        hp_kernel = torch.ones(1, 1, self.hp_size, device=device) / self.hp_size
        
        # Average across leads to get a robust global detection signal
        x_mono = target.mean(dim=1, keepdim=True) # (B, 1, T)
        
        lp = F.conv1d(x_mono, lp_kernel, padding=self.lp_size//2)[..., :t]
        hp = F.conv1d(x_mono, hp_kernel, padding=self.hp_size//2)[..., :t]
        
        # Bandpassed signal focuses on QRS transients
        bp = (lp - hp).abs() 
        
        # 2. Energy Envelope (Moving window of 150ms)
        env_size = int(0.150 * self.fs)
        env_kernel = torch.ones(1, 1, env_size, device=device) / env_size
        energy = F.conv1d(bp**2, env_kernel, padding=env_size//2)[..., :t]
        
        # 3. Adaptive Thresholding (Per batch/sample)
        # QRS regions have much higher energy than P/T or Isoelectric
        thr = energy.mean(dim=-1, keepdim=True) * 2.5
        qrs_mask = (energy > thr).float() # (B, 1, T)
        
        # 4. Weighted MSE
        weight = 1.0 + (self.qrs_boost - 1.0) * qrs_mask
        loss = (weight * (pred - target)**2).mean()
        
        return loss


class MahalanobisLoss(nn.Module):
    """
    Mahalanobis Distance Loss.
    Measures the distance between reconstruction and target in the whitened space 
    defined by the covariance of the target batch.
    
    Effectively penalizes errors that violate the correlation structure of the 
    data (e.g., Einthoven's Law constraints between leads).
    """
    def __init__(self, regularize=1e-5):
        super().__init__()
        self.reg = regularize

    def forward(self, pred, target):
        """
        Args:
            pred, target: (B, C, T) float tensors.
            Computes Mahalanobis distance per timepoint, averaged over batch and time.
        """
        # Disable autocast to ensure float32 precision for Covariance/Inverse
        with torch.autocast(device_type='cuda', enabled=False):
            # Force float32 for stability in covariance calculation
            pred = pred.float()
            target = target.float()
            
            b, c, t = pred.shape
            
            # Reshape to (N, C) where N = B*T is number of samples, C is dimensions (leads)
            # We want to use the TARGET covariance to define the metric space.
            target_flat = target.permute(0, 2, 1).reshape(-1, c) # (N, C)
            pred_flat = pred.permute(0, 2, 1).reshape(-1, c)     # (N, C)
            
            diff = pred_flat - target_flat # (N, C)
            
            # Compute Covariance of TARGET
            # Center target
            mu = target_flat.mean(dim=0, keepdim=True)
            target_centered = target_flat - mu
            
            # Covariance Matrix (C, C)
            # Sigma = (1 / (N-1)) * X^T * X
            sigma = (target_centered.T @ target_centered) / (target_centered.shape[0] - 1)
            
            # Inverse Covariance (Precision Matrix)
            # Use pseudo-inverse to handle rank deficiency (Einthoven's Law)
            # and singular matrices.
            precision = torch.linalg.pinv(sigma, hermitian=True)
            
            # Mahalanobis Distance Squared: d^2 = diff @ P @ diff.T
            # Efficiently: sum( (diff @ P) * diff, dim=1 )
            
            left_term = diff @ precision # (N, C)
            dist_sq = (left_term * diff).sum(dim=1) # (N,)
            
            # Return mean squared distance
            return dist_sq.mean()

class UMALoss(nn.Module):
    """
    Uni-Modal Alignment (UMA) Loss from MERL framework (Liu et al., 2024).
    Uses latent dropout augmentation to form positive pairs for contrastive learning.
    """
    def __init__(self, temperature=0.07, dropout_p=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout_p)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features):
        """
        Args:
            features: [B, D] latent embeddings.
        Returns:
            Scalar loss.
        """
        # 1. Generate two views via independent dropout
        z1 = self.dropout(features)
        z2 = self.dropout(features)
        
        # 2. Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 3. Concatenate views: [2B, D]
        features_cat = torch.cat([z1, z2], dim=0)
        
        # 4. Compute similarity matrix: [2B, 2B]
        logits = torch.matmul(features_cat, features_cat.T) / self.temperature
        
        # 5. Mask self-similarity
        batch_size = features.shape[0]
        mask = torch.eye(2 * batch_size, device=features.device).bool()
        logits.masked_fill_(mask, -1e4)
        
        # 6. Labels: For each i in [0, B), positive is i+B
        # For each i in [B, 2B), positive is i-B
        labels = torch.cat([
            torch.arange(batch_size, device=features.device) + batch_size,
            torch.arange(batch_size, device=features.device)
        ], dim=0)
        
        loss = self.criterion(logits, labels)
        return loss

class R2Loss(nn.Module):
    """
    Direct R2 minimization loss.
    L = 1 - R^2 = MSE / Var
    Optimizing for this instead of pure MSE helps handle signals with different variances
    and aligns with Mason et al.'s goal of high R2.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return mason_batch_r2_loss(pred, target)

class MasonR2Loss(nn.Module):
    """
    Strict implementation of R2 loss as used in Mason et al. (2024).
    Calculates R^2 for each lead, then takes the mean.
    Loss is (1 - Mean(R2)).
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, model_output, model_target):
        return mason_batch_r2_loss(model_output, model_target)
