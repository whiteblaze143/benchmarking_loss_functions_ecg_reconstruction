import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff

def compute_snr_db(y_true, y_pred, eps=1e-8):
    """
    Signal-to-Noise Ratio in dB.
    Args:
        y_true, y_pred: (..., T)
    """
    noise = y_true - y_pred
    snr = 20 * torch.log10(torch.norm(y_true, dim=-1) / (torch.norm(noise, dim=-1) + eps) + eps)
    return snr.mean().item()

def compute_fourier_distance(y_true, y_pred):
    """
    L1 distance between normalized Mel-spectrograms or FFT magnitudes.
    Huang et al. uses multi-resolution Mel-spectrograms.
    For evaluation, we'll use a simple normalized FFT magnitude L1 distance.
    Args:
        y_true, y_pred: (B, C, T)
    """
    T = y_true.shape[-1]
    fft_true = torch.abs(torch.fft.rfft(y_true, dim=-1))
    fft_pred = torch.abs(torch.fft.rfft(y_pred, dim=-1))
    
    # Normalize per sample+lead to treat as probability distribution
    fft_true_norm = fft_true / (fft_true.sum(dim=-1, keepdim=True) + 1e-8)
    fft_pred_norm = fft_pred / (fft_pred.sum(dim=-1, keepdim=True) + 1e-8)
    
    dist = torch.mean(torch.abs(fft_true_norm - fft_pred_norm))
    return dist.item()

def compute_hausdorff_distance(y_true, y_pred):
    """
    Average Hausdorff distance per lead.
    Args:
        y_true, y_pred: (C, T) numpy arrays
    """
    C, T = y_true.shape
    t = np.linspace(0, 1, T)
    distances = []
    
    for c in range(C):
        # Create 2D point sets (time, value)
        # Note: Scaling matters for Hausdorff. Normalize value to [0,1] or similar.
        u = np.stack([t, y_true[c]], axis=1)
        v = np.stack([t, y_pred[c]], axis=1)
        
        # Directed Hausdorff distance d(u,v)
        d_uv = directed_hausdorff(u, v)[0]
        # Directed Hausdorff distance d(v,u)
        d_vu = directed_hausdorff(v, u)[0]
        
        distances.append(max(d_uv, d_vu))
        
    return np.mean(distances)

def compute_inter_lead_correlation_error(y_true, y_pred):
    """
    Mean Absolute Error between Pearson Correlation Matrices.
    Huang et al. (2025) Table 2.
    Args:
        y_true, y_pred: (C, T) numpy arrays
    """
    corr_true = np.corrcoef(y_true)
    corr_pred = np.corrcoef(y_pred)
    
    # Replace NaNs (from zero variance leads) with 0
    corr_true = np.nan_to_num(corr_true)
    corr_pred = np.nan_to_num(corr_pred)
    
    abs_diff = np.abs(corr_true - corr_pred)
    avg_error = np.mean(abs_diff)
    max_error = np.max(abs_diff)
    
    return avg_error, max_error

def get_all_paper_metrics(y_true_torch, y_pred_torch):
    """
    Helper to compute all metrics for a single sample or batch.
    Args:
        y_true_torch, y_pred_torch: (B, C, T) or (C, T)
    """
    if y_true_torch.dim() == 2:
        y_true_torch = y_true_torch.unsqueeze(0)
    if y_pred_torch.dim() == 2:
        y_pred_torch = y_pred_torch.unsqueeze(0)
        
    y_true_np = y_true_torch.detach().cpu().numpy()
    y_pred_np = y_pred_torch.detach().cpu().numpy()
    
    snr = compute_snr_db(y_true_torch, y_pred_torch)
    fourier = compute_fourier_distance(y_true_torch, y_pred_torch)
    
    # Hausdorff and Corr are currently per-sample in my implementation
    B = y_true_np.shape[0]
    hausdorffs = []
    avg_corrs = []
    max_corrs = []
    
    for b in range(B):
        hausdorffs.append(compute_hausdorff_distance(y_true_np[b], y_pred_np[b]))
        ac, mc = compute_inter_lead_correlation_error(y_true_np[b], y_pred_np[b])
        avg_corrs.append(ac)
        max_corrs.append(mc)
        
    return {
        "snr_db": snr,
        "fourier_dist": fourier,
        "hausdorff_dist": np.mean(hausdorffs),
        "interlead_corr_error_avg": np.mean(avg_corrs),
        "interlead_corr_error_max": np.max(max_corrs)
    }

def compute_faithfulness_score(y_true, y_pred, classifier_fn):
    """
    [MIDT-ECG Pillar 3: Clinical Utility]
    Quantifies diagnostic label consistency between real and synthetic data.
    """
    labels_true = classifier_fn(y_true)
    labels_pred = classifier_fn(y_pred)
    # Cosine similarity or Agreement Ratio
    agreement = (labels_true.argmax(dim=-1) == labels_pred.argmax(dim=-1)).float().mean()
    return agreement.item()

def compute_privacy_risk_mir(model, training_data, synthetic_data):
    """
    [MIDT-ECG Pillar 4: Privacy Preservation]
    Membership Inference Risk stubs to assess memorization vs generalization.
    """
    # Placeholder for MIR/NNAA logic
    return 0.0 # To be implemented via shadow models
