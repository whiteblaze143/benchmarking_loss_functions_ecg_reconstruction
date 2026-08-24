import torch
import torch.nn.functional as F
import numpy as np

def calculate_lead_r2(pred: torch.Tensor, target: torch.Tensor):
    """
    Computes R^2 for each lead individually.     
    Args:
        pred, target: (B, C, T)
    """
    # ss_res: (B, C)
    ss_res = torch.sum((pred - target) ** 2, dim=2)
    
    # ss_tot: (B, C)
    target_mean = target.mean(dim=2, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=2)
    
    # R^2 per lead
    lead_r2 = 1.0 - ss_res / ss_tot
    return lead_r2

def mason_batch_r2_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Returns -Mean(Lead R2) across the batch.
    Literal Mason Alignment (Reference: reconstruction_functions.py:110)
    """
    lead_r2 = calculate_lead_r2(pred, target)
    return -lead_r2.mean()

def calculate_global_r2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Computes a single R^2 value across all leads and batch items combined.
    Highly stable against low-variance leads.
    """
    ss_res = torch.sum((pred - target) ** 2)
    target_mean = target.mean(dim=(0, 2), keepdim=True) # Mean across B and T? 
    # Actually, global R2 usually means sum(SSR) / sum(SST)
    ss_tot = torch.sum((target - target.mean(dim=2, keepdim=True))**2) # This is sum of lead-wise SST
    
    return 1.0 - ss_res / (ss_tot + eps)

def calculate_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Computes Pearson correlation per lead, then averages.
    
    Args:
        pred, target: (B, C, T)
    """
    p_mean = pred.mean(dim=-1, keepdim=True)
    t_mean = target.mean(dim=-1, keepdim=True)
    
    p_cent = pred - p_mean
    t_cent = target - t_mean
    
    num = (p_cent * t_cent).sum(dim=-1)
    den = torch.sqrt((p_cent**2).sum(dim=-1) * (t_cent**2).sum(dim=-1) + eps)
    
    rho = num / den
    return rho.mean()

def calculate_prd(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Percentage Root-mean-square Difference"""
    return 100 * torch.sqrt(
        torch.mean((pred - target)**2) / (torch.mean(target**2) + eps)
    )

def calculate_mse(pred: torch.Tensor, target: torch.Tensor):
    """Standard MSE"""
    return F.mse_loss(pred, target)

# --- MASON ET AL. (2024) SCALING UTILITIES ---
# Constants from third_party/ecg_reconstruction/util_functions/general.py
MASON_MIN = -2.5
MASON_AMP = 5.0

def normalize_mason(lead, min_value=MASON_MIN, amplitude=MASON_AMP):
    """
    Literal normalization from load_functions/load_leads.py:178
    Maps original mV signals to [0, 1] range.
    """
    norm_lead = (lead - min_value) / amplitude
    if isinstance(norm_lead, torch.Tensor):
        return torch.clamp(norm_lead, 0.0, 1.0)
    return np.clip(norm_lead, 0.0, 1.0)

def denormalize_mason(lead, min_value=MASON_MIN, amplitude=MASON_AMP):
    """
    Literal denormalization from load_functions/load_leads.py:189
    Maps output back to mV scale for R^2 calculation.
    """
    return lead * amplitude + min_value
