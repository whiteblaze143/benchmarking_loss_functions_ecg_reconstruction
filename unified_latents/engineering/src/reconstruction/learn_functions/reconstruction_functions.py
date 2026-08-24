import torch
import torch.nn.functional as F
import numpy as np

def calculate_lead_r2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Computes R^2 for each lead individually.
    Uses per-sample mean for SST (mathematically standard R^2).
    Args:
        pred, target: (B, C, T)
    Returns: (B, C) R^2 per lead per sample
    """
    ss_res = torch.sum((pred - target) ** 2, dim=2)
    target_mean = target.mean(dim=2, keepdim=True)
    ss_tot = torch.sum((target - target_mean) ** 2, dim=2)
    lead_r2 = 1.0 - ss_res / (ss_tot + eps)
    return lead_r2


def mason_batch_r2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8, use_batch_mean: bool = True):
    """
    [STRICT PAPER ALIGNMENT + STABILITY] Mason et al. (2024)
    Ref: "Supplementary Note 2 – Training the reconstruction algorithm"
    
    Args:
        pred, target: (B, C, T)
        use_batch_mean: If True (default in Mason CODE), calculates SST relative to the lead mean 
                        across the entire batch. This provides significant numerical stability 
                        for flat leads. If False (described in paper text), uses per-sample mean.
    """
    # Force float32
    pred   = pred.float()
    target = target.float()

    B, C, T = target.shape
    batch_r2 = torch.tensor(0.0, device=target.device, dtype=torch.float32)

    for c in range(C):
        output_lead = pred[:, c, :]    # (B, T)
        target_lead = target[:, c, :]  # (B, T)

        if use_batch_mean:
            # MASON CODE: Global Batch Mean (Scalar)
            # This is how they achieve stable training on raw physiological data.
            reference_mean = target_lead.mean() # Scalar
        else:
            # PAPER TEXT: Per-Sample Mean (B, 1)
            reference_mean = target_lead.mean(dim=1, keepdim=True) # (B, 1)

        # Sum over time (dim=1) -> Result is (B,)
        lead_ssr = torch.sum((output_lead - target_lead) ** 2, dim=1)
        lead_sst = torch.sum((target_lead - reference_mean) ** 2, dim=1)
        
        # Calculate R2 per sample
        # STABILITY FIX (GEOLOGIC): 
        # 1. Clamp SST to min=0.1 to prevent division-by-zero on flat leads.
        # 2. Clamp R2 to min=-100.0. This prevents massive outliers while 
        #    Ensuring initial training gradients are non-zero even when 
        #    reconstruction is poor (R2 < -1.0).
        lead_r2 = 1.0 - lead_ssr / torch.clamp(lead_sst, min=0.1)
        lead_r2 = torch.clamp(lead_r2, min=-100.0)

        batch_r2 += lead_r2.mean()

    # Average over leads
    return -(batch_r2 / C)

def calculate_sample_r2(pred: torch.Tensor, target: torch.Tensor, use_batch_mean: bool = False):
    """
    Calculates R^2 for each sample in the batch, averaged across leads.
    Returns: (B,) tensor of R^2 values.
    """
    pred   = pred.float()
    target = target.float()
    B, C, T = target.shape
    
    sample_r2_accum = torch.zeros(B, device=target.device, dtype=torch.float32)
    
    for c in range(C):
        output_lead = pred[:, c, :]
        target_lead = target[:, c, :]
        
        if use_batch_mean:
            reference_mean = target_lead.mean()
        else:
            reference_mean = target_lead.mean(dim=1, keepdim=True)
            
        ssr = torch.sum((output_lead - target_lead) ** 2, dim=1)
        sst = torch.sum((target_lead - reference_mean) ** 2, dim=1)
        
        lead_r2 = 1.0 - ssr / torch.clamp(sst, min=0.1)
        lead_r2 = torch.clamp(lead_r2, min=-100.0)
        sample_r2_accum += lead_r2
        
    return sample_r2_accum / C

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


def mason_element_r2(pred: torch.Tensor, target: torch.Tensor, lead_num: int = 6):
    """
    Exact replica of Mason et al. element_r2_function for TEST-TIME evaluation.
    Supplementary Note 2, Eq. (2): R² with per-lead sample mean ȳ(i). Can produce NaN when a lead has zero variance (same as Mason).
    (third_party/ecg_reconstruction/training_functions/reconstruction_functions.py:63-77)

    Unlike the batch training loss (which uses global mean across batch+time),
    this evaluates one ECG at a time, so .mean() gives the per-sample mean —
    exactly as written in Mason's Supplementary Eq. (2):
        ȳ(i) = Σ_n y(i,n) / N

    Args:
        pred:   (C, T) or (1, C, T) — reconstructed precordial leads
        target: (C, T) or (1, C, T) — ground-truth precordial leads
    Returns:
        scalar R² averaged across leads (float)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if target.dim() == 3:
        target = target.squeeze(0)

    C, T = target.shape
    element_r2 = 0.0

    for c in range(C):
        output_lead = pred[c, :]    # (T,)
        target_lead = target[c, :]  # (T,)

        lead_ssr = torch.sum((output_lead - target_lead) ** 2)
        lead_sst = torch.sum((target_lead - target_lead.mean()) ** 2)

        lead_r2 = 1.0 - lead_ssr / lead_sst
        element_r2 += lead_r2.item() / C

    return element_r2


def mason_element_mse(pred: torch.Tensor, target: torch.Tensor, lead_num: int = 6):
    """
    Exact replica of Mason et al. element_mse_function for TEST-TIME evaluation.

    Args:
        pred:   (C, T) or (1, C, T) — reconstructed precordial leads
        target: (C, T) or (1, C, T) — ground-truth precordial leads
    Returns:
        scalar MSE averaged across leads (float)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(0)
    if target.dim() == 3:
        target = target.squeeze(0)

    C, T = target.shape
    element_mse = 0.0

    for c in range(C):
        lead_mse = torch.mean((pred[c, :] - target[c, :]) ** 2)
        element_mse += lead_mse.item() / C

    return element_mse

# --- MASON ET AL. (2024) SCALING UTILITIES ---
# Same as third_party/ecg_reconstruction/util_functions/general.py get_value_range()
MASON_MIN = -2.5
MASON_AMP = 5.0

def normalize_mason(lead, min_value=None, amplitude=None):
    """
    Literal normalization from Mason load_functions/load_leads.py:178
    Maps original mV signals to [0, 1] range.
    """
    min_value = MASON_MIN if min_value is None else min_value
    amplitude = MASON_AMP if amplitude is None else amplitude
    norm_lead = (lead - min_value) / amplitude
    if isinstance(norm_lead, torch.Tensor):
        return torch.clamp(norm_lead, 0.0, 1.0)
    return np.clip(norm_lead, 0.0, 1.0)

def denormalize_mason(lead, min_value=None, amplitude=None):
    """
    Literal denormalization from Mason load_functions/load_leads.py:189
    Maps output back to mV scale for R^2 calculation.
    """
    min_value = MASON_MIN if min_value is None else min_value
    amplitude = MASON_AMP if amplitude is None else amplitude
    return lead * amplitude + min_value
