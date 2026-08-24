
import torch
import numpy as np
from scipy.stats import pearsonr

def compute_reconstruction_metrics(x_true, x_hat):
    """
    Compute MAE and MSE.
    Args:
        x_true: [B, 12, T]
        x_hat: [B, 12, T]
    Returns:
        mae: scalar
        mse: scalar
    """
    mae = torch.nn.functional.l1_loss(x_hat, x_true).item()
    mse = torch.nn.functional.mse_loss(x_hat, x_true).item()
    return mae, mse

def compute_lead_correlation(x_true, x_hat):
    """
    Args:
        x_true: [B, 12, T] ground truth
        x_hat:  [B, 12, T] reconstructed
    
    Returns:
        corr_mean: scalar, mean correlation over 12 leads
        corr_per_lead: [12] correlation per lead
    """
    B, L, T = x_true.shape
    corr_per_lead = []
    
    # Flatten across Batch and Time for global lead correlation
    for lead_idx in range(L):
        x_t = x_true[:, lead_idx, :].reshape(-1)
        x_h = x_hat[:, lead_idx, :].reshape(-1)
        
        # Pearson Corr in PyTorch
        vx = x_t - torch.mean(x_t)
        vy = x_h - torch.mean(x_h)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        corr_per_lead.append(rho.item())
    
    corr_mean = np.mean(corr_per_lead)
    return corr_mean, corr_per_lead

def compute_correlation_matrix_rmse(x_true, x_hat):
    """
    Args:
        x_true: [B, 12, T]
        x_hat:  [B, 12, T]
    
    Returns:
        rmse: scalar
        r_true, r_recon: [12, 12] correlation matrices
    """
    B, L, T = x_true.shape
    
    # Flatten to [B*T, 12]
    # Move to CPU for numpy corrcoef
    x_true_flat = x_true.permute(0, 2, 1).reshape(-1, L).detach().cpu().numpy()
    x_hat_flat = x_hat.permute(0, 2, 1).reshape(-1, L).detach().cpu().numpy()
    
    # Compute correlation matrices
    r_true = np.corrcoef(x_true_flat.T)    # [12, 12]
    r_recon = np.corrcoef(x_hat_flat.T)    # [12, 12]
    
    # RMSE
    rmse = np.sqrt(np.mean((r_true - r_recon) ** 2))
    
    return rmse, r_true, r_recon
