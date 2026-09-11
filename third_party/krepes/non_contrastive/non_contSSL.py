import torch
import torch.nn.functional as F

def BT_loss(z_A, z_B, lambda_reg):

    num_samples, num_features = z_A.shape
  
    z_A = z_A - z_A.mean(dim=0, keepdim=True)
    z_B = z_B - z_B.mean(dim=0, keepdim=True)

    z_A = z_A / (torch.norm(z_A, dim=0, keepdim=True) + 1e-6)
    z_B = z_B / (torch.norm(z_B, dim=0, keepdim=True) + 1e-6)
    
    C = (z_A.T @ z_B) / num_samples
    
    diag_loss = torch.sum((torch.diag(C) - 1) ** 2)
    off_diag_mask = ~torch.eye(num_features, device=C.device).bool()
    off_diag_loss = torch.sum(C[off_diag_mask] ** 2)

    loss = diag_loss + lambda_reg * off_diag_loss
    return loss

def VICReg_loss(Z_A, Z_B, n, tau, lambda_inv, mu_var, nu_cov):

    Z_A = Z_A - Z_A.mean(dim=0, keepdim=True)
    Z_B = Z_B - Z_B.mean(dim=0, keepdim=True)

    diff = Z_A - Z_B
    invariance_term = torch.trace(diff.T @ diff) / n

    variance_A = torch.sqrt((torch.diag(Z_A.T @ Z_A) / n).clamp(min=1e-4))
    variance_B = torch.sqrt((torch.diag(Z_B.T @ Z_B) / n).clamp(min=1e-4))
    variance_penalty_A = torch.sum(torch.clamp(tau - variance_A, min=0) ** 2)
    variance_penalty_B = torch.sum(torch.clamp(tau - variance_B, min=0) ** 2)
    variance_term = variance_penalty_A + variance_penalty_B

    cov_A = (Z_A.T @ Z_A) / (n - 1)
    cov_B = (Z_B.T @ Z_B) / (n - 1)

    off_diag_A = cov_A - torch.diag(torch.diag(cov_A))
    off_diag_B = cov_B - torch.diag(torch.diag(cov_B))
    covariance_term = (off_diag_A ** 2).sum() + (off_diag_B ** 2).sum()

    loss = (
        lambda_inv * invariance_term +
        mu_var * variance_term +
        nu_cov * covariance_term
    )

    return loss

def BYOL_loss(p, z):

    p_norm = F.normalize(p, dim=1)
    z_norm = F.normalize(z, dim=1)
    
    return F.mse_loss(p_norm, z_norm)