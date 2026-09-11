import torch
import torch.nn as nn
import torch.nn.functional as F

def simple_contrastive_loss(features, positive_views, negative_views, lambda_reg, model):

    features = F.normalize(features, dim=1)         
    positive_views = F.normalize(positive_views, dim=1)  
    negative_views = F.normalize(negative_views, dim=1)  

    neg_pos_difference = negative_views - positive_views 
    

    dot_product = torch.sum(features * neg_pos_difference, dim=1) 
    
  
    loss_term = torch.mean(dot_product)

    reg_term = 0.0
    for param in model.parameters():
        reg_term += torch.norm(param) ** 2
    reg_term *= lambda_reg  
    
    # Total Loss
    loss = loss_term + reg_term
    
    return loss
    
def spectral_loss(features, positive_views, negative_views, lambda_reg, model):


    features = F.normalize(features, dim=1)        
    positive_views = F.normalize(positive_views, dim=1)  
    negative_views = F.normalize(negative_views, dim=1) 

    pos_similarity = torch.sum(features * positive_views, dim=1)  
    pos_term = -2 * torch.mean(pos_similarity) 

    neg_similarity = torch.sum(features * negative_views, dim=1)
    neg_term = torch.mean(neg_similarity ** 2) 

    reg_term = 0.0
    for param in model.parameters():
        reg_term += torch.norm(param) ** 2
    reg_term *= lambda_reg
    
    loss = pos_term + neg_term + reg_term
    
    return loss

class SimCLRLoss(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, z_A, z_B):
        
        z_A = F.normalize(z_A, p=2, dim=1)
        z_B = F.normalize(z_B, p=2, dim=1)

        all_embeddings = torch.cat([z_A, z_B], dim=0)
        similarity_matrix = (all_embeddings @ all_embeddings.T) / self.tau

        n = z_A.shape[0]
        mask = torch.eye(2 * n, device=z_A.device, dtype=torch.bool)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)], dim=0).to(z_A.device)

        return self.cross_entropy_loss(similarity_matrix, labels)

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


def byol_loss_fn(online_pred, target_z):

    online_pred_norm = F.normalize(online_pred, dim=1)
    target_z_norm = F.normalize(target_z, dim=1)
    
    loss = F.mse_loss(online_pred_norm, target_z_norm, reduction='mean')
   
    return 2 * loss