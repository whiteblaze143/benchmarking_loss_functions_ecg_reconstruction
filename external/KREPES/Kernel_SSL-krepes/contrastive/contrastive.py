import torch
import torch.nn as nn
import numpy as np

def contrastive_loss_function(A, K_nm, K_nm_pos, K_nm_neg, n, lambda_reg, K_mm):

    AAt = A @ A.T 
    term_1 = torch.sum(torch.diag(K_nm @ AAt @ K_nm_neg.T)) / n
    term_2 = torch.sum(torch.diag(K_nm @ AAt @ K_nm_pos.T)) / n
    
    constraint_penalty = (torch.sum(torch.diag(A.T @ K_mm @ A)) - A.shape[1]) ** 2
    loss = term_1 - term_2 + lambda_reg * constraint_penalty
    return loss

class SpectralContrastiveLoss(nn.Module):

    def __init__(self, lambda_reg):
        super(SpectralContrastiveLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, A, z_A, z_B, z_A_neg, K_mm):

        n = z_A.shape[0]
        
        pos_term = -2 * torch.sum(z_A * z_B)
        neg_inner_product = torch.sum(z_A * z_A_neg, dim=1)
        neg_term = torch.sum(neg_inner_product ** 2)

        reg_term = self.lambda_reg * torch.trace(torch.matmul(A.T, torch.matmul(K_mm, A)))
        loss = (pos_term + neg_term + reg_term) / n
        
        return loss

class SimCLRLoss(torch.nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, z_A, z_B):
        
        z_A = torch.nn.functional.normalize(z_A, p=2, dim=1)
        z_B = torch.nn.functional.normalize(z_B, p=2, dim=1)

        all_embeddings = torch.cat([z_A, z_B], dim=0)
        similarity_matrix = (all_embeddings @ all_embeddings.T) / self.tau

        n = z_A.shape[0]
        mask = torch.eye(2 * n, device=z_A.device, dtype=torch.bool)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(n)], dim=0).to(z_A.device)

        return self.cross_entropy_loss(similarity_matrix, labels)