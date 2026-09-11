import torch
import numpy as np

def AE_loss(A, B, encoder_output, X, K_mm, K_D, n, lambda_reg, lambda_cov, k, d):
    reconstruction = K_D @ B
    trace_X = torch.sum(torch.diag(X.T @ X))  
    trace_X_reconstruction = torch.sum(torch.diag(X.T @ reconstruction)) 
    trace_reconstruction = torch.sum(torch.diag(reconstruction.T @ reconstruction)) 
    
    term_1 = (trace_X - 2 * trace_X_reconstruction + trace_reconstruction) / n
    encoder_cov_penalty = sample_covariance_penalty(encoder_output, n, k)
    cp_decoder = torch.sum(torch.diag(B.T @ K_mm @ B)) ** 2
    regularization_term = lambda_reg * cp_decoder + lambda_cov * encoder_cov_penalty
    loss = term_1 + regularization_term

    return loss

def sample_covariance_penalty(encoder_output, n, latent_dim):
    covariance_matrix = (encoder_output.T @ encoder_output) / n  
    identity_matrix = torch.eye(latent_dim, device=encoder_output.device, dtype=encoder_output.dtype)
    covariance_penalty = torch.norm(covariance_matrix - identity_matrix, p="fro") ** 2
    
    return covariance_penalty