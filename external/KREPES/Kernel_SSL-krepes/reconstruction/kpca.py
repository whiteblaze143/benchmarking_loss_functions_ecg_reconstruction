import torch

def kpca_loss(A, K, K_nm, K_mm, n, lambda_reg):
    AAt = A @ A.T
    term_1 = torch.sum(torch.diag(K)) / n
    term_2 = -2 * torch.sum(torch.diag(AAt @ K_nm.T @ K_nm)) / n
    term_3 = torch.sum(torch.diag(AAt @ K_mm @ AAt @ K_nm.T @ K_nm)) / n
    term_4 = lambda_reg * torch.sum(torch.diag(A.T @ K_mm @ A)) / n
    loss = term_1 + term_2 + term_3 + term_4
    return loss