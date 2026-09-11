import torch
import copy


class Encoder(torch.nn.Module):
    def __init__(self, A, bias):
        super().__init__()
        self.A = A
        self.bias = bias

    def forward(self, K):
        return (K @ self.A) + self.bias

class KEREPESModel(torch.nn.Module):
    def __init__(self, M, k, d, device, loss_f, criterion, num_classes=None, initialization='kaiming'):
        super(KEREPESModel, self).__init__()

        self.d = d
        self.k = k
        self.M = M
        self.loss_f = loss_f
        self.criterion = criterion
        
        if self.loss_f == 'krr':
            if initialization == 'lecun':
                self.A = torch.nn.Parameter(torch.randn(self.M, self.k, device=device) * (1.0 / M**0.5))
            else:
                self.A = torch.nn.Parameter(torch.empty(self.M, self.k, device=device))                    
                torch.nn.init.kaiming_normal_(self.A, mode='fan_in', nonlinearity='linear')
            self.A.requires_grad = True
            self.bias = None 
        else:
            if initialization == 'lecun':
                self.A = torch.nn.Parameter((1.0 / M**0.5) * torch.randn(self.M, self.k, device=device))
            else:
                self.A = torch.nn.Parameter(torch.empty(self.M, self.k, device=device))            
                torch.nn.init.kaiming_normal_(self.A, mode='fan_in', nonlinearity='linear')
            self.A.requires_grad = True
            if self.loss_f == 'ae':
                if initialization == 'lecun':
                    self.B = torch.nn.Parameter((1.0 / M**0.5) * torch.randn(self.M, self.d, device=device))
                else:
                    self.B = torch.nn.Parameter(torch.empty(self.M, self.d, device=device))               
                    torch.nn.init.kaiming_normal_(self.B, mode='fan_in', nonlinearity='linear')
                self.B.requires_grad = True
            else:
                self.B = None
        self.bias = torch.nn.Parameter(torch.full((self.k,), 0.1, device=device))
        self.bias.requires_grad = True


        self.encoder = Encoder(self.A, self.bias)

    def encode(self, K):
        z = (K @ self.A) + self.bias
        return z
    

    def initialize_A_from_eigen(self, K_mm):
        
        ones = torch.ones_like(K_mm) / self.M
        K_mm_centered = K_mm - ones @ K_mm - K_mm @ ones + ones @ K_mm @ ones

        eigenvalues, eigenvectors = torch.linalg.eigh(K_mm_centered)
        top_k_eigenvectors = eigenvectors[:, -self.k:]

        safe_eigenvalues = torch.abs(eigenvalues[-self.k:]) + 1e-6
        A_initial = top_k_eigenvectors / torch.sqrt(safe_eigenvalues)
        
        with torch.no_grad():
            self.A.copy_(A_initial)

    def forward(self, n, K_A, K_B=None, K_mm=None, encoded_X=None, original_X=None, K_nm_D=None, K_bb=None, y_true=None, config=None):

        Z_A = self.encoder(K_A)
        Z_B = self.encoder(K_B) if K_B is not None else None

        if self.loss_f == 'vicreg':
            return self.criterion(Z_A, Z_B, n, config.tau, config.lambda_inv, config.mu_var, config.nu_cov)
        elif self.loss_f == 'bt':
            return self.criterion(Z_A, Z_B, config.lambda_reg)
        elif self.loss_f == 'ae':
            return self.criterion(self.A, self.B, encoded_X, original_X, K_mm, K_nm_D, self.n,
                                    config.lambda_reg, config.lambda_cov, self.k, self.d)
        elif self.loss_f == 'spectral':
            return self.criterion(self.A, K_A, K_B, config.K_nm_neg)
        elif self.loss_f == 'simclr':
            return self.criterion(Z_A, Z_B)
        elif self.loss_f == 'kpca':
            return self.criterion(self.A, K_bb, K_A, K_mm, config.batch_size, config.lambda_reg)
        elif self.loss_f == 'simple':
            return self.criterion(self.A, K_A, K_B, config.K_nm_neg, self.n, config.lambda_reg, K_mm)