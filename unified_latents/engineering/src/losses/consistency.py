import torch
import torch.nn as nn

class ConsistencyLoss(nn.Module):
    """
    Consistency Loss: Enforces correlation between Latent Slice i and Ground Truth Lead i.
    
    Hypothesis:
    If Slice i is truly 'disentangled' and responsible for Lead i, its temporal activation 
    pattern should correlate with the target waveform.
    
    Method:
    1. Average pool the Slice [B, C, T] -> [B, T].
    2. Compute Pearson Correlation with Target [B, T].
    3. Loss = 1 - Mean(Correlation).
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, latents, targets):
        """
        Args:
            latents: List of 12 tensors, each [B, C, T] (C=32 usually).
            targets: [B, 12, T]
        """
        assert isinstance(latents, (list, tuple)) and len(latents) == 12, "Expected list/tuple of 12 latent slices"
        assert targets.shape[1] == 12, "Expected 12 target leads"
        
        correlations = []
        
        for i in range(12):
            # Slice i: [B, C, T]
            z_i = latents[i]
            # Target i: [B, T]
            x_i = targets[:, i, :]
            
            # 1. Reduction: Mean over channels? Or L2 norm?
            # Mean is simplest projection. If specialized/sparse, might be weak.
            # But forces "overall activity".
            z_proj = z_i.mean(dim=1) # [B, T]
            
            # 2. Pearson Correlation
            # Center variables
            z_mean = z_proj.mean(dim=1, keepdim=True)
            x_mean = x_i.mean(dim=1, keepdim=True)
            
            z_centered = z_proj - z_mean
            x_centered = x_i - x_mean
            
            # Cosine similarity
            num = (z_centered * x_centered).sum(dim=1)
            den = torch.sqrt((z_centered ** 2).sum(dim=1)) * torch.sqrt((x_centered ** 2).sum(dim=1))
            
            # Clip den to avoid div by zero
            den = torch.clamp(den, min=self.eps)
            
            rho = num / den # [B]
            
            correlations.append(rho.mean())
            
        # Average correlation across all 12 leads
        avg_rho = torch.stack(correlations).mean()
        
        # Loss = 1 - rho (Maximize correlation)
        return 1.0 - avg_rho
