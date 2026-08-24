import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_awmd(pred, target):
    """
    Computes Adjusted Weighted Matrix Distance (AWMD) as per Paper Eq. 3-4.
    AWMD = ||E[M(X)^1] - E[M(Y)^1]||^2 + ||E[M(X)^2] - E[M(Y)^2]||^2
    M(X)^1 = Mean, M(X)^2 = Covariance Matrix
    """
    # pred, target: [B, C, T]
    B, C, T = pred.shape
    
    # 1. First-order matching (Mean)
    p_mean = pred.mean(dim=2) # [B, C]
    t_mean = target.mean(dim=2) # [B, C]
    first_order_dist = torch.mean((p_mean - t_mean)**2, dim=1) # [B]
    
    # 2. Second-order matching (Covariance)
    p_centered = pred - p_mean.unsqueeze(2)
    t_centered = target - t_mean.unsqueeze(2)
    
    p_corr = torch.bmm(p_centered, p_centered.transpose(1, 2)) / T # [B, C, C]
    t_corr = torch.bmm(t_centered, t_centered.transpose(1, 2)) / T # [B, C, C]
    
    second_order_dist = torch.mean((p_corr - t_corr)**2, dim=(1, 2)) # [B]
    
    return first_order_dist + second_order_dist

class CovarianceLoss(nn.Module):
    """
    Matches the inter-lead correlation matrix (12x12).
    Encodes anatomical relationships between leads.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target: [B, 12, T]
        B, C, T = pred.shape
        
        # Center the signals
        pred_centered = pred - pred.mean(dim=2, keepdim=True)
        target_centered = target - target.mean(dim=2, keepdim=True)
        
        # Normalize to get correlation (standard deviation normalization)
        pred_std = pred_centered.std(dim=2, keepdim=True) + 1e-8
        target_std = target_centered.std(dim=2, keepdim=True) + 1e-8
        
        pred_norm = pred_centered / pred_std
        target_norm = target_centered / target_std
        
        # Compute 12x12 correlation matrices: [B, 12, 12]
        # Correlation_ij = (1/T) * sum_t (norm_it * norm_jt)
        pred_corr = torch.bmm(pred_norm, pred_norm.transpose(1, 2)) / T
        target_corr = torch.bmm(target_norm, target_norm.transpose(1, 2)) / T
        
        return F.mse_loss(pred_corr, target_corr)

class CurvatureLoss(nn.Module):
    """
    Second-order derivative loss to preserve QRS peak sharpness and morphology.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred, target: [B, 12, T]
        # Second derivative (curvature): f''(x) approx f(x+1) - 2f(x) + f(x-1)
        pred_d2 = pred[:, :, 2:] - 2 * pred[:, :, 1:-1] + pred[:, :, :-2]
        target_d2 = target[:, :, 2:] - 2 * target[:, :, 1:-1] + target[:, :, :-2]
        
        return F.mse_loss(pred_d2, target_d2)

class ConditionAwareUncertaintyLoss(nn.Module):
    """
    Homoscedastic uncertainty weighting with condition-specific modulation.
    Equation: L = sum [ (1/(2*sigma_i^2)) * L_i + log(sigma_i) ] * modulation_i
    """
    def __init__(self, num_losses=6, num_conditions=5):
        super().__init__()
        # Base log variances (learned)
        # Order: mse, mmd, deriv, corr, cov, curv
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
        # Condition modulation weights (learned or fixed)
        # Rows: NORM, MI, STTC, CD, HYP
        # Cols: mse, mmd, deriv, corr, cov, curv
        self.condition_weights = nn.Parameter(torch.ones(num_conditions, num_losses))
        self.loss_names = ['mse', 'mmd', 'deriv', 'corr', 'cov', 'curv']

    def forward(self, losses_dict, condition_labels):
        """
        losses_dict: Dict of loss tensors
        condition_labels: [B, 5] one-hot or soft labels
        """
        device = condition_labels.device
        batch_size = condition_labels.shape[0]
        
        # Stack losses into [num_losses] tensor
        # Ensure they are in the correct order
        loss_tensors = torch.stack([losses_dict[name] for name in self.loss_names]).to(device)
        
        # Compute per-sample modulation: [B, num_losses]
        modulation = torch.mm(condition_labels, self.condition_weights)
        
        # Base weights: [num_losses]
        precisions = torch.exp(-self.log_vars)
        
        # Weighted losses for each component: [num_losses]
        # L_i = 0.5 * exp(-log_var_i) * loss_i + 0.5 * log_var_i
        weighted_components = 0.5 * precisions * loss_tensors + 0.5 * self.log_vars
        
        # Apply modulation and sum: dot product per sample, then mean
        # modulation: [B, num_losses], weighted_components: [num_losses]
        total_loss = (modulation * weighted_components.unsqueeze(0)).sum(dim=1).mean()
        
        # For logging
        with torch.no_grad():
            weights = (0.5 * precisions).cpu().numpy()
            weights_dict = {name: weights[i] for i, name in enumerate(self.loss_names)}
            
        return total_loss, weights_dict

    def get_sigmas(self):
        sigmas = torch.exp(0.5 * self.log_vars)
        return {name: sigmas[i].item() for i, name in enumerate(self.loss_names)}
