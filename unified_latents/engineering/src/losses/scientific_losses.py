
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAELoss(nn.Module):
    """
    Phase 0: Pure L1 Loss.
    Baseline for reconstruction fidelity.
    Equation: L = (1/N) * \sum |y_true - y_pred|
    Why: Less sensitive to outliers than MSE, produces sharper edges than MSE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class DerivativeLoss(nn.Module):
    """
    Phase 1: First-Order Derivative Loss.
    Equation: L = || (y_true[t+1] - y_true[t]) - (y_pred[t+1] - y_pred[t]) ||_1
    Why: Enforces smoothness and rate-of-change (slope) matching. 
    Crucial for ECG QRS complexes where slope determines conduction velocity.
    """
    def __init__(self, dt=1.0):
        super().__init__()
        self.dt = dt

    def forward(self, pred, target):
        # Calculate diffs along time axis (dim=-1)
        diff_pred = (pred[..., 1:] - pred[..., :-1]) / self.dt
        diff_target = (target[..., 1:] - target[..., :-1]) / self.dt
        
        return F.l1_loss(diff_pred, diff_target)


class CorrelationLoss(nn.Module):
    """
    Phase 2: Pearson Correlation Loss.
    Equation: L = 1 - Pearson(y_true, y_pred)
    Why: Measures shape similarity (Morphology) independent of amplitude scaling.
    Standard L1/MSE can be low even if the shape is wrong (e.g. flat line vs small noise).
    Correlation ensures the "ECG-ness" is preserved.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Flatten time dimension: (B, C, T) -> (B, C*T) or (B*C, T)? 
        # Usually we want correlation per lead or per sample.
        # Let's do per lead. 
        # pred: (B, C, T)
        
        avg_pred = torch.mean(pred, dim=-1, keepdim=True)
        avg_target = torch.mean(target, dim=-1, keepdim=True)
        
        diff_pred = pred - avg_pred
        diff_target = target - avg_target
        
        cov = torch.sum(diff_pred * diff_target, dim=-1)
        std_pred = torch.sqrt(torch.sum(diff_pred ** 2, dim=-1) + self.eps)
        std_target = torch.sqrt(torch.sum(diff_target ** 2, dim=-1) + self.eps)
        
        pearson = cov / (std_pred * std_target)
        
        # Loss = 1 - mean correlation (maximize correlation)
        return 1.0 - torch.mean(pearson)


class ScientificLoss(nn.Module):
    """
    Configurable Loss Module for the Ablation Study.
    """
    def __init__(self, phase="phase0", weights={"mae": 1.0, "deriv": 0.5, "corr": 0.1}):
        super().__init__()
        self.phase = phase
        self.weights = weights
        
        self.mae = MAELoss()
        self.deriv = DerivativeLoss()
        self.corr = CorrelationLoss()
        
    def forward(self, pred, target):
        loss = 0.0
        logs = {}
        
        # Phase 0: MAE Only
        val_mae = self.mae(pred, target)
        loss += self.weights["mae"] * val_mae
        logs["mae"] = val_mae.item()
        
        if self.phase in ["phase1", "phase2"]:
            # Phase 1: Add Derivatives
            val_deriv = self.deriv(pred, target)
            loss += self.weights["deriv"] * val_deriv
            logs["deriv"] = val_deriv.item()
            
        if self.phase == "phase2":
            # Phase 2: Add Correlation
            val_corr = self.corr(pred, target)
            loss += self.weights["corr"] * val_corr
            logs["corr"] = val_corr.item()
            
        return loss, logs
