#!/usr/bin/env python
"""
Adaptive Loss Weighting Methods

Implements principled multi-task loss balancing:
1. GradNorm (Chen et al., 2018) - Balance gradient magnitudes
2. Uncertainty Weighting (Kendall et al., 2018) - Learn task uncertainties
3. Dynamic Weight Average (Liu et al., 2019) - Based on loss ratios

These eliminate the need for arbitrary λ constraints like Σλ ≤ 0.35.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty Weighting (Kendall et al., CVPR 2018)
    
    "Multi-Task Learning Using Uncertainty to Weigh Losses 
     for Scene Geometry and Semantics"
    
    Key equations from the paper:
    
    For regression (Eq. 7):
        L = (1/2σ²) * ||y - f(x)||² + log(σ)
        
    Implementation (Section 3.2):
        - Learn s := log(σ²) for numerical stability
        - Weight = 1/(2σ²) = 0.5 * exp(-s)
        - Regularizer = log(σ) = 0.5 * s
        
    Total loss = Σ [ 0.5 * exp(-s_i) * L_i + 0.5 * s_i ]
    
    The 0.5 * s_i term prevents σ from going to infinity (ignoring data).
    """
    
    def __init__(self, num_tasks: int = 4, init_sigma: float = 1.0):
        super().__init__()
        # Learn s = log(σ²) for numerical stability (paper Section 3.2)
        # Initialize so that σ = init_sigma, meaning s = 2*log(σ) = log(σ²)
        init_log_var = 2 * np.log(init_sigma)
        self.log_vars = nn.Parameter(torch.full((num_tasks,), init_log_var))
        self.task_names = ['mse', 'mmd', 'deriv', 'corr']
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: Dict with keys 'mse', 'mmd', 'deriv', 'corr'
        
        Returns:
            total_loss: Weighted sum with uncertainty regularization
            weights: Dict of effective weights for logging
        """
        total = 0.0
        weights = {}
        
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
                
            loss_i = losses[name]
            log_var_i = self.log_vars[i]
            
            # Weight = 1 / (2 * σ²) = 1 / (2 * exp(log_var))
            # = 0.5 * exp(-log_var)
            precision = 0.5 * torch.exp(-log_var_i)
            
            # Weighted loss + regularization
            # L_total = precision * L_i + 0.5 * log_var
            weighted_loss = precision * loss_i + 0.5 * log_var_i
            
            total = total + weighted_loss
            weights[f'w_{name}'] = precision.item()
            weights[f'sigma_{name}'] = torch.exp(0.5 * log_var_i).item()
        
        return total, weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current effective weights."""
        weights = {}
        for i, name in enumerate(self.task_names):
            precision = 0.5 * torch.exp(-self.log_vars[i])
            sigma = torch.exp(0.5 * self.log_vars[i])
            weights[f'w_{name}'] = precision.item()
            weights[f'sigma_{name}'] = sigma.item()
        return weights


class GradNorm(nn.Module):
    """
    GradNorm (Chen et al., 2018)
    
    Dynamically adjusts task weights to balance gradient magnitudes.
    Ensures all tasks train at similar rates.
    
    Key idea: If task i is training faster, reduce its weight.
    """
    
    def __init__(
        self, 
        num_tasks: int = 4,
        alpha: float = 1.5,  # Asymmetry parameter (1.5 is standard)
        init_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_names = ['mse', 'mmd', 'deriv', 'corr']
        
        # Learnable weights (will be normalized via softmax)
        if init_weights is None:
            init_weights = [1.0] * num_tasks
        self.log_weights = nn.Parameter(torch.log(torch.tensor(init_weights)))
        
        # Track initial losses for relative loss computation
        self.register_buffer('initial_losses', torch.ones(num_tasks))
        self.initialized = False
        
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        shared_layer: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted loss. 
        
        Note: GradNorm weight updates happen in a separate step (see update_weights).
        """
        # Get normalized weights
        weights = F.softmax(self.log_weights, dim=0) * self.num_tasks
        
        total = 0.0
        weight_dict = {}
        
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            total = total + weights[i] * losses[name]
            weight_dict[f'w_{name}'] = weights[i].item()
        
        return total, weight_dict
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        shared_layer: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        """
        Update weights using GradNorm algorithm.
        
        Should be called after backward() but before optimizer.step().
        """
        # Initialize baseline losses on first call
        if not self.initialized:
            for i, name in enumerate(self.task_names):
                if name in losses:
                    self.initial_losses[i] = losses[name].detach()
            self.initialized = True
            return
        
        # Compute relative inverse training rates
        # r_i = L_i(t) / L_i(0) - relative loss
        weights = F.softmax(self.log_weights, dim=0) * self.num_tasks
        
        loss_ratios = []
        grad_norms = []
        
        for i, name in enumerate(self.task_names):
            if name not in losses:
                loss_ratios.append(1.0)
                grad_norms.append(0.0)
                continue
                
            # Relative loss
            r_i = losses[name].detach() / (self.initial_losses[i] + 1e-8)
            loss_ratios.append(r_i.item())
            
            # Gradient norm w.r.t. shared layer
            if shared_layer is not None:
                grad_i = torch.autograd.grad(
                    weights[i] * losses[name],
                    shared_layer.parameters(),
                    retain_graph=True,
                    allow_unused=True
                )
                grad_norm_i = sum(g.norm() for g in grad_i if g is not None)
                grad_norms.append(grad_norm_i.item())
            else:
                grad_norms.append(1.0)
        
        # Target: balance gradient norms weighted by relative training rate
        loss_ratios = torch.tensor(loss_ratios)
        grad_norms = torch.tensor(grad_norms)
        
        mean_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / (mean_ratio + 1e-8)
        
        # Target gradient norm
        mean_grad = grad_norms.mean()
        target_grad = mean_grad * (relative_rates ** self.alpha)
        
        # GradNorm loss: minimize difference between actual and target gradients
        gradnorm_loss = F.l1_loss(grad_norms, target_grad)
        
        # Update only the weight parameters
        gradnorm_loss.backward()
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        weights = F.softmax(self.log_weights, dim=0) * self.num_tasks
        return {f'w_{name}': weights[i].item() for i, name in enumerate(self.task_names)}


class DynamicWeightAverage(nn.Module):
    """
    Dynamic Weight Average (Liu et al., 2019)
    
    Simpler than GradNorm - weights based on loss ratios.
    w_i(t) = softmax(L_i(t-1) / L_i(t-2))
    
    Tasks with increasing loss get higher weight.
    """
    
    def __init__(self, num_tasks: int = 4, temperature: float = 2.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.task_names = ['mse', 'mmd', 'deriv', 'corr']
        
        # Track loss history
        self.register_buffer('prev_losses', torch.ones(num_tasks))
        self.register_buffer('prev_prev_losses', torch.ones(num_tasks))
        self.register_buffer('weights', torch.ones(num_tasks))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted loss using dynamic weights."""
        total = 0.0
        weight_dict = {}
        
        for i, name in enumerate(self.task_names):
            if name not in losses:
                continue
            total = total + self.weights[i] * losses[name]
            weight_dict[f'w_{name}'] = self.weights[i].item()
        
        return total, weight_dict
    
    def update_weights(self, losses: Dict[str, torch.Tensor]):
        """Update weights based on loss ratios."""
        # Compute loss ratios
        ratios = []
        for i, name in enumerate(self.task_names):
            if name in losses:
                # Ratio of current to previous loss
                ratio = self.prev_losses[i] / (self.prev_prev_losses[i] + 1e-8)
                ratios.append(ratio)
                
                # Update history
                self.prev_prev_losses[i] = self.prev_losses[i]
                self.prev_losses[i] = losses[name].detach()
            else:
                ratios.append(1.0)
        
        # Compute weights via softmax
        ratios = torch.tensor(ratios, device=self.weights.device)
        self.weights = F.softmax(ratios / self.temperature, dim=0) * self.num_tasks
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return {f'w_{name}': self.weights[i].item() for i, name in enumerate(self.task_names)}


# =============================================================================
# Integration with MasonMMD
# =============================================================================

class AdaptiveMasonMMD(nn.Module):
    """
    MasonMMD with adaptive loss weighting.
    
    Replaces fixed λ values with learned/dynamic weights.
    """
    
    def __init__(
        self,
        base_model: nn.Module,  # The reconstruction U-Net
        weighting_method: str = 'uncertainty',  # 'uncertainty', 'gradnorm', 'dwa'
        **kwargs
    ):
        super().__init__()
        self.base_model = base_model
        self.weighting_method = weighting_method
        
        # Initialize weighting module
        if weighting_method == 'uncertainty':
            self.weighter = UncertaintyWeighting(num_tasks=4)
        elif weighting_method == 'gradnorm':
            self.weighter = GradNorm(num_tasks=4)
        elif weighting_method == 'dwa':
            self.weighter = DynamicWeightAverage(num_tasks=4)
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")
        
        # Loss functions (from base model or define here)
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x: torch.Tensor, target: torch.Tensor = None):
        """Forward pass with adaptive loss weighting."""
        # Get reconstruction
        pred = self.base_model(x)
        
        if target is None:
            return pred, {}
        
        # Compute individual losses
        losses = {
            'mse': self.mse_loss(pred, target),
            # Add MMD, deriv, corr losses from base_model
        }
        
        # Get weighted total
        total_loss, weights = self.weighter(losses)
        
        losses['total'] = total_loss
        losses['weights'] = weights
        
        return pred, losses


# =============================================================================
# Utility: Compare Methods
# =============================================================================

def compare_weighting_methods():
    """Demo comparing the three methods."""
    print("=" * 60)
    print("Adaptive Loss Weighting Comparison")
    print("=" * 60)
    
    # Simulated losses (varying scales)
    losses = {
        'mse': torch.tensor(0.05),
        'mmd': torch.tensor(0.002),
        'deriv': torch.tensor(0.08),
        'corr': torch.tensor(0.15)
    }
    
    print(f"\nRaw losses: {', '.join(f'{k}={v.item():.4f}' for k, v in losses.items())}")
    print()
    
    # 1. Uncertainty Weighting
    uw = UncertaintyWeighting()
    total, weights = uw(losses)
    print("Uncertainty Weighting:")
    print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items() if k.startswith('w_'))}")
    print(f"  Sigmas:  {', '.join(f'{k}={v:.3f}' for k, v in weights.items() if k.startswith('sigma_'))}")
    print(f"  Total:   {total.item():.4f}")
    print()
    
    # 2. GradNorm (simplified - no gradient computation here)
    gn = GradNorm()
    total, weights = gn(losses)
    print("GradNorm (initial):")
    print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
    print(f"  Total:   {total.item():.4f}")
    print()
    
    # 3. DWA
    dwa = DynamicWeightAverage()
    total, weights = dwa(losses)
    print("Dynamic Weight Average (initial):")
    print(f"  Weights: {', '.join(f'{k}={v:.3f}' for k, v in weights.items())}")
    print(f"  Total:   {total.item():.4f}")


if __name__ == '__main__':
    compare_weighting_methods()
