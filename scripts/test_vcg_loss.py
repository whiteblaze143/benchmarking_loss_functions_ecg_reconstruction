import torch
import torch.nn as nn
import torch.nn.functional as F

class KorsVCGLoss(nn.Module):
    def __init__(self, lambda_angle=1.0, lambda_mag=1.0, eps=1e-8):
        super().__init__()
        self.lambda_angle = lambda_angle
        self.lambda_mag = lambda_mag
        self.eps = eps
        
        # Kors matrix for (V1, V2, V3, V4, V5, V6, I, II) -> (X, Y, Z)
        # Shape: (3, 8)
        self.kors_matrix = torch.tensor([
            [-0.130,  0.050, -0.010,  0.140,  0.260,  0.110,  0.380, -0.070], # X
            [ 0.060, -0.020, -0.050,  0.060,  0.170,  0.130, -0.070,  0.930], # Y
            [-0.430, -0.060, -0.040, -0.050, -0.080, -0.090,  0.110, -0.230]  # Z
        ], dtype=torch.float32)
        
        # PTB-XL order is ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # We need to gather indices: V1=6, V2=7, V3=8, V4=9, V5=10, V6=11, I=0, II=1
        self.lead_indices = [6, 7, 8, 9, 10, 11, 0, 1]
        
    def forward(self, pred, target):
        """
        pred, target shape: (batch, 12, time)
        """
        device = pred.device
        self.kors_matrix = self.kors_matrix.to(device)
        
        # Extract 8 independent leads: (batch, 8, time)
        pred_8 = pred[:, self.lead_indices, :]
        target_8 = target[:, self.lead_indices, :]
        
        # Transform to VCG (X, Y, Z): (batch, 3, time)
        # using einsum for batch matrix multiplication: (3, 8) @ (batch, 8, time) -> (batch, 3, time)
        pred_vcg = torch.einsum('ij,bjt->bit', self.kors_matrix, pred_8)
        target_vcg = torch.einsum('ij,bjt->bit', self.kors_matrix, target_8)
        
        # 1. Spatial Angle Loss (Cosine Distance)
        # We want the 3D dipole direction to match at each time step.
        # cosine_sim: (batch, time)
        cos_sim = F.cosine_similarity(pred_vcg, target_vcg, dim=1, eps=self.eps)
        # cosine distance: 1 - cos_sim
        loss_angle = (1.0 - cos_sim).mean()
        
        # 2. Spatial Magnitude Loss (Counters R2M Dampening)
        # Enforce that the 3D dipole magnitude (L2 norm) matches the ground truth.
        pred_mag = torch.norm(pred_vcg, p=2, dim=1, keepdim=True) + self.eps
        target_mag = torch.norm(target_vcg, p=2, dim=1, keepdim=True) + self.eps
        
        # L1 difference of the magnitudes
        loss_mag = F.l1_loss(pred_mag, target_mag)
        
        loss = (self.lambda_angle * loss_angle) + (self.lambda_mag * loss_mag)
        
        return loss, loss_angle, loss_mag

if __name__ == '__main__':
    # Sanity Check (M0)
    batch = 4
    leads = 12
    time = 1000
    
    pred = torch.randn(batch, leads, time, requires_grad=True)
    target = torch.randn(batch, leads, time)
    
    criterion = KorsVCGLoss()
    loss, l_ang, l_mag = criterion(pred, target)
    
    print(f"Loss: {loss.item():.4f} (Angle: {l_ang.item():.4f}, Mag: {l_mag.item():.4f})")
    
    # Check backward pass for NaNs
    loss.backward()
    
    grad_norm = pred.grad.norm().item()
    print(f"Gradient norm: {grad_norm:.4f}")
    assert not torch.isnan(pred.grad).any(), "NaN in gradients!"
    print("M0 Sanity Check Passed! VCG Loss gradients are stable.")
