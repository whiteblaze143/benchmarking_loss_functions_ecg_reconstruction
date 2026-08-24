from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

def pearson_loss(pred, target):
    # Shape: (batch, leads, time)
    pred_mean = pred.mean(dim=-1, keepdim=True)
    target_mean = target.mean(dim=-1, keepdim=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    # Clamp each variance before sqrt. Adding epsilon after sqrt still leaves
    # an infinite derivative at exactly zero variance, which caused the cNVAE
    # Pearson runs to corrupt their weights on a deterministic flat-signal
    # batch. This form keeps both the forward value and backward pass finite.
    variance_floor = 1e-8
    cov = (pred_centered * target_centered).sum(dim=-1)
    var_pred_raw = (pred_centered**2).sum(dim=-1)
    var_target_raw = (target_centered**2).sum(dim=-1)
    valid = (var_pred_raw >= variance_floor) & (var_target_raw >= variance_floor)
    denominator = torch.sqrt(
        var_pred_raw.clamp_min(variance_floor)
        * var_target_raw.clamp_min(variance_floor)
    )
    corr = torch.where(valid, cov / denominator, torch.zeros_like(cov))
    return 1.0 - corr.mean()

def derivative_loss(pred, target):
    # Match the first derivatives
    pred_d1 = pred[..., 1:] - pred[..., :-1]
    target_d1 = target[..., 1:] - target[..., :-1]
    return F.mse_loss(pred_d1, target_d1)

def mmd_loss(
    pred,
    target,
    bandwidth_multipliers=(0.5, 1.0, 2.0, 4.0),
    eps=1e-12,
):
    """Scale-adaptive multi-kernel MMD for full ECG records.

    The legacy implementation applied a fixed-width RBF kernel to Euclidean
    distances in 45,000--60,000 dimensions. At ECG scale all off-diagonal
    kernels underflowed, leaving the constant self-kernel diagonal and an
    effectively zero gradient. Here distances are mean squared per feature and
    the base bandwidth is the detached median of the current batch distances.
    Multiple widths keep gradients useful when prediction and target
    distributions are at different stages of convergence.
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("Expected matching [batch, leads, time] tensors")
    if not bandwidth_multipliers or any(value <= 0 for value in bandwidth_multipliers):
        raise ValueError("bandwidth_multipliers must contain positive values")

    batch_size = pred.size(0)
    feature_count = pred[0].numel()
    # Distance/kernel arithmetic in float32 avoids AMP underflow while
    # preserving autograd back to the model's original dtype.
    pred_flat = pred.reshape(batch_size, -1).float()
    target_flat = target.reshape(batch_size, -1).float()
    distance_xx = torch.cdist(pred_flat, pred_flat, p=2).square() / feature_count
    distance_yy = torch.cdist(target_flat, target_flat, p=2).square() / feature_count
    distance_xy = torch.cdist(pred_flat, target_flat, p=2).square() / feature_count

    with torch.no_grad():
        if batch_size > 1:
            off_diagonal = ~torch.eye(
                batch_size, dtype=torch.bool, device=pred.device
            )
            reference = torch.cat(
                (
                    distance_xx[off_diagonal],
                    distance_yy[off_diagonal],
                    distance_xy.reshape(-1),
                )
            )
        else:
            reference = distance_xy.reshape(-1)
        positive = reference[torch.isfinite(reference) & (reference > eps)]
        base_bandwidth = (
            positive.median() if positive.numel() else reference.new_tensor(1.0)
        ).clamp_min(eps)

    value = pred_flat.new_zeros(())
    for multiplier in bandwidth_multipliers:
        bandwidth = base_bandwidth * float(multiplier)
        k_xx = torch.exp(-distance_xx / (2.0 * bandwidth)).mean()
        k_yy = torch.exp(-distance_yy / (2.0 * bandwidth)).mean()
        k_xy = torch.exp(-distance_xy / (2.0 * bandwidth)).mean()
        value = value + k_xx + k_yy - 2.0 * k_xy
    return value / len(bandwidth_multipliers)


def rational_quadratic_mmd_loss(pred, target, alpha=1.0, c=1.0):
    """Paper-parity per-lead MMD with a rational-quadratic kernel."""
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("Expected matching [batch, leads, time] tensors")

    # Treat records as samples and compute a separate kernel for each lead.
    pred_lead = pred.transpose(0, 1)
    target_lead = target.transpose(0, 1)

    def kernel(x, y):
        distance_sq = torch.cdist(x, y, p=2).pow(2)
        return (1.0 + distance_sq / (2.0 * alpha * c)) ** (-alpha)

    values = []
    for x, y in zip(pred_lead, target_lead):
        values.append(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())
    return torch.stack(values).mean()


def derivative_l1_loss(pred, target):
    """Paper-parity L1 distance between first temporal differences."""
    return F.l1_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])

class KorsVCGLoss(nn.Module):
    def __init__(self, lambda_angle=1.0, lambda_mag=1.0, eps=1e-8):
        super().__init__()
        self.lambda_angle = lambda_angle
        self.lambda_mag = lambda_mag
        self.eps = eps
        
        # Kors matrix for (V1, V2, V3, V4, V5, V6, I, II) -> (X, Y, Z)
        # Shape: (3, 8)
        self.register_buffer('kors_matrix', torch.tensor([
            [-0.130,  0.050, -0.010,  0.140,  0.260,  0.110,  0.380, -0.070], # X
            [ 0.060, -0.020, -0.050,  0.060,  0.170,  0.130, -0.070,  0.930], # Y
            [-0.430, -0.060, -0.040, -0.050, -0.080, -0.090,  0.110, -0.230]  # Z
        ], dtype=torch.float32))
        
        # PTB-XL order is ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # We need to gather indices: V1=6, V2=7, V3=8, V4=9, V5=10, V6=11, I=0, II=1
        self.lead_indices = [6, 7, 8, 9, 10, 11, 0, 1]
        
    def forward(self, pred, target):
        """
        pred, target shape: (batch, 12, time)
        """
        kors = self.kors_matrix.to(pred.device)
        # Extract 8 independent leads: (batch, 8, time)
        pred_8 = pred[:, self.lead_indices, :]
        target_8 = target[:, self.lead_indices, :]
        
        # Transform to VCG (X, Y, Z): (batch, 3, time)
        pred_vcg = torch.einsum('ij,bjt->bit', kors, pred_8)
        target_vcg = torch.einsum('ij,bjt->bit', kors, target_8)
        
        # 1. Spatial Angle Loss (Cosine Distance)
        cos_sim = F.cosine_similarity(pred_vcg, target_vcg, dim=1, eps=self.eps)
        loss_angle = (1.0 - cos_sim).mean()
        
        # 2. Spatial Magnitude Loss (Counters R2M Dampening)
        pred_mag = torch.norm(pred_vcg, p=2, dim=1, keepdim=True) + self.eps
        target_mag = torch.norm(target_vcg, p=2, dim=1, keepdim=True) + self.eps
        loss_mag = F.l1_loss(pred_mag, target_mag)
        
        loss = (self.lambda_angle * loss_angle) + (self.lambda_mag * loss_mag)
        return loss, loss_angle, loss_mag

def energy_distance_loss(pred, target):
    """
    Empirical Energy Distance computed per lead.
    pred, target: (batch, leads, time)
    """
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("Expected matching [batch, leads, time] tensors")
        
    batch_size, leads, time = pred.shape
    loss = pred.new_zeros(())
    
    pred_flat = pred.transpose(0, 1) # (leads, batch, time)
    target_flat = target.transpose(0, 1)
    
    for l in range(leads):
        x = pred_flat[l] # (batch, time)
        y = target_flat[l] # (batch, time)
        
        d_xy = torch.cdist(x, y, p=2).mean()
        d_xx = torch.cdist(x, x, p=2).mean()
        d_yy = torch.cdist(y, y, p=2).mean()
        
        loss += 2.0 * d_xy - d_xx - d_yy
        
    return loss / leads

def laplacian_kernel(x, y, bandwidth):
    dist = torch.cdist(x, y, p=2)
    return torch.exp(-dist / bandwidth)

def multiscale_imq_kernel(x, y, eps=1e-8):
    """Multiscale Inverse-Multiquadratic kernel."""
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    
    # Calculate median pairwise distance of target as the base bandwidth m
    with torch.no_grad():
        dist_yy_sq = torch.cdist(y, y, p=2).pow(2)
        batch_size = y.size(0)
        if batch_size > 1:
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=y.device)
            positive = dist_yy_sq[off_diagonal]
        else:
            positive = dist_yy_sq.reshape(-1)
            
        positive = positive[torch.isfinite(positive) & (positive > eps)]
        m_sq = (positive.median() if positive.numel() else y.new_tensor(1.0)).clamp_min(eps)
        m = torch.sqrt(m_sq)
        
    multipliers = [0.5, 1.0, 2.0, 4.0]
    kernel_val = x.new_zeros(())
    
    for c_mult in multipliers:
        c = c_mult * m
        c_sq = c.pow(2)
        k = (dist_sq + c_sq).pow(-0.5)
        kernel_val = kernel_val + k.mean()
        
    return kernel_val / len(multipliers)

def single_imq_kernel(x, y, c_sq):
    dist_sq = torch.cdist(x, y, p=2).pow(2)
    return (dist_sq + c_sq).pow(-0.5).mean()

def anatomical_block_mmd_loss(pred, target, kernel_type='imq_multi'):
    """
    Applies MMD locally within named anatomical blocks.
    Blocks:
      Inferior: II, III, aVF
      Anteroseptal: V1–V4
      Lateral: I, aVL, V5, V6
      Limb plane: I, II, III, aVR, aVL, aVF
      Precordial: V1–V6
    PTB-XL Order: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    """
    lead_map = {
        'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
        'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11
    }
    
    blocks = {
        'Inferior': ['II', 'III', 'aVF'],
        'Anteroseptal': ['V1', 'V2', 'V3', 'V4'],
        'Lateral': ['I', 'aVL', 'V5', 'V6'],
        'Limb': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF'],
        'Precordial': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    }
    
    total_loss = pred.new_zeros(())
    
    for block_name, lead_names in blocks.items():
        indices = [lead_map[ln] for ln in lead_names]
        # pred_block: (batch, num_leads_in_block * time)
        pred_block = pred[:, indices, :].contiguous().view(pred.size(0), -1)
        target_block = target[:, indices, :].contiguous().view(target.size(0), -1)
        
        if kernel_type == 'imq_multi':
            k_xx = multiscale_imq_kernel(pred_block, pred_block)
            k_yy = multiscale_imq_kernel(target_block, target_block)
            k_xy = multiscale_imq_kernel(pred_block, target_block)
            mmd = k_xx + k_yy - 2.0 * k_xy
        elif kernel_type == 'laplacian':
            with torch.no_grad():
                d_yy = torch.cdist(target_block, target_block, p=2)
                bs = target_block.size(0)
                if bs > 1:
                    off = ~torch.eye(bs, dtype=torch.bool, device=pred.device)
                    pos = d_yy[off]
                else:
                    pos = d_yy.reshape(-1)
                pos = pos[pos > 1e-8]
                bw = pos.median() if pos.numel() else target_block.new_tensor(1.0)
                
            k_xx = laplacian_kernel(pred_block, pred_block, bw).mean()
            k_yy = laplacian_kernel(target_block, target_block, bw).mean()
            k_xy = laplacian_kernel(pred_block, target_block, bw).mean()
            mmd = k_xx + k_yy - 2.0 * k_xy
        elif kernel_type == 'imq_single':
            with torch.no_grad():
                d_yy_sq = torch.cdist(target_block, target_block, p=2).pow(2)
                bs = target_block.size(0)
                if bs > 1:
                    off = ~torch.eye(bs, dtype=torch.bool, device=pred.device)
                    pos = d_yy_sq[off]
                else:
                    pos = d_yy_sq.reshape(-1)
                pos = pos[pos > 1e-8]
                m_sq = (pos.median() if pos.numel() else target_block.new_tensor(1.0)).clamp_min(1e-8)
            
            k_xx = single_imq_kernel(pred_block, pred_block, m_sq)
            k_yy = single_imq_kernel(target_block, target_block, m_sq)
            k_xy = single_imq_kernel(pred_block, target_block, m_sq)
            mmd = k_xx + k_yy - 2.0 * k_xy
            
        # Normalize block weights so overlapping leads don't get massive gradients
        total_loss += mmd / len(lead_names)
        
    return total_loss / len(blocks)

def lead_consistency_loss(pred):
    """
    Penalizes deviations from Goldberger's equations for the generated limb leads.
    """
    if pred.shape[1] < 6:
        return pred.new_zeros(())
        
    I, II = pred[:, 0, :], pred[:, 1, :]
    III, aVR, aVL, aVF = pred[:, 2, :], pred[:, 3, :], pred[:, 4, :], pred[:, 5, :]
    
    expected_III = II - I
    expected_aVR = -(I + II) / 2.0
    expected_aVL = I - II / 2.0
    expected_aVF = II - I / 2.0
    
    loss = (
        F.mse_loss(III, expected_III) +
        F.mse_loss(aVR, expected_aVR) +
        F.mse_loss(aVL, expected_aVL) +
        F.mse_loss(aVF, expected_aVF)
    )
    return loss / 4.0

@dataclass(frozen=True)
class FactorialLossConfig:
    # Standard losses
    mse: bool
    corr: bool
    deriv: bool
    vcg: bool
    # Non-parametric / structural losses
    ed: bool
    lead: bool
    mmd_kernel: int # 0: None, 1: Global RBF, 2: Anatomical Laplacian, 3: Anatomical IMQ-Multi, 4: Temporal KMeans IMQ-Multi
    
    lambda_mse: float = 1.0
    lambda_corr: float = 1.7
    lambda_deriv: float = 4.1
    lambda_vcg: float = 5.0
    lambda_ed: float = 1.0
    lambda_lead: float = 1.0
    lambda_mmd: float = 2.3

    @property
    def mask(self) -> str:
        return f"{int(self.mse)}{int(self.corr)}{int(self.deriv)}{int(self.vcg)}{int(self.ed)}{int(self.lead)}{self.mmd_kernel}"

    @classmethod
    def from_mask(cls, mask: str, **weights):
        if len(mask) != 7:
            raise ValueError(f"Invalid factorial mask {mask!r}; expected 7 digits")
        return cls(
            mse=(mask[0] == "1"),
            corr=(mask[1] == "1"),
            deriv=(mask[2] == "1"),
            vcg=(mask[3] == "1"),
            ed=(mask[4] == "1"),
            lead=(mask[5] == "1"),
            mmd_kernel=int(mask[6]),
            **weights
        )

    def to_dict(self):
        return {**asdict(self), "mask": self.mask}

def kmeans_temporal_block_mmd_loss(pred, target, num_clusters=4, eps=1e-8):
    """
    Implements dynamic temporal blocking inspired by Senanayake & Jeganathan (2024).
    Instead of fixed anatomical leads, we cluster the temporal signal into contiguous-like 
    phases (e.g. P-wave, QRS, T-wave, baseline) using K-Means on the mean target signal,
    and then compute Multiscale IMQ MMD across the batch within those dynamic blocks.
    """
    batch_size, leads, time = pred.shape
    loss = pred.new_zeros(())
    
    # 1. Compute mean signal across batch to define the phases
    # mean_target: (time, leads)
    mean_target = target.mean(dim=0).transpose(0, 1)
    
    # Simple, fast 1D K-means (Lloyd's algorithm) implemented in PyTorch
    with torch.no_grad():
        indices = torch.randperm(time, device=target.device)[:num_clusters]
        centroids = mean_target[indices]
        
        for _ in range(5): # 5 iterations is usually enough for temporal ECG clustering
            dists = torch.cdist(mean_target, centroids) # (time, K)
            assignments = torch.argmin(dists, dim=1)
            
            # Update centroids
            for k in range(num_clusters):
                mask = (assignments == k)
                if mask.any():
                    centroids[k] = mean_target[mask].mean(dim=0)
                    
    # 2. Compute MMD across the batch for each temporal cluster
    for k in range(num_clusters):
        mask = (assignments == k)
        if not mask.any():
            continue
            
        # extract points for this cluster across the batch
        # shape: (batch, leads * len_k)
        t_k = target[:, :, mask].contiguous().view(batch_size, -1)
        p_k = pred[:, :, mask].contiguous().view(batch_size, -1)
        
        if batch_size < 2:
            continue
            
        k_xx = multiscale_imq_kernel(p_k, p_k, eps)
        k_yy = multiscale_imq_kernel(t_k, t_k, eps)
        k_xy = multiscale_imq_kernel(p_k, t_k, eps)
        
        loss += (k_xx + k_yy - 2.0 * k_xy)
        
    return loss / num_clusters

class CombinatorialCompositeLoss(nn.Module):
    def __init__(self, mask="1000000"):
        super().__init__()
        self.config = FactorialLossConfig.from_mask(mask)
        self.vcg_loss_fn = KorsVCGLoss() if self.config.vcg else None
        
        # Scaling normalizers to keep everything relative to MSE
        self.NORM = {"mse": 0.18, "mmd": 0.042, "deriv": 0.12, "corr": 1.0, "vcg": 1.0, "ed": 0.05, "lead": 0.1}

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        
        corr = pearson_loss(pred, target) if self.config.corr else pred.new_zeros(())
        deriv = derivative_l1_loss(pred, target) if self.config.deriv else pred.new_zeros(())
        ed = energy_distance_loss(pred, target) if self.config.ed else pred.new_zeros(())
        lead = lead_consistency_loss(pred) if self.config.lead else pred.new_zeros(())
        
        vcg = pred.new_zeros(())
        if self.config.vcg:
            vcg, _, _ = self.vcg_loss_fn(pred, target)
            
        mmd = pred.new_zeros(())
        if self.config.mmd_kernel == 1:
            mmd = mmd_loss(pred, target)
        elif self.config.mmd_kernel == 2:
            mmd = anatomical_block_mmd_loss(pred, target, kernel_type='laplacian')
        elif self.config.mmd_kernel == 3:
            mmd = anatomical_block_mmd_loss(pred, target, kernel_type='imq_multi')
        elif self.config.mmd_kernel == 4:
            mmd = kmeans_temporal_block_mmd_loss(pred, target)
            
        # Composite accumulation
        loss = pred.new_zeros(())
        if self.config.mse:
            loss = loss + (self.config.lambda_mse * mse / self.NORM["mse"])
        if self.config.corr:
            loss = loss + (self.config.lambda_corr * corr / self.NORM["corr"])
        if self.config.deriv:
            loss = loss + (self.config.lambda_deriv * deriv / self.NORM["deriv"])
        if self.config.vcg:
            loss = loss + (self.config.lambda_vcg * vcg / self.NORM["vcg"])
        if self.config.ed:
            loss = loss + (self.config.lambda_ed * ed / self.NORM["ed"])
        if self.config.lead:
            loss = loss + (self.config.lambda_lead * lead / self.NORM["lead"])
        if self.config.mmd_kernel > 0:
            loss = loss + (self.config.lambda_mmd * mmd / self.NORM["mmd"])
            
        return loss, mse, corr, deriv, vcg, ed, lead, mmd


