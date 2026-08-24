
import torch
import torch.nn.functional as F
import numpy as np

def mixture_logistic_nll(target, logits, num_mixtures=8, num_dims=12):
    """
    Computes Negative Log Likelihood for Discretized Mixture of Logistics.
    
    Args:
        target: [B, C, T] - Ground truth ECG (normalized -1 to 1 approx)
        logits: [B, C * 3 * M, T] - Predicted parameters
        num_mixtures (M): Number of mixture components
        num_dims (C): Number of channels (e.g., 12)
        
    Returns:
        nll: Scalar loss (mean over batch, sum over Time/Channels)
    """
    B, _, T = target.shape
    
    # Reshape logits to [B, C, 3, M, T]
    # Layout assumed: First M are structure params?
    # Actually, cNVAE usually outputs [B, M + (C*M*3_ish), T]? No.
    # Standard PixelCNN++ layout:
    # 1. Logit Probs (Mix Weights): [B, C, M,  T] (or shared across C?)
    # 2. Means: [B, C, M, T]
    # 3. Log Scales: [B, C, M, T]
    # 4. Coeffs (Autoregression): [B, C, M, T] - WE SKIP THIS FOR INDEPENDENT BASELINE
    
    # Let's assume the Head Output is structured as:
    # [logit_probs (M), params (C * M * 2)] -> Need to align with Wrapper
    
    # Alternative: The Wrapper outputs [B, C * M * 3, T]
    # C groups of (MixWeight, Mean, LogScale) per mixture?
    # Or M groups of C?
    
    # Let's define the STANDARD used in our wrapper:
    # Output: [B, C * M * 3, T]
    # Reshape: [B, C, M, 3, T] => 0:Weight, 1:Mean, 2:Scale
    
    logits = logits.view(B, num_dims, num_mixtures, 3, T)
    
    mix_logits = logits[:, :, :, 0, :] # [B, C, M, T] - Weights
    means      = logits[:, :, :, 1, :] # [B, C, M, T]
    
    # Stability Fix: Use Softplus to constrain scale > 1e-5 smoothly
    # Hard clamp min=-7.0 allows scale ~ 1e-3. 
    # Use Softplus + eps
    unconstrained_scale = logits[:, :, :, 2, :]
    min_scale = 1e-5
    scales = F.softplus(unconstrained_scale) + min_scale
    log_scales = torch.log(scales)
    
    # Target reshape for broadcasting: [B, C, 1, T]
    y = target.unsqueeze(2)
    
    # Standard Discretized Logistic Log-Prob
    # References: PixelCNN++, OpenAI NVAE
    # We assume 'y' is continuous but we model it as discretized density if needed
    # Or just continuous logistic PDF if data is floats. 
    # cNVAE uses discretized (cdf_delta). For ECG floats, CDF delta is robust.
    
    inv_stdv = torch.exp(-log_scales)
    
    # We assume data is scaled roughly [-1, 1]. 
    # Bin width 1/256? Or just small epsilon? 
    # cNVAE uses 2^8 - 1 for max_val. 
    bin_width = 1.0 / 255.0 # approximation
    
    plus_in = inv_stdv * (y - means + bin_width/2)
    min_in  = inv_stdv * (y - means - bin_width/2)
    
    cdf_plus = torch.sigmoid(plus_in)
    cdf_min  = torch.sigmoid(min_in)
    
    # Robust Log-CDF
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    
    cdf_delta = cdf_plus - cdf_min
    
    # Mid-range robust
    mid_in = inv_stdv * (y - means)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)
    
    # If delta is too small, approximate with PDF * bin_width
    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5,
        torch.log(torch.clamp(cdf_delta, min=1e-10)),
        log_pdf_mid + np.log(bin_width) 
    )
    
    # Edge cases (saturation)
    # If y < -0.999...
    log_probs = torch.where(
        y < -0.999, 
        log_cdf_plus, 
        torch.where(
            y > 0.999, 
            log_one_minus_cdf_min,
            log_prob_mid_safe
        )
    ) # [B, C, M, T]
    
    # Add Mixture Weights
    # mix_logits are unnormalized.
    log_mix_probs = F.log_softmax(mix_logits, dim=2) # Normalize over M
    
    # Final Log Sum Exp
    # sum_m ( pi_m * prob_m ) -> logsumexp( log_pi + log_prob )
    final_log_prob = torch.logsumexp(log_mix_probs + log_probs, dim=2) # [B, C, T]
    
    return -torch.mean(torch.sum(final_log_prob, dim=[1,2])) # Mean over Batch, Sum over C, T
