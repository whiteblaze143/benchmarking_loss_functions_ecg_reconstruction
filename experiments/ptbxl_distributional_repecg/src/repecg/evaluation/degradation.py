import numpy as np

# Standard Einthoven-Goldberger measurement matrix (Lead I, II, III, aVR, aVL, aVF, V1-V6)
# Often we represent leads as linear combinations of orthogonal dimensions.
# Here we can just define rank manually or use a standard measurement matrix.
# For simplicity, 12-lead ECG is rank 8 (2 limb + 6 precordial).
# Limb leads (I, II, III, aVR, aVL, aVF) have rank 2 because III = II - I, aVR = -(I+II)/2, aVL = (I-III)/2, aVF = (II+III)/2.
# V1-V6 are independent, each adds 1 to the rank.

def get_lead_rank(leads: list[str]) -> int:
    """
    E8: Computes the linear information rank of a set of leads.
    """
    limb_leads = {"I", "II", "III", "aVR", "aVL", "aVF"}
    precordial = {"V1", "V2", "V3", "V4", "V5", "V6"}
    
    limb_present = [l for l in leads if l in limb_leads]
    prec_present = [l for l in leads if l in precordial]
    
    # Limb rank logic
    # Rank is 2 if any two of (I, II, III) or derived are independent.
    # Actually, any 2 limb leads give rank 2. Any 1 gives rank 1.
    limb_rank = min(2, len(limb_present))
    
    # Precordial rank is simply the number of precordial leads
    prec_rank = len(prec_present)
    
    return limb_rank + prec_rank

def compute_aulac(mean_auroc_per_k: dict[int, float]) -> tuple[float, float]:
    """
    E7: Compute Area under the Lead-Availability Curve (AULAC) and Normalized Robustness (AULAC_R).
    
    Args:
        mean_auroc_per_k: Dict mapping k (1 to 12) to mean AUROC A_k
        
    Returns:
        aulac, aulac_r
    """
    if not mean_auroc_per_k or 12 not in mean_auroc_per_k:
        return 0.0, 0.0
        
    a_12 = mean_auroc_per_k[12]
    
    aulac_sum = sum(mean_auroc_per_k.get(k, 0.5) for k in range(1, 13))
    aulac = aulac_sum / 12.0
    
    if a_12 <= 0.5:
        aulac_r = 0.0
    else:
        aulac_r_sum = sum((mean_auroc_per_k.get(k, 0.5) - 0.5) / (a_12 - 0.5) for k in range(1, 13))
        aulac_r = aulac_r_sum / 12.0
        
    return aulac, aulac_r
