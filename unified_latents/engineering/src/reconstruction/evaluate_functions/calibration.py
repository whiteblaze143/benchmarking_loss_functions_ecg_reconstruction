
import numpy as np
import torch
import torch.nn.functional as F

def compute_ence(probs, labels, n_bins=15):
    """
    Computes Expected Normalized Calibration Error (ENCE).
    ENCE = (1/K) * sum(|Acc(Bk) - Conf(Bk)| / Conf(Bk)) ? 
    Standard ECE = sum(bk/N * |acc - conf|)
    
    ENCE normalizes by confidence, penalizing overconfidence in low-conf regions more?
    Actually, standard ECE is most common.
    Let's implement ECE and Adaptive ECE.
    """
    # ...
    pass

def compute_calibration_metrics(probs, labels, n_bins=10, strategy='uniform'):
    """
    Computes ECE, MCE, and Reliability Diagram data.
    
    Args:
        probs: [N, n_classes] or [N] (binary)
        labels: [N] 
        n_bins: number of bins
        strategy: 'uniform' or 'quantile' (adaptive)
        
    Returns:
        dict: {ece, mce, bins_acc, bins_conf, bins_count}
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    # Handle Binary vs Multi-class
    if len(probs.shape) == 1 or probs.shape[1] == 1:
        # Binary
        confidences = probs.reshape(-1)
        predictions = (confidences > 0.5).astype(int)
        accuracies = (predictions == labels) # ? No, calibration is about P(True) vs True
        # For binary: calibration is usually checking if P(y=1) matches empirical freq of 1.
        # But standard ECE uses max(probs) and checks if it matches accuracy.
        # Strict Calibration: P(y=1) vs y
        
        # Let's use max-class calibration (standard for classification)
        pass 
    else:
        # Multi-class
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        labels = labels.reshape(-1)
        correct = (predictions == labels).astype(int)
        
    # Binning
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(confidences, quantiles * 100)
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    ece = 0.0
    mce = 0.0
    
    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i+1]
        
        # indices in this bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        count = np.sum(in_bin)
        if count > 0:
            acc = np.mean(correct[in_bin])
            conf = np.mean(confidences[in_bin])
            
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(count)
            
            abs_diff = np.abs(acc - conf)
            ece += (count / len(confidences)) * abs_diff
            mce = max(mce, abs_diff)
            
    return {
        'ece': ece,
        'mce': mce,
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts
    }
