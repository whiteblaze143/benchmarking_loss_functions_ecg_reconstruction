
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def patient_cluster_bootstrap_auroc(y_true, y_pred, patient_ids, n_bootstraps=1000, seed=42):
    """
    Computes AUROC with Clustered Bootstrapping (resampling patients).
    
    Args:
        y_true: [N, n_classes] or [N] numpy array
        y_pred: [N, n_classes] or [N] numpy array (probabilities)
        patient_ids: [N] array of patient IDs corresponding to each record
        n_bootstraps: int
        
    Returns:
        dict: {
            'auroc_mean': float,
            'auroc_std': float,
            'ci_lower': float,
            'ci_upper': float,
            'boot_scores': list
        }
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    boot_scores = []
    
    # Pre-group indices by patient for speed
    df = pd.DataFrame({'pid': patient_ids, 'idx': range(len(patient_ids))})
    patient_map = df.groupby('pid')['idx'].apply(np.array).to_dict()
    
    print(f"Bootstrapping AUROC ({n_bootstraps} iterations) on {n_patients} patients...")
    
    for _ in tqdm(range(n_bootstraps), leave=False):
        # resample patients with replacement
        resampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        
        # collect indices
        indices = []
        for pid in resampled_patients:
            indices.append(patient_map[pid])
            
        indices = np.concatenate(indices)
        
        # calculate metric
        if len(np.unique(y_true[indices])) < 2:
            # Handle edge case where resample has only 1 class
            continue
            
        try:
            score = roc_auc_score(y_true[indices], y_pred[indices], average='macro')
            boot_scores.append(score)
        except ValueError:
            pass
            
    boot_scores = np.array(boot_scores)
    
    return {
        'auroc_mean': np.mean(boot_scores),
        'auroc_std': np.std(boot_scores),
        'ci_lower': np.percentile(boot_scores, 2.5),
        'ci_upper': np.percentile(boot_scores, 97.5),
        'boot_scores': boot_scores
    }

def compare_models_bootstrap(y_true, y_pred_a, y_pred_b, patient_ids, n_bootstraps=1000, seed=42):
    """
    Computes p-value for Model A > Model B using paired cluster bootstrap.
    """
    rng = np.random.RandomState(seed)
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    df = pd.DataFrame({'pid': patient_ids, 'idx': range(len(patient_ids))})
    patient_map = df.groupby('pid')['idx'].apply(np.array).to_dict()
    
    diffs = []
    
    print(f"Comparing Models ({n_bootstraps} iterations)...")
    
    for _ in tqdm(range(n_bootstraps), leave=False):
        resampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
        indices = np.concatenate([patient_map[pid] for pid in resampled_patients])
        
        setup_valid = len(np.unique(y_true[indices])) > 1
        
        if setup_valid:
            score_a = roc_auc_score(y_true[indices], y_pred_a[indices], average='macro')
            score_b = roc_auc_score(y_true[indices], y_pred_b[indices], average='macro')
            diffs.append(score_a - score_b)
            
    diffs = np.array(diffs)
    # p-value: fraction of times A was NOT better than B (if testing A > B)
    # 1-sided p-value
    p_value = np.mean(diffs <= 0)
    
    return {
        'delta_mean': np.mean(diffs),
        'p_value': p_value,
        'ci_lower': np.percentile(diffs, 2.5),
        'ci_upper': np.percentile(diffs, 97.5)
    }
