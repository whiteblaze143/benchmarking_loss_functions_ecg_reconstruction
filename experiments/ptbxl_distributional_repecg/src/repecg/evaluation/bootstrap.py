import numpy as np
from joblib import Parallel, delayed

def _single_bootstrap(rng_seed: int, unique_patients: np.ndarray, num_patients: int, patient_to_indices: dict, y_true: np.ndarray, y_pred: np.ndarray, metric_fn: callable) -> float:
    rng = np.random.RandomState(rng_seed)
    sampled_patients = rng.choice(unique_patients, size=num_patients, replace=True)
    
    # Gather all indices for the sampled patients
    sampled_indices = []
    for pid in sampled_patients:
        sampled_indices.extend(patient_to_indices[pid])
        
    sampled_indices = np.array(sampled_indices)
    
    try:
        return metric_fn(y_true[sampled_indices], y_pred[sampled_indices])
    except Exception:
        return np.nan

def patient_clustered_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn: callable,
    n_bootstraps: int = 2000,
    seed: int = 2026,
    n_jobs: int = -1
) -> tuple[float, float, float]:
    """
    Computes patient-clustered bootstrap confidence intervals using all available CPU cores.
    """
    unique_patients = np.unique(patient_ids)
    num_patients = len(unique_patients)
    
    patient_to_indices = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(i)
        
    # Generate deterministic seeds for each bootstrap iteration
    base_rng = np.random.RandomState(seed)
    seeds = base_rng.randint(0, 2**31 - 1, size=n_bootstraps)
    
    # Run bootstrap loops in parallel using all CPU cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(
            seeds[i], unique_patients, num_patients, patient_to_indices, y_true, y_pred, metric_fn
        ) for i in range(n_bootstraps)
    )
    
    boot_metrics = [r for r in results if not np.isnan(r)]
            
    if not boot_metrics:
        return 0.0, 0.0, 0.0
        
    boot_metrics = np.array(boot_metrics)
    mean_est = np.mean(boot_metrics)
    ci_lower = np.percentile(boot_metrics, 2.5)
    ci_upper = np.percentile(boot_metrics, 97.5)
    
    return float(mean_est), float(ci_lower), float(ci_upper)
