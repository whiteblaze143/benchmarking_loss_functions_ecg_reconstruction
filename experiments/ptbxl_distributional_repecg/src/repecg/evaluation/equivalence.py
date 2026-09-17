import numpy as np
from joblib import Parallel, delayed

def _single_equivalence(rng_seed: int, unique_patients: np.ndarray, num_patients: int, patient_to_indices: dict, y_true: np.ndarray, y_pred_A: np.ndarray, y_pred_B: np.ndarray, metric_fn: callable) -> float:
    rng = np.random.RandomState(rng_seed)
    sampled_patients = rng.choice(unique_patients, size=num_patients, replace=True)
    
    sampled_indices = []
    for pid in sampled_patients:
        sampled_indices.extend(patient_to_indices[pid])
        
    sampled_indices = np.array(sampled_indices)
    
    try:
        m_A = metric_fn(y_true[sampled_indices], y_pred_A[sampled_indices])
        m_B = metric_fn(y_true[sampled_indices], y_pred_B[sampled_indices])
        return m_A - m_B
    except Exception:
        return np.nan

def test_equivalence(
    y_true: np.ndarray,
    y_pred_A: np.ndarray,
    y_pred_B: np.ndarray,
    patient_ids: np.ndarray,
    metric_fn: callable,
    margin: float = 0.01,
    n_bootstraps: int = 2000,
    seed: int = 2026,
    n_jobs: int = -1
) -> dict:
    """
    E15: TOST-style equivalence analysis using patient-clustered bootstrap.
    Fully leverages multi-core CPU using joblib.Parallel.
    """
    unique_patients = np.unique(patient_ids)
    num_patients = len(unique_patients)
    
    patient_to_indices = {}
    for i, pid in enumerate(patient_ids):
        if pid not in patient_to_indices:
            patient_to_indices[pid] = []
        patient_to_indices[pid].append(i)
        
    base_rng = np.random.RandomState(seed)
    seeds = base_rng.randint(0, 2**31 - 1, size=n_bootstraps)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_equivalence)(
            seeds[i], unique_patients, num_patients, patient_to_indices, y_true, y_pred_A, y_pred_B, metric_fn
        ) for i in range(n_bootstraps)
    )
    
    delta_metrics = [r for r in results if not np.isnan(r)]
            
    if not delta_metrics:
        return {"result": "inconclusive", "mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "margin": margin}
        
    delta_metrics = np.array(delta_metrics)
    mean_delta = np.mean(delta_metrics)
    
    ci_lower = np.percentile(delta_metrics, 5.0)
    ci_upper = np.percentile(delta_metrics, 95.0)
    
    if ci_lower > margin:
        result = "A is superior to B"
    elif ci_upper < -margin:
        result = "B is superior to A"
    elif ci_lower > -margin and ci_upper < margin:
        result = "statistically equivalent within margin"
    elif ci_lower > -margin:
        result = "A is noninferior to B"
    elif ci_upper < margin:
        result = "B is noninferior to A"
    else:
        result = "inconclusive"
        
    return {
        "result": result,
        "mean_delta": float(mean_delta),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "margin": margin
    }
