"""Fixed native-float32 RR design from independently observed channels."""
import numpy as np

PATTERNS = [[True, False], [False, True], [True, True]]


def rr_design(features, rr_valid, pattern):
    rr = np.asarray(features);valid = np.asarray(rr_valid);observed = np.asarray(pattern)
    if rr.ndim != 3 or rr.shape[1:] != (2, 16) or rr.dtype != np.float32:
        raise ValueError('Expected float32 (windows,2,16) observed RR features')
    if valid.shape != rr.shape[:2] or valid.dtype != bool or observed.shape != (2,) or observed.dtype != bool or not observed.any():
        raise ValueError('Expected boolean RR validity and nonempty stored-channel subset')
    active = valid & observed[None, :]
    if np.isinf(rr[active]).any() or not np.isfinite(rr[:, :, 2][active]).all():
        raise ValueError('Invalid active RR feature cache')
    masked = np.where(active[:, :, None], rr, np.nan).reshape(len(rr), 32)
    design = np.column_stack((masked, np.broadcast_to(observed, valid.shape), active)).astype(np.float32)
    return design, active.any(axis=1)
