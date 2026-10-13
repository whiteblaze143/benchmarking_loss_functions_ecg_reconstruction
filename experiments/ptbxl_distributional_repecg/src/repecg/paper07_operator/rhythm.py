"""
Rhythm Channel for Rhythm-Aware SetOperator (z_SO+ = [z_morphology; r_rhythm]).
Computes 2D Poincaré interval atoms u_i = [log RR_i, Δlog RR_i] from detected R-peaks,
standardizes coordinates using training statistics, and embeds the empirical interval
distribution via a Nyström RKHS mean embedding.

Also provides:
- Simple RR baseline: h_RR = [mean(log RR), SD(log RR), SD(Δlog RR)] in R^3
- R-peak detector quality audit for tracking peak counts and rejection fractions.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import scipy.spatial.distance


def extract_rr_atoms(
    rpeaks: np.ndarray,
    fs: int = 500,
    min_rr_s: float = 0.25,
    max_rr_s: float = 2.50,
) -> np.ndarray:
    """Extract 2-D Poincaré log-space rhythm atoms from R-peak sample indices.

    Args:
        rpeaks: 1D array of sample indices of detected R-peaks.
        fs: Sampling rate in Hz (default: 500 Hz).
        min_rr_s: Minimum plausible RR interval in seconds (default: 0.25s = 240 bpm).
        max_rr_s: Maximum plausible RR interval in seconds (default: 2.50s = 24 bpm).

    Returns:
        np.ndarray of shape (N, 2) where column 0 is log(RR_i) and column 1 is
        Δlog(RR_i) = log(RR_{i+1}) - log(RR_i). Returns shape (0, 2) if < 3 valid intervals.
    """
    peaks = np.sort(np.asarray(rpeaks, dtype=np.float64))
    if len(peaks) < 3:
        return np.empty((0, 2), dtype=np.float32)

    # Compute RR intervals in seconds
    rr_intervals = np.diff(peaks) / float(fs)

    # Filter physiological bounds
    valid_mask = (rr_intervals >= min_rr_s) & (rr_intervals <= max_rr_s)
    rr_clean = rr_intervals[valid_mask]

    if len(rr_clean) < 2:
        return np.empty((0, 2), dtype=np.float32)

    log_rr = np.log(np.maximum(rr_clean, 1e-4))
    delta_log_rr = np.diff(log_rr)

    # Form (N, 2) array: [log_rr[i], delta_log_rr[i]]
    atoms = np.column_stack((log_rr[:-1], delta_log_rr))
    return atoms.astype(np.float32)


def extract_simple_rr_features(
    rpeaks: np.ndarray,
    fs: int = 500,
    min_rr_s: float = 0.25,
    max_rr_s: float = 2.50,
) -> np.ndarray:
    """Extract trivial 3-D RR variability baseline vector:
    h_RR = [mean(log RR), SD(log RR), SD(Δlog RR)] in R^3.

    Returns zeros if fewer than 5 valid intervals.
    """
    peaks = np.sort(np.asarray(rpeaks, dtype=np.float64))
    if len(peaks) < 3:
        return np.zeros(3, dtype=np.float32)

    rr = np.diff(peaks) / float(fs)
    valid = rr[(rr >= min_rr_s) & (rr <= max_rr_s)]
    if len(valid) < 5:
        return np.zeros(3, dtype=np.float32)

    log_rr = np.log(valid)
    d_log_rr = np.diff(log_rr)

    mean_lrr = float(np.mean(log_rr))
    sd_lrr = float(np.std(log_rr))
    sd_dlrr = float(np.std(d_log_rr)) if len(d_log_rr) > 1 else 0.0

    return np.array([mean_lrr, sd_lrr, sd_dlrr], dtype=np.float32)


def audit_rpeak_quality(
    rpeaks: np.ndarray,
    fs: int = 500,
    min_rr_s: float = 0.25,
    max_rr_s: float = 2.50,
    min_intervals: int = 5,
) -> Dict[str, Union[int, float, bool]]:
    """Audit R-peak detector performance and interval quality for a recording."""
    peaks = np.sort(np.asarray(rpeaks, dtype=np.float64))
    n_peaks = len(peaks)
    if n_peaks < 2:
        return {
            "n_detected_peaks": n_peaks,
            "n_accepted_rr": 0,
            "rr_rejection_fraction": 1.0,
            "is_valid": False,
        }

    rr = np.diff(peaks) / float(fs)
    n_total_rr = len(rr)
    valid_rr = rr[(rr >= min_rr_s) & (rr <= max_rr_s)]
    n_accepted = len(valid_rr)
    rejection_frac = float(1.0 - (n_accepted / max(n_total_rr, 1)))

    return {
        "n_detected_peaks": int(n_peaks),
        "n_accepted_rr": int(n_accepted),
        "rr_rejection_fraction": float(rejection_frac),
        "is_valid": bool(n_accepted >= min_intervals),
    }


class RhythmResponseMap:
    """Standardized Nyström RKHS map for empirical RR interval distributions.

    Maps 2-D rhythm points u = [log RR, Δlog RR] into an M-dimensional RKHS feature
    space and computes the distribution mean embedding r_RR = (1/N) * sum_i phi(u_i).
    Coordinates are standardized via training mean and scale: u_tilde = (u - mu_u) / sigma_u.
    """

    def __init__(
        self,
        landmarks: np.ndarray = None,
        whitening: np.ndarray = None,
        mean_u: np.ndarray = None,
        std_u: np.ndarray = None,
        gamma: float = 1.0,
        epsilon: float = 1e-4,
    ):
        self.landmarks = landmarks      # (M, 2) in standardized space
        self.whitening = whitening      # (M, M)
        self.mean_u = mean_u            # (2,) training mean [log_rr, delta_log_rr]
        self.std_u = std_u              # (2,) training std [log_rr, delta_log_rr]
        self.gamma = gamma
        self.epsilon = epsilon

    @property
    def dim(self) -> int:
        return len(self.landmarks) if self.landmarks is not None else 0

    def fit(
        self,
        atoms_list: List[np.ndarray],
        n_landmarks: int = 32,
        seed: int = 42,
    ) -> "RhythmResponseMap":
        """Fit landmark points using k-means on standardized training interval atoms.
        Bandwidth gamma is determined deterministically via the median pairwise distance heuristic.
        """
        valid_atoms = [a for a in atoms_list if a is not None and len(a) > 0]
        if not valid_atoms:
            raise ValueError("No valid rhythm atoms provided to fit.")

        pooled = np.vstack(valid_atoms)
        if len(pooled) < n_landmarks:
            raise ValueError(f"Not enough rhythm points ({len(pooled)}) to fit {n_landmarks} landmarks.")

        # Compute training mean and std for coordinate standardization
        self.mean_u = np.mean(pooled, axis=0).astype(np.float32)
        self.std_u = np.std(pooled, axis=0).astype(np.float32)
        self.std_u = np.maximum(self.std_u, 1e-4)  # avoid division by zero

        # Standardize pooled atoms
        pooled_std = (pooled - self.mean_u) / self.std_u

        # Subsample if pool is large for distance heuristic and k-means
        rng = np.random.default_rng(seed)
        if len(pooled_std) > 50000:
            sub_idx = rng.choice(len(pooled_std), size=50000, replace=False)
            fit_data = pooled_std[sub_idx]
        else:
            fit_data = pooled_std

        # Deterministic median pairwise distance heuristic for gamma in standardized space
        dist_sample_idx = rng.choice(len(fit_data), size=min(len(fit_data), 3000), replace=False)
        dists = scipy.spatial.distance.pdist(fit_data[dist_sample_idx])
        dists_pos = dists[dists > 1e-5]
        median_dist = float(np.median(dists_pos)) if len(dists_pos) > 0 else 1.0
        self.gamma = float(1.0 / (2.0 * max(median_dist, 0.1) ** 2))

        # Select landmark centers via k-means in standardized space
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_landmarks, random_state=seed, n_init=5)
        km.fit(fit_data)
        self.landmarks = km.cluster_centers_.astype(np.float32)

        # Compute landmark kernel matrix K_MM in standardized space
        dist_sq = scipy.spatial.distance.cdist(self.landmarks, self.landmarks, metric="sqeuclidean")
        k_mm = np.exp(-self.gamma * dist_sq).astype(np.float64)

        # Regularized whitening: (K_MM + eps * I)^{-1/2}
        k_reg = k_mm + self.epsilon * np.eye(n_landmarks, dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(k_reg)
        eigvals = np.maximum(eigvals, 1e-7)
        self.whitening = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype(np.float32)

        return self

    def embed_atoms(self, atoms: np.ndarray) -> np.ndarray:
        """Compute Nyström feature map phi(u) for each atom in atoms (N, 2)."""
        if self.landmarks is None or self.whitening is None or self.mean_u is None:
            raise RuntimeError("RhythmResponseMap is not fitted or loaded.")
        if len(atoms) == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        # Standardize coordinates using frozen training statistics
        atoms_std = (atoms - self.mean_u) / self.std_u

        # Kernel distances to standardized landmarks: K(atoms_std, landmarks) -> (N, M)
        dists_sq = scipy.spatial.distance.cdist(atoms_std, self.landmarks, metric="sqeuclidean")
        k_nm = np.exp(-self.gamma * dists_sq).astype(np.float32)

        # Whitened features: phi(u) = K_NM * W -> (N, M)
        phi = k_nm @ self.whitening
        return phi

    def mean_embedding(self, atoms: np.ndarray, min_intervals: int = 5) -> np.ndarray:
        """Compute empirical mean embedding r_RR = (1/N) * sum_i phi(u_i) in R^M.

        If atoms has fewer than min_intervals (default: 5 intervals), returns zero vector.
        """
        if self.landmarks is None:
            raise RuntimeError("RhythmResponseMap is not fitted.")
        if len(atoms) < min_intervals:
            return np.zeros(self.dim, dtype=np.float32)

        phi = self.embed_atoms(atoms)
        return np.mean(phi, axis=0).astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        """Save landmarks, whitening matrix, and parameters to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            landmarks=self.landmarks,
            whitening=self.whitening,
            mean_u=self.mean_u,
            std_u=self.std_u,
            gamma=np.array(self.gamma, dtype=np.float32),
            epsilon=np.array(self.epsilon, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RhythmResponseMap":
        """Load landmarks, whitening matrix, and parameters from .npz file."""
        data = np.load(path)
        return cls(
            landmarks=data["landmarks"],
            whitening=data["whitening"],
            mean_u=data["mean_u"],
            std_u=data["std_u"],
            gamma=float(data["gamma"]),
            epsilon=float(data["epsilon"]),
        )
