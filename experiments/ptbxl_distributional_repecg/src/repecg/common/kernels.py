from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans


@dataclass(frozen=True)
class WhiteningTransform:
    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, *, relative_floor: float = 1e-8) -> "WhiteningTransform":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("whitening requires a two-dimensional sample matrix")
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(len(values) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        floor = max(float(eigenvalues[0]) * relative_floor, 1e-12)
        scales = np.maximum(eigenvalues, floor) ** -0.5
        return cls(mean=mean, components=eigenvectors, scales=scales)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.components * self.scales


@dataclass(frozen=True)
class TruncatedPCAWhitening:
    mean: np.ndarray
    components: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, n_components: int = 64) -> "TruncatedPCAWhitening":
        values = np.asarray(x, dtype=np.float64)
        mean = values.mean(axis=0)
        centered = values - mean
        covariance = centered.T @ centered / max(len(values) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        keep = eigenvalues >= max(eigenvalues[0] * 1e-8, 1e-12)
        count = min(n_components, int(keep.sum()))
        if count == 0:
            raise ValueError("descriptor covariance has no supported direction")
        return cls(
            mean=mean,
            components=eigenvectors[:, :count],
            scales=eigenvalues[:count] ** -0.5,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) @ self.components * self.scales


def imq_kernel(x: np.ndarray, y: np.ndarray, *, c2: float = 1.0) -> np.ndarray:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    d2 = ((left[:, None, :] - right[None, :, :]) ** 2).sum(axis=-1)
    return 1.0 / np.sqrt(d2 + c2)


def biased_mmd2(x: np.ndarray, y: np.ndarray, *, c2: float = 1.0) -> float:
    return float(
        imq_kernel(x, x, c2=c2).mean()
        + imq_kernel(y, y, c2=c2).mean()
        - 2.0 * imq_kernel(x, y, c2=c2).mean()
    )


@dataclass(frozen=True)
class NystromMap:
    landmarks: np.ndarray
    inverse_root: np.ndarray
    c2: float

    @classmethod
    def fit(
        cls,
        whitened: np.ndarray,
        *,
        landmarks: int = 128,
        c2: float = 1.0,
        seed: int = 42,
    ) -> "NystromMap":
        values = np.asarray(whitened, dtype=np.float64)
        if len(values) < landmarks:
            raise ValueError("fewer observations than requested landmarks")
        model = MiniBatchKMeans(
            n_clusters=landmarks,
            random_state=seed,
            batch_size=min(4096, len(values)),
            n_init=3,
        ).fit(values)
        anchors = model.cluster_centers_
        gram = imq_kernel(anchors, anchors, c2=c2)
        ridge = 1e-6 * np.trace(gram) / landmarks
        eigenvalues, eigenvectors = np.linalg.eigh(gram + ridge * np.eye(landmarks))
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        inverse_root = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
        return cls(landmarks=anchors, inverse_root=inverse_root, c2=c2)

    def transform(self, whitened: np.ndarray) -> np.ndarray:
        return imq_kernel(whitened, self.landmarks, c2=self.c2) @ self.inverse_root

    def mean(self, whitened: np.ndarray) -> np.ndarray:
        return self.transform(whitened).mean(axis=0)


def audit_nystrom(
    exact: np.ndarray,
    approximate: np.ndarray,
    *,
    min_spearman: float = 0.90,
    max_median_relative_error: float = 0.15,
) -> dict[str, float | bool]:
    exact = np.asarray(exact, dtype=np.float64)
    approximate = np.asarray(approximate, dtype=np.float64)
    positive = exact[exact > 0]
    median_positive = float(np.median(positive)) if len(positive) else 0.0
    floor = max(1e-8, 0.001 * median_positive)
    relative = np.abs(approximate - exact) / np.maximum(exact, floor)
    rho = float(spearmanr(exact, approximate).statistic)
    median_relative = float(np.median(relative))
    return {
        "spearman": rho,
        "median_relative_error": median_relative,
        "min_spearman": min_spearman,
        "max_median_relative_error": max_median_relative_error,
        "passed": bool(rho >= min_spearman and median_relative < max_median_relative_error),
    }
