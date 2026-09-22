from __future__ import annotations

import numpy as np
import torch

UNKNOWN_ID = -1


def normalize_operator(q: np.ndarray) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(value)
    if norm < 1e-12:
        raise ValueError("zero measurement operator")
    return value / norm


def operator_waveform(basis_mv: np.ndarray, q: np.ndarray) -> np.ndarray:
    value = normalize_operator(q)
    if basis_mv.ndim != 2 or basis_mv.shape[1] != len(value):
        raise ValueError("basis/operator dimensions disagree")
    return basis_mv @ value


def response_atoms(beats_mv: np.ndarray, q: np.ndarray, voltage_scale: float) -> np.ndarray:
    if voltage_scale <= 0 or not np.isfinite(voltage_scale):
        raise ValueError("voltage_scale must be positive and finite")
    values = np.asarray(beats_mv, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (256, 8):
        raise ValueError(f"expected physical beats with shape (beat,256,8), got {values.shape}")
    waveform = values @ normalize_operator(q) / voltage_scale
    derivative = np.gradient(waveform, axis=1)
    return np.stack((waveform, derivative), axis=-1)


def time_indexed_response_atoms(
    beats_mv: np.ndarray, q: np.ndarray, voltage_scale: float,
) -> np.ndarray:
    """Return joint phase-voltage-slope atoms ``[tau, x_q, dx_q/dtau]``.

    Unlike ``response_atoms``, the empirical distribution retains the joint
    association between location within the normalized cardiac cycle and the
    electrical response. ``tau`` is fixed and dimensionless on [-1, 1].
    """
    response = response_atoms(beats_mv, q, voltage_scale)
    tau = np.linspace(-1.0, 1.0, response.shape[1], dtype=np.float64)
    time = np.broadcast_to(tau[None, :, None], (*response.shape[:2], 1))
    return np.concatenate((time, response), axis=-1)


def projective_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Compute projective distance on RP^7: d_RP(q1, q2) = arccos(|q1^T q2|)."""
    v1 = normalize_operator(q1)
    v2 = normalize_operator(q2)
    inner = float(np.clip(np.abs(np.dot(v1, v2)), 0.0, 1.0))
    return float(np.arccos(inner))


def projective_response_atoms(beats_mv: np.ndarray, q: np.ndarray, voltage_scale: float) -> np.ndarray:
    """Compute projective response atoms g_q(t) = [q * x_q(t), q * dx_q/dt(t)] in R^16.
    
    Identically invariant under (q, x_q) -> (-q, -x_q) by construction:
      (-q) * x_{-q}(t) = (-q) * (-x_q(t)) = q * x_q(t).
    """
    if voltage_scale <= 0 or not np.isfinite(voltage_scale):
        raise ValueError("voltage_scale must be positive and finite")
    values = np.asarray(beats_mv, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (256, 8):
        raise ValueError(f"expected physical beats with shape (beat,256,8), got {values.shape}")
    q_norm = normalize_operator(q)
    waveform = values @ q_norm / voltage_scale
    derivative = np.gradient(waveform, axis=1)
    q_waveform = waveform[..., None] * q_norm[None, None, :]
    q_derivative = derivative[..., None] * q_norm[None, None, :]
    return np.concatenate((q_waveform, q_derivative), axis=-1)



def sample_sparse_operators(count: int, rng: np.random.Generator) -> np.ndarray:
    result = []
    for _ in range(count):
        q = rng.normal(size=8)
        keep = np.argpartition(np.abs(q), -3)[-3:]
        sparse = np.zeros(8)
        sparse[keep] = q[keep]
        result.append(normalize_operator(sparse))
    return np.stack(result)


def canonical_operators() -> np.ndarray:
    return np.eye(8, dtype=np.float64)


def derived_limb_operators() -> np.ndarray:
    coefficients = np.asarray(
        [
            [-1.0, 1.0, 0, 0, 0, 0, 0, 0],
            [-0.5, -0.5, 0, 0, 0, 0, 0, 0],
            [1.0, -0.5, 0, 0, 0, 0, 0, 0],
            [-0.5, 1.0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    return np.stack([normalize_operator(value) for value in coefficients])


def interpolation_operators() -> np.ndarray:
    """Nine frozen interior points from lead I to V2 (alpha=0.1,...,0.9)."""
    lead_i = canonical_operators()[0]
    lead_v2 = canonical_operators()[3]
    return np.stack(
        [normalize_operator((1.0 - alpha) * lead_i + alpha * lead_v2) for alpha in np.arange(0.1, 1.0, 0.1)]
    )


def _overlaps_up_to_sign(candidate: np.ndarray, reference: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.any(np.linalg.norm(reference - candidate, axis=1) <= tolerance)
        or np.any(np.linalg.norm(reference + candidate, axis=1) <= tolerance)
    )


def frozen_operator_banks(tolerance: float = 1e-8) -> dict[str, np.ndarray]:
    seen = np.concatenate(
        (canonical_operators(), sample_sparse_operators(100, np.random.default_rng(1701)))
    )
    dense = np.random.default_rng(1702).normal(size=(100, 8))
    dense = np.stack([normalize_operator(value) for value in dense])
    proposed = {
        "derived_limb": derived_limb_operators(),
        "dense": dense,
        "i_to_v2": interpolation_operators(),
    }
    unseen = {
        name: np.stack(
            [value for value in values if not _overlaps_up_to_sign(value, seen, tolerance)]
        )
        for name, values in proposed.items()
    }
    return {"seen": seen, **unseen}


def record_training_operators(record_id: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(record_id)]))
    canonical = canonical_operators()[rng.choice(8, size=4, replace=False)]
    return np.concatenate((canonical, sample_sparse_operators(4, rng)))


def build_training_vocabulary(operators: np.ndarray) -> np.ndarray:
    """Return a stable vocabulary containing only exact training operators."""
    if operators.ndim != 3 or operators.shape[-1] != 8:
        raise ValueError("operators must have shape (records,set,8)")
    return np.unique(np.ascontiguousarray(operators).reshape(-1, 8), axis=0)


def map_operator_ids(operators: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    """Map exact rows to training IDs and every unseen row to one UNK ID."""
    lookup = {row.tobytes(): index for index, row in enumerate(np.ascontiguousarray(vocabulary))}
    flat = np.ascontiguousarray(operators).reshape(-1, 8)
    ids = np.fromiter((lookup.get(row.tobytes(), UNKNOWN_ID) for row in flat), dtype=np.int64)
    return ids.reshape(operators.shape[:-1])


def sample_context_target_indices(
    batch_size: int, candidates: int, generator: torch.Generator, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly sample m in [1,6], with a distinct held-out target per row."""
    if candidates < 7:
        raise ValueError("Paper 7 requires at least seven candidate pairs")
    context_size = int(torch.randint(1, 7, (), generator=generator, device=device))
    order = torch.rand((batch_size, candidates), generator=generator, device=device).argsort(dim=1)
    return order[:, :context_size], order[:, context_size]


def nearest_known_operator(q: np.ndarray, vocabulary: np.ndarray) -> tuple[int, np.ndarray]:
    """Map unseen operator q to the nearest known operator in vocabulary by projective distance."""
    q_norm = normalize_operator(q)
    vocab_norm = np.stack([normalize_operator(v) for v in vocabulary])
    projections = np.abs(vocab_norm @ q_norm)
    best_idx = int(np.argmax(projections))
    return best_idx, vocab_norm[best_idx]


def analytic_pinv_reconstruct(
    y_observed: np.ndarray,
    q_context: np.ndarray,
    q_target: np.ndarray,
) -> np.ndarray:
    """Reconstruct target operator signal via pseudo-inverse of context operators.
    
    y_observed: shape (..., T, m)
    q_context: shape (m, 8)
    q_target: shape (8,)
    Returns:
      y_reconstructed: shape (..., T)
    """
    Q = np.asarray(q_context, dtype=np.float64)
    q_t = normalize_operator(q_target)
    pinv_Q_T = np.linalg.pinv(Q.T)
    x_reconstructed = y_observed @ pinv_Q_T  # (..., T, 8)
    return x_reconstructed @ q_t


def lmmse_reconstruct(
    y_observed: np.ndarray,
    q_context: np.ndarray,
    q_target: np.ndarray,
    sigma_x: np.ndarray,
    noise_var: float = 1e-4,
) -> np.ndarray:
    """Reconstruct target operator signal via LMMSE estimator given signal covariance.
    
    x_hat = y @ (Q Sigma_x Q^T + noise_var I)^-1 Q Sigma_x
    """
    Q = np.asarray(q_context, dtype=np.float64)
    q_t = normalize_operator(q_target)
    m = len(Q)
    sigma = np.asarray(sigma_x, dtype=np.float64)
    q_sigma_qT = Q @ sigma @ Q.T + noise_var * np.eye(m)
    weight = np.linalg.solve(q_sigma_qT, Q @ sigma)  # (m, 8)
    x_hat = y_observed @ weight  # (..., T, 8)
    return x_hat @ q_t

