from __future__ import annotations

import hashlib

import numpy as np

from .equivalence import complete_linkage_merge_with_trace, simultaneous_upper_bounds


def patient_construction_mask(
    patient_ids: np.ndarray, seed: int, construction_fraction: float = 0.8
) -> np.ndarray:
    """Deterministically split unique patients without using labels or phases."""
    if not 0.0 < construction_fraction < 1.0:
        raise ValueError("construction_fraction must be in (0,1)")
    result = np.empty(len(patient_ids), dtype=bool)
    boundary = int(construction_fraction * 10_000)
    for index, patient_id in enumerate(np.asarray(patient_ids).reshape(-1)):
        digest = hashlib.sha256(f"paper08-patient:{seed}:{int(patient_id)}".encode()).digest()
        result[index] = int.from_bytes(digest[:8], "little") % 10_000 < boundary
    return result


def phase_balanced_indices(record_ids: np.ndarray, per_record: int, seed: int) -> np.ndarray:
    """Choose the same global count from every phase, up to rounding."""
    if not 1 <= per_record <= 16:
        raise ValueError("per_record must be in 1..16")
    ids = np.asarray(record_ids).reshape(-1)
    rows = np.empty((len(ids), per_record), dtype=np.int64)
    keys = [
        hashlib.sha256(f"paper08-phases:{seed}:{int(record_id)}".encode()).digest()
        for record_id in ids
    ]
    order = np.asarray(sorted(range(len(ids)), key=lambda index: (keys[index], int(ids[index]))))
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(len(ids))
    if per_record == 8:
        pattern = np.asarray([0, 2, 4, 6, 9, 11, 13, 15])
    else:
        pattern = np.floor(np.arange(per_record) * 16 / per_record).astype(np.int64)
    for index in range(len(ids)):
        offset = int(rank[index]) % 16
        rows[index] = (offset + pattern) % 16
    return rows


def pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(count, k=1)


def squared_pair_distances(means: np.ndarray) -> np.ndarray:
    values = np.asarray(means, dtype=np.float64)
    squared_norm = np.square(values).sum(axis=1)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * values @ values.T
    return np.maximum(distances, 0.0)


def _weighted_code_means(
    features: np.ndarray,
    codes: np.ndarray,
    occurrence_weights: np.ndarray,
    code_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((code_count, features.shape[1]), dtype=np.float64)
    counts = np.bincount(codes, weights=occurrence_weights, minlength=code_count)
    np.add.at(sums, codes, features * occurrence_weights[:, None])
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    return means, counts


def patient_block_simultaneous_bounds(
    features: np.ndarray,
    codes: np.ndarray,
    patient_ids: np.ndarray,
    code_count: int,
    *,
    bootstraps: int,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Patient-clustered basic-bootstrap simultaneous UCBs for all code pairs."""
    features = np.asarray(features, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    patient_ids = np.asarray(patient_ids)
    if not (len(features) == len(codes) == len(patient_ids)):
        raise ValueError("features, codes, and patient_ids must align")
    unique_patients, patient_inverse = np.unique(patient_ids, return_inverse=True)
    means, counts = _weighted_code_means(features, codes, np.ones(len(codes)), code_count)
    left, right = pair_indices(code_count)
    point = squared_pair_distances(means)[left, right]
    draws = np.empty((bootstraps, len(point)), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for replicate in range(bootstraps):
        sampled = rng.integers(0, len(unique_patients), size=len(unique_patients))
        patient_weights = np.bincount(sampled, minlength=len(unique_patients)).astype(np.float64)
        boot_means, _ = _weighted_code_means(
            features, codes, patient_weights[patient_inverse], code_count
        )
        draws[replicate] = squared_pair_distances(boot_means)[left, right]
    upper = simultaneous_upper_bounds(point, draws, alpha=alpha)
    matrix = np.full((code_count, code_count), np.inf, dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    matrix[left, right] = upper
    matrix[right, left] = upper
    return point, draws, matrix


def unique_patient_support(codes: np.ndarray, patient_ids: np.ndarray, code_count: int) -> np.ndarray:
    support = np.zeros(code_count, dtype=np.int64)
    for code in range(code_count):
        support[code] = len(np.unique(patient_ids[codes == code]))
    return support


def certified_clusters(
    upper: np.ndarray,
    delta: float,
    patient_support: np.ndarray,
    minimum_patients: int,
    *,
    return_trace: bool = False,
) -> tuple[list[tuple[int, ...]], np.ndarray] | tuple[list[tuple[int, ...]], np.ndarray, list[dict[str, object]]]:
    """Merge supported codes; preserve unsupported codes as unresolved singletons."""
    supported = np.asarray(patient_support) >= minimum_patients
    restricted = np.asarray(upper, dtype=np.float64).copy()
    unresolved = np.flatnonzero(~supported)
    restricted[unresolved, :] = np.inf
    restricted[:, unresolved] = np.inf
    restricted[unresolved, unresolved] = 0.0
    clusters, trace = complete_linkage_merge_with_trace(restricted, delta)
    resolved_clusters = [cluster for cluster in clusters if supported[list(cluster)].all()]
    if return_trace:
        return resolved_clusters, unresolved, trace
    return resolved_clusters, unresolved


def frequency_matched_random_partitions(
    cluster_sizes: np.ndarray,
    eligible_codes: np.ndarray,
    code_frequencies: np.ndarray,
    target_cluster_frequencies: np.ndarray,
    repeats: int,
    candidates_per_repeat: int,
    seed: int,
) -> tuple[list[list[tuple[int, ...]]], np.ndarray]:
    """Pick exact-size random partitions closest to the target usage profile."""
    rng = np.random.default_rng(seed)
    target = np.sort(np.log1p(np.asarray(target_cluster_frequencies, dtype=np.float64)))
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    eligible = np.asarray(eligible_codes, dtype=np.int64)
    results: list[list[tuple[int, ...]]] = []
    scores = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        best_partition = None
        best_score = np.inf
        for _ in range(candidates_per_repeat):
            order = rng.permutation(eligible)
            start = 0
            partition = []
            totals = []
            for size in sizes:
                cluster = tuple(sorted(order[start:start + size].tolist()))
                partition.append(cluster)
                totals.append(code_frequencies[list(cluster)].sum())
                start += int(size)
            score = float(np.mean(np.square(np.sort(np.log1p(totals)) - target)))
            if score < best_score:
                best_partition, best_score = partition, score
        assert best_partition is not None
        results.append(best_partition)
        scores[repeat] = best_score
    return results, scores


def exact_size_random_partitions(
    cluster_sizes: np.ndarray,
    base_count: int,
    repeats: int,
    seed: int,
    eligible_codes: np.ndarray | None = None,
) -> list[list[tuple[int, ...]]]:
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    eligible = np.arange(base_count) if eligible_codes is None else np.asarray(eligible_codes)
    if np.any(sizes <= 0) or sizes.sum() != len(eligible):
        raise ValueError("cluster sizes must be positive and sum to eligible code count")
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(repeats):
        order = rng.permutation(eligible)
        start = 0
        partition = []
        for size in sizes:
            partition.append(tuple(sorted(order[start:start + size].tolist())))
            start += int(size)
        results.append(partition)
    return results


def cluster_map(
    clusters: list[tuple[int, ...]], base_count: int, *, allow_unassigned: bool = False
) -> np.ndarray:
    mapping = np.full(base_count, -1, dtype=np.int64)
    for token, cluster in enumerate(clusters):
        mapping[list(cluster)] = token
    if np.any(mapping < 0) and not allow_unassigned:
        raise ValueError("clusters do not partition every base code")
    return mapping


def assign_with_unknown(
    features: np.ndarray, prototypes: np.ndarray, radii_squared: np.ndarray
) -> np.ndarray:
    distances = (
        np.square(features).sum(1)[:, None]
        + np.square(prototypes).sum(1)[None, :]
        - 2.0 * features @ prototypes.T
    )
    distances = np.maximum(distances, 0.0)
    nearest = distances.argmin(1)
    accepted = distances[np.arange(len(features)), nearest] <= radii_squared[nearest]
    return np.where(accepted, nearest, -1)


def assign_certified_tokens(
    features: np.ndarray,
    base_centers: np.ndarray,
    base_to_token: np.ndarray,
    token_prototypes: np.ndarray,
    token_radii_squared: np.ndarray,
) -> np.ndarray:
    """Reject unsupported base regions and states outside token support radii."""
    base_distances = (
        np.square(features).sum(1)[:, None]
        + np.square(base_centers).sum(1)[None, :]
        - 2.0 * features @ base_centers.T
    )
    base = np.maximum(base_distances, 0.0).argmin(1)
    proposed = base_to_token[base]
    result = np.full(len(features), -1, dtype=np.int64)
    eligible = proposed >= 0
    if not np.any(eligible):
        return result
    rows = np.flatnonzero(eligible)
    tokens = proposed[eligible]
    distances = np.square(features[eligible] - token_prototypes[tokens]).sum(1)
    accepted = distances <= token_radii_squared[tokens]
    result[rows[accepted]] = tokens[accepted]
    return result
