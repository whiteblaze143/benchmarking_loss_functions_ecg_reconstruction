#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score

from repecg.paper08_tokens import (
    assign_certified_tokens,
    certified_clusters,
    cluster_map,
    exact_size_random_partitions,
    frequency_matched_random_partitions,
    odd_even_agreement,
    patient_block_simultaneous_bounds,
    phase_balanced_indices,
    unique_patient_support,
)


PRIMARY_K0 = 512
FALLBACK_K0 = 256
MINIMUM_PATIENTS = 40
MAX_UNSUPPORTED_FRACTION = 0.20
BOOTSTRAPS = 2000
ALPHA = 0.05
RANDOM_CONTROLS = 10
CELLS_PER_RECORD = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


class PhaseKMEMap:
    def __init__(self, fit_path: Path, device: torch.device):
        with np.load(fit_path) as fit:
            self.mean = torch.as_tensor(fit["whitening_mean"], device=device, dtype=torch.float32)
            self.components = torch.as_tensor(fit["whitening_components"], device=device, dtype=torch.float32)
            self.scales = torch.as_tensor(fit["whitening_scales"], device=device, dtype=torch.float32)
            self.landmarks = torch.as_tensor(fit["landmarks"], device=device, dtype=torch.float32)
            self.inverse_root = torch.as_tensor(fit["inverse_root"], device=device, dtype=torch.float32)
            self.c2 = float(fit["c2"])
        self.device = device

    def means(self, cells: list[np.ndarray]) -> np.ndarray:
        output = []
        with torch.inference_mode():
            for cell in cells:
                values = torch.as_tensor(cell, device=self.device, dtype=torch.float32)
                white = (values - self.mean) @ self.components * self.scales
                features = torch.rsqrt(torch.cdist(white, self.landmarks).square() + self.c2)
                output.append((features @ self.inverse_root).mean(0).cpu().numpy())
        return np.asarray(output, dtype=np.float32)


def _eligible(cache: Path) -> pd.DataFrame:
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest.get("kind") != "production_phase_cache":
        raise ValueError(f"not a production phase cache: {cache}")
    return pd.read_csv(cache / "qc.csv").query("eligible").reset_index(drop=True)


def _align_representations(path: Path, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    with np.load(path) as item:
        payload = {name: np.asarray(item[name]) for name in item.files}
    lookup = {int(ecg_id): index for index, ecg_id in enumerate(payload["ecg_ids"])}
    indices = np.asarray([lookup[int(ecg_id)] for ecg_id in frame.ecg_id], dtype=np.int64)
    return {name: values[indices] for name, values in payload.items()}


def _odd_even_states(
    cache: Path,
    frame: pd.DataFrame,
    phases: np.ndarray,
    mapping: PhaseKMEMap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    odd, even, patients, records, phase_ids = [], [], [], [], []
    for index, row in frame.iterrows():
        with np.load(cache / str(row.artifact)) as item:
            beats = np.asarray(item["beats"], dtype=np.float32)
        if len(beats) < 2:
            continue
        shaped = beats.reshape(len(beats), 16, 16, 8)
        odd_cells, even_cells = [], []
        admitted = []
        for phase in phases[index]:
            left = shaped[::2, phase].reshape(-1, 8)
            right = shaped[1::2, phase].reshape(-1, 8)
            if len(left) and len(right):
                odd_cells.append(left)
                even_cells.append(right)
                admitted.append(int(phase))
        if not admitted:
            continue
        odd.append(mapping.means(odd_cells))
        even.append(mapping.means(even_cells))
        patients.extend([int(row.patient_id)] * len(admitted))
        records.extend([int(row.ecg_id)] * len(admitted))
        phase_ids.extend(admitted)
    return (
        np.concatenate(odd), np.concatenate(even), np.asarray(patients),
        np.asarray(records), np.asarray(phase_ids),
    )


def _sample_states(
    representations: dict[str, np.ndarray], frame: pd.DataFrame, phases: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(len(frame))[:, None]
    features = representations["kernel"][rows, phases]
    return (
        features.reshape(-1, features.shape[-1]),
        np.repeat(frame.patient_id.to_numpy(dtype=np.int64), phases.shape[1]),
        np.repeat(frame.ecg_id.to_numpy(dtype=np.int64), phases.shape[1]),
        phases.reshape(-1),
    )


def _token_prototypes_and_radii(
    features: np.ndarray,
    base_codes: np.ndarray,
    clusters: list[tuple[int, ...]],
    base_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = cluster_map(clusters, base_count, allow_unassigned=True)
    token_ids = mapping[base_codes]
    prototypes, radii = [], []
    for token in range(len(clusters)):
        members = features[token_ids == token]
        prototype = members.mean(0)
        prototypes.append(prototype)
        radii.append(np.quantile(np.square(members - prototype).sum(1), 0.95))
    return np.asarray(prototypes), np.asarray(radii), mapping


def _record_agreements(
    odd_tokens: np.ndarray,
    reference_tokens: np.ndarray,
    record_ids: np.ndarray,
    patient_ids: np.ndarray,
    token_count: int,
) -> pd.DataFrame:
    record_scores, record_patients = [], []
    for record in np.unique(record_ids):
        mask = record_ids == record
        odd = np.bincount(odd_tokens[mask] + 1, minlength=token_count + 1)
        even = np.bincount(reference_tokens[mask] + 1, minlength=token_count + 1)
        record_scores.append(odd_even_agreement(odd, even))
        record_patients.append(patient_ids[mask][0])
    return pd.DataFrame({"patient_id": record_patients, "score": record_scores})


def _patient_equal_mean(scores: pd.DataFrame) -> float:
    return float(scores.groupby("patient_id").score.mean().mean())


def _patient_bootstrap_difference(
    left: pd.DataFrame, right: pd.DataFrame, *, bootstraps: int, seed: int
) -> tuple[float, np.ndarray, list[float]]:
    patient = left.groupby("patient_id").score.mean().to_frame("score_left").join(
        right.groupby("patient_id").score.mean().rename("score_right"), how="inner"
    )
    difference = (patient.score_left - patient.score_right).to_numpy()
    if not len(difference):
        raise ValueError("no patient scores for odd/even bootstrap")
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(difference), size=(bootstraps, len(difference)))
    draws = difference[sample].mean(1)
    return float(difference.mean()), draws.astype(np.float32), [
        float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    ]


def _different_patient_matched_phase_reference(
    tokens: np.ndarray,
    record_ids: np.ndarray,
    patient_ids: np.ndarray,
    phase_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic different-patient reference at the same phase."""
    result = np.empty_like(tokens)
    rng = np.random.default_rng(seed)
    for phase in np.unique(phase_ids):
        rows = np.flatnonzero(phase_ids == phase)
        for row in rows:
            candidates = rows[patient_ids[rows] != patient_ids[row]]
            if not len(candidates):
                raise RuntimeError(f"no different-patient reference for phase {phase}")
            result[row] = tokens[candidates[rng.integers(len(candidates))]]
    return result


def _unknown_rates(tokens: np.ndarray, labels: np.ndarray) -> tuple[float, list[float], dict[str, dict[str, float]]]:
    names = ("NORM", "MI", "STTC", "CD", "HYP")
    unknown = tokens < 0
    if labels.shape[1] != len(names):
        raise ValueError(f"expected five PTB-XL superclass labels, got {labels.shape[1]}")
    by_diagnosis = {}
    for index, name in enumerate(names):
        positive = labels[:, index] > 0
        by_diagnosis[name] = {
            "positive": float(np.mean(unknown[positive])) if np.any(positive) else float("nan"),
            "negative": float(np.mean(unknown[~positive])) if np.any(~positive) else float("nan"),
        }
    return float(np.mean(unknown)), np.mean(unknown, axis=0).tolist(), by_diagnosis


def _phase_topology(token_phase_counts: np.ndarray, epsilon: float = 0.05) -> dict[str, object]:
    """Summarize phase support of resolved tokens; row zero is UNK and excluded."""
    counts = np.asarray(token_phase_counts[1:], dtype=np.float64)
    conditional = counts / counts.sum(1, keepdims=True)
    support = conditional > epsilon
    log_probability = np.zeros_like(conditional)
    positive = conditional > 0
    log_probability[positive] = np.log2(conditional[positive])
    entropy = -(conditional * log_probability).sum(1)
    diameters, nonadjacent = [], []
    for phases in support:
        present = np.flatnonzero(phases)
        if len(present) < 2:
            diameters.append(0)
            nonadjacent.append(False)
            continue
        distances = np.abs(present[:, None] - present[None, :])
        circular = np.minimum(distances, 16 - distances)
        diameters.append(int(circular.max()))
        nonadjacent.append(bool(np.any(circular > 1)))
    return {
        "epsilon": epsilon,
        "phase_entropy_bits": entropy.tolist(),
        "phase_support_counts": support.sum(1).astype(int).tolist(),
        "circular_phase_diameter": diameters,
        "fraction_single_phase": float(np.mean(support.sum(1) == 1)),
        "fraction_multi_phase": float(np.mean(support.sum(1) >= 2)),
        "fraction_nonadjacent_phase": float(np.mean(nonadjacent)),
        "median_circular_phase_diameter": float(np.median(diameters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--phase-kme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-records", type=int)
    parser.add_argument("--pilot-k0", type=int, default=32)
    parser.add_argument("--pilot-bootstraps", type=int, default=20)
    parser.add_argument("--pilot-minimum-patients", type=int, default=3)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    parser.add_argument("--bootstrap-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-workers", type=int, default=1)
    args = parser.parse_args()

    train = _eligible(args.train_cache)
    pseudo_test = _eligible(args.validation_cache)
    if set(train.fold.unique()) != set(range(1, 8)):
        raise ValueError("Paper 8 development cache must contain exactly folds 1--7")
    if set(pseudo_test.fold.unique()) != {8}:
        raise ValueError("Paper 8 pseudo-test cache must contain exactly fold 8")
    if args.pilot_records is not None:
        train = train.groupby("fold", group_keys=False).head(args.pilot_records).reset_index(drop=True)
        pseudo_test = pseudo_test.iloc[: min(args.pilot_records, len(pseudo_test))].reset_index(drop=True)
    construction_frame = train.query("fold in [1, 2, 3, 4, 5, 6]").reset_index(drop=True)
    selection_frame = train.query("fold == 7").reset_index(drop=True)
    train_rep = _align_representations(args.phase_kme / "representation_train.npz", train)
    pseudo_test_rep = _align_representations(args.phase_kme / "representation_val.npz", pseudo_test)
    construction_rep = _align_representations(args.phase_kme / "representation_train.npz", construction_frame)
    selection_rep = _align_representations(args.phase_kme / "representation_train.npz", selection_frame)
    construction_phases = phase_balanced_indices(
        construction_frame.ecg_id.to_numpy(), CELLS_PER_RECORD, args.seed
    )
    selection_phases = phase_balanced_indices(
        selection_frame.ecg_id.to_numpy(), CELLS_PER_RECORD, args.seed
    )
    construction_features, construction_patients, _, _ = _sample_states(
        construction_rep, construction_frame, construction_phases
    )
    selection_features, selection_patients, _, selection_phase_ids = _sample_states(
        selection_rep, selection_frame, selection_phases
    )
    if set(construction_frame.patient_id).intersection(set(selection_frame.patient_id)):
        raise RuntimeError("patient leakage between fold 1--6 construction and fold 7 selection")

    pilot = args.pilot_records is not None
    k_candidates = [args.pilot_k0] if pilot else [PRIMARY_K0, FALLBACK_K0]
    bootstraps = args.pilot_bootstraps if pilot else args.bootstraps
    minimum_patients = args.pilot_minimum_patients if pilot else MINIMUM_PATIENTS
    selected = None
    support_audit = []
    for k0 in k_candidates:
        model = MiniBatchKMeans(
            n_clusters=k0, batch_size=4096, n_init=3, random_state=args.seed
        ).fit(construction_features[:200_000])
        construction_codes = model.predict(construction_features)
        support = unique_patient_support(construction_codes, construction_patients, k0)
        unsupported_fraction = float(np.mean(support < minimum_patients))
        support_audit.append({
            "k0": int(k0),
            "supported_codes": int(np.sum(support >= minimum_patients)),
            "unsupported_codes": int(np.sum(support < minimum_patients)),
            "unsupported_fraction": unsupported_fraction,
            "minimum_support": int(support.min()),
            "median_support": float(np.median(support)),
            "maximum_support": int(support.max()),
        })
        selected = (k0, model, support, unsupported_fraction)
        if pilot or unsupported_fraction <= MAX_UNSUPPORTED_FRACTION:
            break
    assert selected is not None
    k0, base_model, support, unsupported_fraction = selected
    if not pilot and unsupported_fraction > MAX_UNSUPPORTED_FRACTION:
        raise RuntimeError(
            f"Paper 8 support gate failed at K0={k0}: {unsupported_fraction:.3f} unsupported"
        )

    construction_codes = base_model.predict(construction_features)
    mapping = PhaseKMEMap(args.phase_kme / "kernel_fit.npz", torch.device(args.device))
    odd, even, split_patients, split_records, split_phases = _odd_even_states(
        args.train_cache, selection_frame, selection_phases, mapping
    )
    self_distances = np.square(odd - even).sum(1)
    delta = float(np.quantile(self_distances, 0.95))
    delta_sensitivity = {
        "q90": float(np.quantile(self_distances, 0.90)),
        "q95": delta,
        "q99": float(np.quantile(self_distances, 0.99)),
    }
    point, draws, upper = patient_block_simultaneous_bounds(
        construction_features, construction_codes, construction_patients, k0,
        bootstraps=bootstraps, alpha=ALPHA, seed=args.seed, device=args.bootstrap_device,
        workers=args.bootstrap_workers,
    )
    sensitivity_clusters = {}
    for name, threshold in delta_sensitivity.items():
        value = certified_clusters(upper, threshold, support, minimum_patients)
        clusters_for_delta, unresolved_for_delta = value
        if not clusters_for_delta:
            raise RuntimeError(f"support/equivalence rules produced no tokens at {name}")
        sensitivity_clusters[name] = (clusters_for_delta, unresolved_for_delta)
    clusters, unresolved = sensitivity_clusters["q95"]
    _, _, merge_tree = certified_clusters(
        upper, delta, support, minimum_patients, return_trace=True
    )
    prototypes, radii, base_to_token = _token_prototypes_and_radii(
        construction_features, construction_codes, clusters, k0
    )
    odd_tokens = assign_certified_tokens(
        odd, base_model.cluster_centers_, base_to_token, prototypes, radii
    )
    even_tokens = assign_certified_tokens(
        even, base_model.cluster_centers_, base_to_token, prototypes, radii
    )
    same_scores = _record_agreements(
        odd_tokens, even_tokens, split_records, split_patients, len(clusters)
    )
    between_tokens = _different_patient_matched_phase_reference(
        even_tokens, split_records, split_patients, split_phases, args.seed + 11
    )
    between_scores = _record_agreements(
        odd_tokens, between_tokens, split_records, split_patients, len(clusters)
    )
    token_probability = np.bincount(even_tokens + 1, minlength=len(clusters) + 1).astype(np.float64)
    token_probability /= token_probability.sum()
    random_tokens = np.random.default_rng(args.seed + 12).choice(
        np.arange(-1, len(clusters)), size=len(even_tokens), p=token_probability
    )
    random_scores = _record_agreements(
        odd_tokens, random_tokens, split_records, split_patients, len(clusters)
    )
    stability_draws = {}
    stability_audit = {"same_record_odd_even": _patient_equal_mean(same_scores)}
    for name, reference, offset in (
        ("different_patient_matched_phase", between_scores, 13),
        ("random_token_assignment", random_scores, 14),
    ):
        difference, difference_draws, interval = _patient_bootstrap_difference(
            same_scores, reference, bootstraps=bootstraps, seed=args.seed + offset
        )
        stability_draws[name] = difference_draws
        stability_audit[name] = {
            "patient_equal_agreement": _patient_equal_mean(reference),
            "same_minus_reference": difference,
            "patient_bootstrap_95ci": interval,
        }

    selection_all = selection_rep["kernel"].reshape(-1, selection_rep["kernel"].shape[-1])
    pseudo_test_all = pseudo_test_rep["kernel"].reshape(-1, pseudo_test_rep["kernel"].shape[-1])
    selection_eq = assign_certified_tokens(
        selection_all, base_model.cluster_centers_, base_to_token, prototypes, radii
    ).reshape(len(selection_frame), 16)
    pseudo_test_eq = assign_certified_tokens(
        pseudo_test_all, base_model.cluster_centers_, base_to_token, prototypes, radii
    ).reshape(len(pseudo_test), 16)
    selection_unknown, selection_unknown_phase, selection_unknown_diagnosis = _unknown_rates(
        selection_eq, selection_rep["labels"]
    )
    pseudo_test_unknown, pseudo_test_unknown_phase, pseudo_test_unknown_diagnosis = _unknown_rates(
        pseudo_test_eq, pseudo_test_rep["labels"]
    )
    valid = selection_eq >= 0
    phase_grid = np.broadcast_to(np.arange(16), selection_eq.shape)
    phase_nmi = float(normalized_mutual_info_score(phase_grid[valid], selection_eq[valid]))
    token_phase_counts = np.zeros((len(clusters) + 1, 16), dtype=np.int64)
    for phase in range(16):
        token_phase_counts[:, phase] = np.bincount(
            selection_eq[:, phase] + 1, minlength=len(clusters) + 1
        )
    topology = _phase_topology(token_phase_counts)

    cluster_sizes = np.asarray([len(cluster) for cluster in clusters])
    eligible_codes = np.flatnonzero(support >= minimum_patients)
    random_partitions = exact_size_random_partitions(
        cluster_sizes, k0, RANDOM_CONTROLS, args.seed + 1, eligible_codes=eligible_codes
    )
    random_maps = np.stack([
        cluster_map(partition, k0, allow_unassigned=True) for partition in random_partitions
    ])
    code_frequencies = np.bincount(construction_codes, minlength=k0)
    target_cluster_frequencies = np.asarray([
        code_frequencies[list(cluster)].sum() for cluster in clusters
    ])
    frequency_partitions, frequency_scores = frequency_matched_random_partitions(
        cluster_sizes, eligible_codes, code_frequencies, target_cluster_frequencies,
        RANDOM_CONTROLS, 100, args.seed + 3,
    )
    frequency_maps = np.stack([
        cluster_map(partition, k0, allow_unassigned=True)
        for partition in frequency_partitions
    ])
    size_kmeans = MiniBatchKMeans(
        n_clusters=len(clusters), batch_size=4096, n_init=3, random_state=args.seed + 2
    ).fit(construction_features[:200_000])

    def encode(frame: pd.DataFrame, representation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        continuous = representation["kernel"]
        flat = continuous.reshape(-1, continuous.shape[-1])
        return {
            "continuous": continuous,
            "equivalence_token_ids": assign_certified_tokens(
                flat, base_model.cluster_centers_, base_to_token, prototypes, radii
            ).reshape(len(frame), 16),
            "fine_kmeans_ids": base_model.predict(flat).reshape(len(frame), 16),
            "size_kmeans_ids": size_kmeans.predict(flat).reshape(len(frame), 16),
            "labels": representation["labels"],
            "ecg_ids": representation["ecg_ids"],
            "patient_ids": representation["patient_ids"],
        }

    train_payload = encode(construction_frame, construction_rep)
    selection_payload = encode(selection_frame, selection_rep)
    pseudo_test_payload = encode(pseudo_test, pseudo_test_rep)
    if not np.array_equal(selection_payload["equivalence_token_ids"], selection_eq):
        raise RuntimeError("selection representation assignment was not deterministic")
    if not np.array_equal(pseudo_test_payload["equivalence_token_ids"], pseudo_test_eq):
        raise RuntimeError("pseudo-test representation assignment was not deterministic")
    sensitivity = {}
    for name, (clusters_for_delta, unresolved_for_delta) in sensitivity_clusters.items():
        prototypes_for_delta, radii_for_delta, map_for_delta = _token_prototypes_and_radii(
            construction_features, construction_codes, clusters_for_delta, k0
        )
        selected_tokens = assign_certified_tokens(
            selection_all, base_model.cluster_centers_, map_for_delta,
            prototypes_for_delta, radii_for_delta,
        ).reshape(len(selection_frame), 16)
        odd_for_delta = assign_certified_tokens(
            odd, base_model.cluster_centers_, map_for_delta, prototypes_for_delta, radii_for_delta
        )
        even_for_delta = assign_certified_tokens(
            even, base_model.cluster_centers_, map_for_delta, prototypes_for_delta, radii_for_delta
        )
        scores_for_delta = _record_agreements(
            odd_for_delta, even_for_delta, split_records, split_patients, len(clusters_for_delta)
        )
        sensitivity[name] = {
            "final_vocabulary_size": len(clusters_for_delta),
            "unresolved_base_codes": int(len(unresolved_for_delta)),
            "cluster_size_histogram": np.bincount(
                np.asarray([len(cluster) for cluster in clusters_for_delta])
            ).tolist(),
            "selection_unknown_rate": float(np.mean(selected_tokens < 0)),
            "patient_equal_odd_even_agreement": _patient_equal_mean(scores_for_delta),
        }
    safeguard_passed = bool(selection_unknown <= 0.20 and phase_nmi < 0.90)
    audit = {
        "passed": bool(not pilot and safeguard_passed),
        "engineering_pipeline_passed": True,
        "safeguards_would_pass": safeguard_passed,
        "selection_unknown_rate": selection_unknown,
        "maximum_unknown_rate": 0.20,
        "phase_nmi": phase_nmi,
        "maximum_phase_nmi": 0.90,
        "patient_equal_odd_even_agreement": stability_audit["same_record_odd_even"],
    }
    status = "pilot_only" if pilot else "complete" if audit["passed"] else "failed_safeguard"
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        args.output / "vocabulary_certificate.npz",
        base_centers=base_model.cluster_centers_.astype(np.float32),
        patient_support=support,
        self_distances=self_distances,
        pair_point=point,
        pair_bootstrap=draws,
        simultaneous_upper=upper,
        base_to_token=base_to_token,
        token_prototypes=prototypes.astype(np.float32),
        token_radii_squared=radii.astype(np.float32),
        random_merge_maps=random_maps,
        frequency_matched_random_merge_maps=frequency_maps,
        frequency_match_scores=frequency_scores,
        size_matched_kmeans_centers=size_kmeans.cluster_centers_.astype(np.float32),
        unresolved_base_codes=unresolved,
        token_phase_counts=token_phase_counts,
        odd_even_same_record_scores=same_scores.score.to_numpy(dtype=np.float32),
        odd_even_score_patient_ids=same_scores.patient_id.to_numpy(dtype=np.int64),
        odd_even_different_patient_delta_draws=stability_draws["different_patient_matched_phase"],
        odd_even_random_token_delta_draws=stability_draws["random_token_assignment"],
    )
    _atomic_npz(
        args.output / "representation_train.npz", **train_payload,
    )
    _atomic_npz(
        args.output / "representation_selection.npz", **selection_payload,
    )
    _atomic_npz(args.output / "representation_pseudotest.npz", **pseudo_test_payload)
    manifest = {
        "kind": "paper08_equivalence_token_representations",
        "status": status,
        "audit": audit,
        "folds": {
            "construction": [1, 2, 3, 4, 5, 6],
            "calibration_and_selection": [7],
            "pseudo_test": [8],
        },
        "fold_firewall": {
            "fold8_used_for_fit_or_selection": False,
            "prototypes_and_radii_fit_folds": [1, 2, 3, 4, 5, 6],
            "support_and_ucb_construction_folds": [1, 2, 3, 4, 5, 6],
            "delta_calibration_and_model_selection_fold": 7,
        },
        "k0": int(k0),
        "k0_fallback_rule": {
            "primary": PRIMARY_K0, "fallback": FALLBACK_K0,
            "minimum_unique_patients": minimum_patients,
            "maximum_unsupported_fraction": MAX_UNSUPPORTED_FRACTION,
        },
        "k0_support_audit": support_audit,
        "unsupported_fraction": unsupported_fraction,
        "unresolved_base_codes": len(unresolved),
        "final_vocabulary_size": len(clusters),
        "supported_base_codes": int(k0 - len(unresolved)),
        "certified_merges": int(k0 - len(unresolved) - len(clusters)),
        "cluster_size_histogram": np.bincount(cluster_sizes).tolist(),
        "merge_tree": merge_tree,
        "delta": delta_sensitivity,
        "delta_sensitivity": sensitivity,
        "distance": "squared_euclidean_in_common_nystrom_phase_kme_coordinates",
        "bootstrap": {
            "unit": "patient", "replicates": bootstraps, "alpha": ALPHA,
            "seed": args.seed, "device": args.bootstrap_device,
            "workers": args.bootstrap_workers,
        },
        "random_controls": {
            "count": RANDOM_CONTROLS,
            "exact_cluster_size_multiset": True,
            "frequency_matched_secondary_count": RANDOM_CONTROLS,
            "frequency_match_candidates_per_control": 100,
        },
        "assignment_radius_quantile": 0.95,
        "selection_unknown_rate_by_phase": selection_unknown_phase,
        "selection_unknown_rate_by_diagnosis": selection_unknown_diagnosis,
        "pseudo_test_unknown_rate": pseudo_test_unknown,
        "pseudo_test_unknown_rate_by_phase": pseudo_test_unknown_phase,
        "pseudo_test_unknown_rate_by_diagnosis": pseudo_test_unknown_diagnosis,
        "odd_even_stability": stability_audit,
        "phase_topology": topology,
        "phase_balanced_cells_per_record": CELLS_PER_RECORD,
        "records": {
            "train": len(construction_frame),
            "selection": len(selection_frame),
            "pseudo_test": len(pseudo_test),
        },
        "source": {
            "phase_kme_manifest_sha256": _sha256(args.phase_kme / "manifest.json"),
            "train_cache_manifest_sha256": _sha256(args.train_cache / "manifest.json"),
            "validation_cache_manifest_sha256": _sha256(args.validation_cache / "manifest.json"),
        },
        "patient_hashes": {
            "construction": _array_sha256(np.sort(construction_frame.patient_id.unique())),
            "calibration_and_selection": _array_sha256(np.sort(selection_frame.patient_id.unique())),
            "pseudo_test": _array_sha256(np.sort(pseudo_test.patient_id.unique())),
        },
        "software": {
            "builder_sha256": _sha256(Path(__file__)),
            "vocabulary_module_sha256": _sha256(
                Path(__file__).resolve().parents[2] / "src/repecg/paper08_tokens/vocabulary.py"
            ),
        },
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    if status == "failed_safeguard":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
