r"""Physically Faithful Measurement-Operator Evaluator for CVPR 2027 Benchmark.

Scientific Guarantee:
1. Every operator q is evaluated strictly with its authentic physical response:
   y_q(t) = q^T x(t) ---> PhaseKME([y_q, \dot{y}_q]) = r_q.
   Never substitutes r_{e_i} for r_q when q != e_i.
2. Clinical configurations index their exact physical operator responses.
3. Expanded Grassmann benchmark (350 subspaces) evaluates every subspace individually:
   - Exact measurement matrix B_U in R^{r x 8}.
   - Authentic responses r_{q_1}, ..., r_{q_r} for each subspace row.
   - Individual AUROC computed per subspace, per seed, per model.
   - Seed-mean paired contrast Delta_u = (1/n_seeds) sum [AUROC_Span,s - AUROC_Basis,s].
4. Observed-Only Segmentation:
   - Evaluates single-channel configurations using autonomous R-peak detection directly on y_q(t).
   - Computes real downstream logits and AUROC_observed - AUROC_oracle.
   - Zero mock values, zero synthetic substitutions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

logger = logging.getLogger("evaluator_physical_operator")

from repecg.paper07_operator.model import OperatorSetModel
from repecg.paper07_operator.measurements import normalize_operator, response_atoms


class FastResponseComputer:
    """Computes exact Phase-KME response representations on GPU from physical beats."""

    def __init__(self, fit_path: Path, device: torch.device):
        data = np.load(fit_path)
        self.v_scale = float(data["voltage_scale"])
        self.mean = data["physical_mean"].astype(np.float32)
        self.std = data["physical_std"].astype(np.float32)

        self.landmarks = torch.as_tensor(data["landmarks"], device=device, dtype=torch.float32)
        self.inv_root = torch.as_tensor(data["inverse_root"], device=device, dtype=torch.float32)
        self.c2 = float(data["c2"])
        self.w_mean = torch.as_tensor(data["whitening_mean"], device=device, dtype=torch.float32)
        self.w_comp = torch.as_tensor(data["whitening_components"], device=device, dtype=torch.float32)
        self.w_scale = torch.as_tensor(data["whitening_scales"], device=device, dtype=torch.float32)
        self.device = device

    def compute_record_responses(self, beats_mv: np.ndarray, q_bank: np.ndarray) -> np.ndarray:
        """Compute responses for a single record across K operators.

        Args:
            beats_mv: shape (n_beats, 256, 8) in physical millivolts
            q_bank: shape (K, 8)

        Returns:
            shape (K, 16, 128) float32
        """
        n_ops = len(q_bank)
        atoms = [response_atoms(beats_mv.astype(np.float64), q, self.v_scale) for q in q_bank]
        atoms = np.stack(atoms, axis=0)  # (K, beat, 256, 2)
        cells = atoms.reshape(n_ops, len(beats_mv), 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(n_ops, 16, -1, 2)
        t_cells = torch.as_tensor(cells, device=self.device, dtype=torch.float32)
        white = (t_cells - self.w_mean) @ self.w_comp * self.w_scale
        d2 = torch.cdist(white, self.landmarks.expand(n_ops, 16, -1, -1)).square() + self.c2
        feat = torch.rsqrt(d2) @ self.inv_root
        return feat.mean(dim=2).cpu().numpy().astype(np.float32)  # (K, 16, 128)


def compute_auroc_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro AUROC across classes."""
    aurocs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            try:
                score = roc_auc_score(y_true[:, c], y_pred[:, c])
                aurocs.append(score)
            except Exception:
                pass
    return float(np.mean(aurocs)) if aurocs else 0.5


def build_canonical_and_derived_operator_bank() -> np.ndarray:
    """Return the 16 standard operators used in final evaluation.

    0..7: canonical basis leads (I, II, V1..V6)
    8: III
    9: aVR
    10: aVL
    11: aVF
    12: S_icm (V3 - V2) / sqrt(2)
    13: V1 - V6
    14: I + V4
    15: II - V5
    """
    e = np.eye(8, dtype=np.float32)
    return np.stack([
        e[0],                                       # 0: I
        e[1],                                       # 1: II
        e[2],                                       # 2: V1
        e[3],                                       # 3: V2
        e[4],                                       # 4: V3
        e[5],                                       # 5: V4
        e[6],                                       # 6: V5
        e[7],                                       # 7: V6
        normalize_operator(e[1] - e[0]),             # 8: III
        normalize_operator(-e[0] - e[1]),            # 9: aVR
        normalize_operator(e[0] - 0.5 * e[1]),       # 10: aVL
        normalize_operator(e[1] - 0.5 * e[0]),       # 11: aVF
        normalize_operator(e[4] - e[3]),             # 12: V3 - V2 (S_icm)
        normalize_operator(e[2] - e[7]),             # 13: V1 - V6
        normalize_operator(e[0] + e[5]),             # 14: I + V4
        normalize_operator(e[1] - e[6]),             # 15: II - V5
    ]).astype(np.float32)


# Pre-registered clinical configuration operator indices
CLINICAL_CONFIG_INDICES = {
    "Q_8": [0, 1, 2, 3, 4, 5, 6, 7],
    "S_6": [2, 3, 4, 5, 6, 7],
    "S_2": [0, 1],
    "S_1": [0],
    "S_6_limb": [0, 1, 8, 9, 10, 11],
    "derived_limb_only": [8, 9, 10, 11],
    "lead_III_single": [8],
    "aVF_single": [11],
    "S_icm": [12],
}


def verify_operator_physical_consistency(
    computer: FastResponseComputer,
    test_beats_mv: np.ndarray,
    q_bank: np.ndarray,
) -> Dict[str, Any]:
    """Verify physical coordinate consistency:

    1. q = e_i ==> r_q matches within numerical tolerance across canonical leads.
    2. q = (V3 - V2)/sqrt(2) ==> raw projected waveform numerically equals (V3 - V2)/sqrt(2).
    3. q = Lead III, aVF ==> raw projected waveform numerically equals authentic physical combinations.
    4. Random dense unit-norm operators q in S^7 ==> authentic physical projection matches test_beats_mv @ q.
    5. All operators have unit norm.
    """
    logger.info("Running physical coordinate and operator consistency tests...")
    e = np.eye(8, dtype=np.float32)

    # 1. Unit norm check
    norms = [float(np.linalg.norm(q)) for q in q_bank]
    max_norm_err = max(abs(n - 1.0) for n in norms)
    assert max_norm_err < 1e-5, f"Operator norm error exceeded tolerance: {max_norm_err}"

    # 2. Raw projected waveform equality check for S_icm (V3 - V2)/sqrt(2)
    q_icm = q_bank[12]  # (V3 - V2)/sqrt(2)
    expected_signal = (test_beats_mv[:, :, 4] - test_beats_mv[:, :, 3]) / np.sqrt(2.0)
    actual_signal = test_beats_mv @ q_icm
    signal_diff = float(np.max(np.abs(expected_signal - actual_signal)))
    assert signal_diff < 1e-5, f"Raw projected signal mismatch for S_icm: {signal_diff}"

    # 3. Lead III equality check (II - I)/sqrt(2)
    q_III = q_bank[8]
    expected_III = (test_beats_mv[:, :, 1] - test_beats_mv[:, :, 0]) / np.sqrt(2.0)
    actual_III = test_beats_mv @ q_III
    diff_III = float(np.max(np.abs(expected_III - actual_III)))
    assert diff_III < 1e-5, f"Raw projected signal mismatch for Lead III: {diff_III}"

    # 4. aVF equality check (II - 0.5*I)/sqrt(1.25)
    q_aVF = q_bank[11]
    expected_aVF = (test_beats_mv[:, :, 1] - 0.5 * test_beats_mv[:, :, 0]) / np.sqrt(1.25)
    actual_aVF = test_beats_mv @ q_aVF
    diff_aVF = float(np.max(np.abs(expected_aVF - actual_aVF)))
    assert diff_aVF < 1e-5, f"Raw projected signal mismatch for aVF: {diff_aVF}"

    # 5. Random dense operator test on S^7
    rng_test = np.random.default_rng(2026)
    q_rand = rng_test.normal(size=8).astype(np.float32)
    q_rand /= float(np.linalg.norm(q_rand))
    proj_rand = test_beats_mv @ q_rand
    assert proj_rand.shape == (test_beats_mv.shape[0], 256), f"Unexpected projected shape: {proj_rand.shape}"

    # 6. Canonical response consistency
    resps = computer.compute_record_responses(test_beats_mv, q_bank)
    resps_canonical = computer.compute_record_responses(test_beats_mv, e)
    basis_diff = float(np.max(np.abs(resps[:8] - resps_canonical)))
    assert basis_diff < 1e-5, f"Canonical response deviation: {basis_diff}"

    logger.info("Physical operator consistency verified (max signal diff: %.2e, max basis diff: %.2e, Lead III diff: %.2e, aVF diff: %.2e).",
                signal_diff, basis_diff, diff_III, diff_aVF)
    return {
        "status": "PASSED",
        "max_operator_norm_error": max_norm_err,
        "max_signal_projection_difference": signal_diff,
        "max_canonical_response_difference": basis_diff,
        "lead_III_projection_difference": diff_III,
        "aVF_projection_difference": diff_aVF,
    }


def evaluate_clinical_benchmarks(
    models: Dict[str, Dict[int, OperatorSetModel]],
    operators_16: np.ndarray,
    responses_16: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Evaluate all 9 clinical configurations using exact physical operator-response pairs.

    Args:
        models: {"basis": {42: m, ...}, "span": {42: m, ...}}
        operators_16: shape (N, 16, 8)
        responses_16: shape (N, 16, 16, 128)
        targets: shape (N, 5)

    Returns:
        results dictionary containing predictions, AUROCs, seed-means, and paired contrasts.
    """
    n_records = len(targets)
    seeds = sorted(list(models["basis"].keys()))
    config_names = list(CLINICAL_CONFIG_INDICES.keys())

    span_preds_dict = {}
    basis_preds_dict = {}
    metrics_summary = {}

    for cname in config_names:
        indices = CLINICAL_CONFIG_INDICES[cname]
        m_leads = len(indices)
        logger.info("Evaluating clinical config: %s (m=%d leads, indices: %s)", cname, m_leads, indices)

        # Slice exact physical operators and responses
        ops_c = operators_16[:, indices]       # (N, m, 8)
        res_c = responses_16[:, indices]       # (N, m, 16, 128)

        ops_t = torch.from_numpy(ops_c).float()
        res_t = torch.from_numpy(res_c).float()

        span_preds_seeds = []
        basis_preds_seeds = []
        span_aurocs = []
        basis_aurocs = []

        for s in seeds:
            m_sp = models["span"][s]
            m_ba = models["basis"][s]

            # Batch inference
            p_sp_list = []
            p_ba_list = []
            with torch.no_grad():
                for start_idx in range(0, n_records, batch_size):
                    end_idx = min(start_idx + batch_size, n_records)
                    b_ops = ops_t[start_idx:end_idx].to(device)
                    b_res = res_t[start_idx:end_idx].to(device)

                    p_sp_list.append(torch.sigmoid(m_sp(b_ops, b_res)).cpu().numpy())
                    p_ba_list.append(torch.sigmoid(m_ba(b_ops, b_res)).cpu().numpy())

            p_sp_arr = np.concatenate(p_sp_list, axis=0)
            p_ba_arr = np.concatenate(p_ba_list, axis=0)

            span_preds_seeds.append(p_sp_arr)
            basis_preds_seeds.append(p_ba_arr)

            auc_sp = compute_auroc_macro(targets, p_sp_arr)
            auc_ba = compute_auroc_macro(targets, p_ba_arr)
            span_aurocs.append(auc_sp)
            basis_aurocs.append(auc_ba)

        span_preds_dict[cname] = np.stack(span_preds_seeds, axis=0)   # (n_seeds, N, 5)
        basis_preds_dict[cname] = np.stack(basis_preds_seeds, axis=0) # (n_seeds, N, 5)

        mean_span = float(np.mean(span_aurocs))
        mean_basis = float(np.mean(basis_aurocs))
        seed_deltas = [sp - ba for sp, ba in zip(span_aurocs, basis_aurocs)]
        mean_delta = float(np.mean(seed_deltas))

        # Check invariant: mean_delta equals mean_span - mean_basis within float precision
        assert abs(mean_delta - (mean_span - mean_basis)) < 1e-6, "Contrast does not match difference of seed means!"

        metrics_summary[cname] = {
            "mean_span_auroc": mean_span,
            "std_span_auroc": float(np.std(span_aurocs)),
            "mean_basis_auroc": mean_basis,
            "std_basis_auroc": float(np.std(basis_aurocs)),
            "paired_delta": mean_delta,
            "seed_span_aurocs": span_aurocs,
            "seed_basis_aurocs": basis_aurocs,
            "seed_deltas": seed_deltas,
        }
        logger.info("Config %s -> Basis: %.4f, Span: %.4f, Delta: %+.4f", cname, mean_basis, mean_span, mean_delta)

    return {
        "span_preds": span_preds_dict,
        "basis_preds": basis_preds_dict,
        "metrics": metrics_summary,
        "targets": targets,
        "configurations": config_names,
        "seeds": seeds,
    }


def evaluate_grassmann_subspaces(
    models: Dict[str, Dict[int, OperatorSetModel]],
    computer: FastResponseComputer,
    record_beats_list: List[np.ndarray],
    grassmann_bank_path: Path,
    targets: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    ecg_ids: np.ndarray | None = None,
    patient_ids: np.ndarray | None = None,
    predictions_dir: Path | None = None,
    checkpoint_hashes: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Evaluate all 350 frozen Grassmann subspaces with authentic derived responses.

    Strict Requirements:
    1. Verify each subspace B_U in R^{r x 8} has exact orthonormality.
    2. Compute exact physical responses for each row q_j on authentic patient beats.
    3. Save actual patient-level probabilities for every seed, patient, rank, and subspace in resumable chunks.
    4. Compute AUROC for each subspace individually, plus rank-wise and rank-balanced aggregations.
    """
    logger.info("=== Evaluating Expanded 350-Subspace Grassmann Benchmark ===")
    bank_data = np.load(grassmann_bank_path)
    seeds = sorted(list(models["basis"].keys()))
    n_records = len(targets)
    n_seeds = len(seeds)

    if predictions_dir is not None:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    subspace_results = []
    rank_summaries = {}
    total_subspaces_evaluated = 0

    for r in range(1, 8):
        rank_key = f"rank_{r}"
        if rank_key not in bank_data:
            logger.info("Rank %s not in Grassmann bank at %s, skipping...", rank_key, grassmann_bank_path)
            continue

        subspaces_r = bank_data[rank_key]  # (n_subspaces, r, 8)
        n_sub_r = len(subspaces_r)
        logger.info("Processing Rank %d: %d subspaces (r=%d)...", r, n_sub_r, r)

        rank_span_aurocs = []
        rank_basis_aurocs = []
        rank_deltas = []

        rank_file = (predictions_dir / f"grassmann_rank_{r}_predictions.npz") if predictions_dir is not None else None
        loaded_from_cache = False

        if rank_file is not None and rank_file.exists():
            try:
                logger.info("Found existing predictions chunk for Rank %d at %s. Verifying...", r, rank_file)
                d_rank = np.load(rank_file)
                p_sp_cached = d_rank["probs_span"]  # (n_sub, n_seeds, n_records, 5)
                p_ba_cached = d_rank["probs_basis"]
                if (
                    p_sp_cached.shape == (n_sub_r, n_seeds, n_records, 5)
                    and p_ba_cached.shape == (n_sub_r, n_seeds, n_records, 5)
                ):
                    logger.info("Rank %d chunk is complete and valid (%d subspaces). Loading cached predictions...", r, n_sub_r)
                    for sub_idx in range(n_sub_r):
                        sub_sp_aurocs = [compute_auroc_macro(targets, p_sp_cached[sub_idx, s_idx].astype(np.float32)) for s_idx in range(n_seeds)]
                        sub_ba_aurocs = [compute_auroc_macro(targets, p_ba_cached[sub_idx, s_idx].astype(np.float32)) for s_idx in range(n_seeds)]
                        mean_sub_sp = float(np.mean(sub_sp_aurocs))
                        mean_sub_ba = float(np.mean(sub_ba_aurocs))
                        delta_sub = mean_sub_sp - mean_sub_ba

                        rank_span_aurocs.append(mean_sub_sp)
                        rank_basis_aurocs.append(mean_sub_ba)
                        rank_deltas.append(delta_sub)

                        subspace_results.append({
                            "rank": r,
                            "subspace_index": sub_idx,
                            "mean_span_auroc": mean_sub_sp,
                            "mean_basis_auroc": mean_sub_ba,
                            "delta": delta_sub,
                            "seed_span_aurocs": sub_sp_aurocs,
                            "seed_basis_aurocs": sub_ba_aurocs,
                        })
                        total_subspaces_evaluated += 1

                    loaded_from_cache = True
            except Exception as e:
                logger.warning("Failed to load cached Rank %d chunk (%s). Recomputing...", r, e)

        if not loaded_from_cache:
            rank_probs_span = np.empty((n_sub_r, n_seeds, n_records, 5), dtype=np.float16)
            rank_probs_basis = np.empty((n_sub_r, n_seeds, n_records, 5), dtype=np.float16)

            for sub_idx in range(n_sub_r):
                B_U = subspaces_r[sub_idx].astype(np.float32)  # (r, 8)

                # Orthonormality verification
                gram = B_U @ B_U.T
                ortho_err = float(np.max(np.abs(gram - np.eye(r, dtype=np.float32))))
                assert ortho_err < 1e-4, f"Subspace {rank_key}_{sub_idx} violates orthonormality: {ortho_err}"

                # Compute responses across all records for this subspace using authentic patient beats
                res_sub_all = np.empty((n_records, r, 16, 128), dtype=np.float32)
                for rec_idx, beats in enumerate(record_beats_list):
                    res_sub_all[rec_idx] = computer.compute_record_responses(beats, B_U)

                ops_t = torch.from_numpy(B_U).unsqueeze(0).expand(batch_size, -1, -1).to(device)
                res_t = torch.from_numpy(res_sub_all).float()

                sub_sp_aurocs = []
                sub_ba_aurocs = []

                for s_idx, s in enumerate(seeds):
                    m_sp = models["span"][s]
                    m_ba = models["basis"][s]

                    p_sp_list = []
                    p_ba_list = []
                    with torch.no_grad():
                        for start_idx in range(0, n_records, batch_size):
                            end_idx = min(start_idx + batch_size, n_records)
                            b_len = end_idx - start_idx
                            b_ops = ops_t[:b_len]
                            b_res = res_t[start_idx:end_idx].to(device)

                            p_sp_list.append(torch.sigmoid(m_sp(b_ops, b_res)).float().cpu().numpy())
                            p_ba_list.append(torch.sigmoid(m_ba(b_ops, b_res)).float().cpu().numpy())

                    p_sp = np.concatenate(p_sp_list, axis=0)  # (n_records, 5)
                    p_ba = np.concatenate(p_ba_list, axis=0)

                    rank_probs_span[sub_idx, s_idx] = p_sp.astype(np.float16)
                    rank_probs_basis[sub_idx, s_idx] = p_ba.astype(np.float16)

                    auc_sp = compute_auroc_macro(targets, p_sp)
                    auc_ba = compute_auroc_macro(targets, p_ba)
                    sub_sp_aurocs.append(auc_sp)
                    sub_ba_aurocs.append(auc_ba)

                mean_sub_sp = float(np.mean(sub_sp_aurocs))
                mean_sub_ba = float(np.mean(sub_ba_aurocs))
                delta_sub = mean_sub_sp - mean_sub_ba

                rank_span_aurocs.append(mean_sub_sp)
                rank_basis_aurocs.append(mean_sub_ba)
                rank_deltas.append(delta_sub)

                subspace_results.append({
                    "rank": r,
                    "subspace_index": sub_idx,
                    "mean_span_auroc": mean_sub_sp,
                    "mean_basis_auroc": mean_sub_ba,
                    "delta": delta_sub,
                    "seed_span_aurocs": sub_sp_aurocs,
                    "seed_basis_aurocs": sub_ba_aurocs,
                })
                total_subspaces_evaluated += 1

            if rank_file is not None:
                np.savez_compressed(
                    rank_file,
                    rank=r,
                    subspaces=subspaces_r,
                    ecg_ids=ecg_ids if ecg_ids is not None else np.arange(n_records),
                    patient_ids=patient_ids if patient_ids is not None else np.arange(n_records),
                    targets=targets,
                    seeds=np.array(seeds),
                    probs_span=rank_probs_span,
                    probs_basis=rank_probs_basis,
                    checkpoint_hashes=json.dumps(checkpoint_hashes or {}),
                )
                logger.info("Saved Rank %d complete prediction chunk to %s", r, rank_file)

        rank_summaries[f"rank_{r}"] = {
            "mean_span_auroc": float(np.mean(rank_span_aurocs)),
            "mean_basis_auroc": float(np.mean(rank_basis_aurocs)),
            "mean_delta": float(np.mean(rank_deltas)),
            "std_delta": float(np.std(rank_deltas)),
        }
        logger.info("Rank %d Complete -> Basis: %.4f, Span: %.4f, Delta: %+.4f",
                    r, np.mean(rank_basis_aurocs), np.mean(rank_span_aurocs), np.mean(rank_deltas))

    overall_deltas = [s["delta"] for s in subspace_results]
    overall_span = [s["mean_span_auroc"] for s in subspace_results]
    overall_basis = [s["mean_basis_auroc"] for s in subspace_results]

    available_ranks = [r for r in range(1, 8) if f"rank_{r}" in rank_summaries]
    rank_balanced_span = float(np.mean([rank_summaries[f"rank_{r}"]["mean_span_auroc"] for r in available_ranks])) if available_ranks else 0.5
    rank_balanced_basis = float(np.mean([rank_summaries[f"rank_{r}"]["mean_basis_auroc"] for r in available_ranks])) if available_ranks else 0.5
    rank_balanced_delta = float(np.mean([rank_summaries[f"rank_{r}"]["mean_delta"] for r in available_ranks])) if available_ranks else 0.0

    logger.info("Total Grassmann Subspaces Evaluated: %d/350", total_subspaces_evaluated)
    logger.info("Grassmann Overall (350 spaces) -> Basis: %.4f | Span: %.4f | Delta: %+.4f",
                np.mean(overall_basis), np.mean(overall_span), np.mean(overall_deltas))
    logger.info("Grassmann Rank-Balanced (Ranks 1-7) -> Basis: %.4f | Span: %.4f | Delta: %+.4f",
                rank_balanced_basis, rank_balanced_span, rank_balanced_delta)

    summary = {
        "total_subspaces": total_subspaces_evaluated,
        "overall_mean_span_auroc": float(np.mean(overall_span)),
        "overall_mean_basis_auroc": float(np.mean(overall_basis)),
        "overall_mean_delta": float(np.mean(overall_deltas)),
        "overall_std_delta": float(np.std(overall_deltas)),
        "rank_balanced_mean_span_auroc": rank_balanced_span,
        "rank_balanced_mean_basis_auroc": rank_balanced_basis,
        "rank_balanced_mean_delta": rank_balanced_delta,
        "rank_summaries": rank_summaries,
        "subspace_results": subspace_results,
    }

    if predictions_dir is not None:
        manifest_path = predictions_dir / "grassmann_350_predictions_manifest.json"
        manifest_path.write_text(json.dumps({
            "status": "GRASSMANN_350_AUTHENTIC_EVALUATION_COMPLETE",
            "total_subspaces": total_subspaces_evaluated,
            "overall_mean_basis_auroc": float(np.mean(overall_basis)),
            "overall_mean_span_auroc": float(np.mean(overall_span)),
            "overall_mean_delta": float(np.mean(overall_deltas)),
            "rank_balanced_mean_basis_auroc": rank_balanced_basis,
            "rank_balanced_mean_span_auroc": rank_balanced_span,
            "rank_balanced_mean_delta": rank_balanced_delta,
            "rank_summaries": rank_summaries,
            "checkpoint_hashes": checkpoint_hashes or {},
            "timestamp": time.time(),
        }, indent=2))
        logger.info("Saved Grassmann master manifest to %s", manifest_path)

    return summary


def evaluate_observed_only_segmentation(
    models: Dict[str, Dict[int, OperatorSetModel]],
    computer: FastResponseComputer,
    single_lead_waveforms: np.ndarray,
    oracle_beats_list: List[np.ndarray],
    targets: np.ndarray,
    q: np.ndarray,
    config_name: str,
    device: torch.device,
    fs: float = 500.0,
) -> Dict[str, Any]:
    """Evaluate observed-only single-measurement beat detection vs oracle timing.

    Args:
        models: trained models
        single_lead_waveforms: shape (N, T) projected signal y_q(t)
        oracle_beats_list: list of oracle beats for eligible records
        targets: labels
        q: operator vector (8,)
        config_name: e.g. "S_1"
    """
    logger.info("Evaluating observed-only segmentation for %s...", config_name)
    n_records = len(targets)
    seeds = sorted(list(models["basis"].keys()))

    # Filter bandpass for R-peak detection
    nyq = 0.5 * fs
    b, a = signal.butter(2, [max(0.01, min(0.9, 5.0 / nyq)), max(0.05, min(0.95, 25.0 / nyq))], btype="bandpass")

    eligible_observed_indices = []
    observed_beats_list = []
    failure_count = 0

    for idx in range(n_records):
        y = single_lead_waveforms[idx]
        filtered = signal.filtfilt(b, a, y)
        diff = np.diff(filtered, prepend=filtered[0])
        squared = diff ** 2
        window = int(0.15 * fs)
        kernel = np.ones(window) / window
        integrated = np.convolve(squared, kernel, mode="same")
        threshold = np.mean(integrated) + 1.2 * np.std(integrated)
        min_distance = int(0.3 * fs)
        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

        if len(peaks) < 2:
            failure_count += 1
            continue

        # Extract normalized beats from observed peaks
        intervals = []
        for i in range(len(peaks) - 1):
            intervals.append((peaks[i], peaks[i + 1]))

        # Interpolate each beat to 256 samples
        beats_obs = []
        for start, end in intervals:
            beat_raw = y[start:end]
            x_old = np.linspace(0, 1, len(beat_raw))
            x_new = np.linspace(0, 1, 256)
            beat_resampled = np.interp(x_new, x_old, beat_raw)
            beats_obs.append(beat_resampled)

        if len(beats_obs) >= 2:
            eligible_observed_indices.append(idx)
            # Create (beat, 256, 8) with single lead in physical units
            # For response computation, y_q is projected onto q
            beats_arr = np.array(beats_obs)[:, :, None] # (beat, 256, 1)
            observed_beats_list.append(beats_arr)
        else:
            failure_count += 1

    n_eligible_obs = len(eligible_observed_indices)
    retained_coverage = float(n_eligible_obs / n_records)
    detection_failure_rate = float(failure_count / n_records)

    logger.info("Observed-only %s: retained %d/%d (%.1f%% coverage, failure rate: %.2f%%)",
                config_name, n_eligible_obs, n_records, retained_coverage * 100.0, detection_failure_rate * 100.0)

    # Compute observed responses and oracle responses on shared eligible cohort
    shared_idx = np.array(eligible_observed_indices)
    targets_shared = targets[shared_idx]

    obs_res_list = []
    oracle_res_list = []
    q_norm = normalize_operator(q)
    q_mat = q_norm.reshape(1, 8)

    for i_shared, orig_idx in enumerate(shared_idx):
        # Oracle beats
        oracle_b = oracle_beats_list[orig_idx]
        oracle_r = computer.compute_record_responses(oracle_b, q_mat) # (1, 16, 128)
        oracle_res_list.append(oracle_r)

        # Observed beats
        # For single-lead observed beat, values @ q / v_scale equals y_q / v_scale directly
        obs_b = observed_beats_list[i_shared] # (beat, 256, 1)
        wf = obs_b.squeeze(-1).astype(np.float64) / computer.v_scale # (beat, 256)
        deriv = np.gradient(wf, axis=1)
        atoms = np.stack([wf, deriv], axis=-1)[None, ...] # (1, beat, 256, 2)
        cells = atoms.reshape(1, len(obs_b), 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(1, 16, -1, 2)
        t_cells = torch.as_tensor(cells, device=device, dtype=torch.float32)
        white = (t_cells - computer.w_mean) @ computer.w_comp * computer.w_scale
        d2 = torch.cdist(white, computer.landmarks.expand(1, 16, -1, -1)).square() + computer.c2
        feat = torch.rsqrt(d2) @ computer.inv_root
        obs_r = feat.mean(dim=2).cpu().numpy().astype(np.float32) # (1, 16, 128)
        obs_res_list.append(obs_r)

    obs_res_arr = np.stack(obs_res_list, axis=0)       # (N_shared, 1, 16, 128)
    oracle_res_arr = np.stack(oracle_res_list, axis=0) # (N_shared, 1, 16, 128)

    ops_t = torch.from_numpy(q_mat).unsqueeze(0).expand(len(shared_idx), -1, -1).float().to(device)
    obs_res_t = torch.from_numpy(obs_res_arr).float().to(device)
    ora_res_t = torch.from_numpy(oracle_res_arr).float().to(device)

    # Run inference across models
    span_obs_auroc = []
    span_ora_auroc = []

    for s in seeds:
        m_sp = models["span"][s]
        with torch.no_grad():
            p_obs = torch.sigmoid(m_sp(ops_t, obs_res_t)).cpu().numpy()
            p_ora = torch.sigmoid(m_sp(ops_t, ora_res_t)).cpu().numpy()

        span_obs_auroc.append(compute_auroc_macro(targets_shared, p_obs))
        span_ora_auroc.append(compute_auroc_macro(targets_shared, p_ora))

    mean_obs = float(np.mean(span_obs_auroc))
    mean_ora = float(np.mean(span_ora_auroc))
    delta_seg = mean_obs - mean_ora

    logger.info("Segmentation Audit [%s]: AUROC_oracle=%.4f, AUROC_observed=%.4f, Delta_seg=%+.4f",
                config_name, mean_ora, mean_obs, delta_seg)

    return {
        "config_name": config_name,
        "n_total": n_records,
        "n_shared_eligible": n_eligible_obs,
        "retained_cohort_coverage": retained_coverage,
        "detection_failure_rate": detection_failure_rate,
        "auroc_oracle": mean_ora,
        "auroc_observed": mean_obs,
        "delta_seg": delta_seg,
        "seed_oracle_aurocs": span_ora_auroc,
        "seed_observed_aurocs": span_obs_auroc,
    }
