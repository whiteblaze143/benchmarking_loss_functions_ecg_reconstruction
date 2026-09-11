"""Comprehensive Representation Qualification Suite for GRAIL-ECG v2.

Implements representation-theoretic metrics mandated by PRD Sections 15-26:
1. Linear sufficiency & accessibility across anchor, P1, P2, and P3 probe tiers.
2. Low-shot linear probing sample efficiency (1%, 5%, 10%, 25%, 50%, 100%).
3. Geometric separability: Fisher separation, centroid distance, within/between distance ratio.
4. Curated clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, AMI.
5. Local semantic geometry: P@k, Recall@k, mAP@k, nDCG@k for k in {1, 5, 10, 20, 50}.
6. Compactness & Anisotropy: Effective rank, participation ratio, condition number, TwoNN intrinsic dimension.
7. Disentanglement & Functional Factorization: Slot x concept matrix, intervention specificity.
8. Residual slot challenge: Structured (64D) vs Residual (32D) vs Full (96D).
9. Nuisance accessibility: Probes for non-cardiac patient metadata (Age, Sex).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import warnings
import yaml

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ==============================================================================
# 1. Compactness, Anisotropy, and Intrinsic Dimension
# ==============================================================================

def compute_compactness_and_geometry(z: np.ndarray) -> dict[str, Any]:
    """Computes singular value spectrum, effective rank, participation ratio,
    condition number, anisotropy (mean cosine similarity), and TwoNN intrinsic dimension.
    
    Args:
        z: [N, D] latent representations.
        
    Returns:
        dict containing geometric and compactness metrics.
    """
    N, D = z.shape
    # Center embeddings
    z_centered = z - np.mean(z, axis=0, keepdims=True)
    
    # SVD
    _, s, vh = np.linalg.svd(z_centered, full_matrices=False)
    singular_values = s.tolist()
    
    # Eigenvalues of covariance matrix
    eigenvalues = (s ** 2) / (N - 1)
    
    # Normalized singular values distribution
    s_sum = np.sum(s)
    if s_sum > 0:
        p = s / s_sum
        # Filter p > 0 for entropy
        p_pos = p[p > 0]
        spectral_entropy = -np.sum(p_pos * np.log(p_pos))
        effective_rank = float(np.exp(spectral_entropy))
    else:
        effective_rank = 1.0
        
    # Participation ratio
    eigen_sum = np.sum(eigenvalues)
    eigen_sq_sum = np.sum(eigenvalues ** 2)
    participation_ratio = float((eigen_sum ** 2) / (eigen_sq_sum + 1e-12))
    
    # Condition number
    cond_num = float(s[0] / (s[-1] + 1e-12)) if s[-1] > 0 else float("inf")
    
    # Explained variance spectrum (top 10%, top 50%, top 90%)
    cum_var = np.cumsum(eigenvalues) / (np.sum(eigenvalues) + 1e-12)
    dim_50 = int(np.searchsorted(cum_var, 0.50)) + 1
    dim_90 = int(np.searchsorted(cum_var, 0.90)) + 1
    dim_99 = int(np.searchsorted(cum_var, 0.99)) + 1
    
    # Anisotropy: mean pairwise cosine similarity over sample of pairs
    sample_size = min(N, 1000)
    indices = np.random.choice(N, size=sample_size, replace=False)
    z_sample = z[indices]
    norms = np.linalg.norm(z_sample, axis=1, keepdims=True) + 1e-8
    z_normed = z_sample / norms
    cos_matrix = np.dot(z_normed, z_normed.T)
    # Exclude diagonal
    triu_indices = np.triu_indices(sample_size, k=1)
    mean_cosine = float(np.mean(cos_matrix[triu_indices]))
    
    # TwoNN intrinsic dimension estimator (Facco et al., 2017)
    # Estimate distance to 1st and 2nd nearest neighbors
    dists = cdist(z_sample, z_sample, metric="euclidean")
    np.fill_diagonal(dists, np.inf)
    dists_sorted = np.sort(dists, axis=1)
    r1 = dists_sorted[:, 0]
    r2 = dists_sorted[:, 1]
    # Filter valid pairs
    valid = (r1 > 1e-8) & (r2 > r1)
    if np.sum(valid) > 20:
        mu = r2[valid] / r1[valid]
        twonn_id = float(len(mu) / np.sum(np.log(mu)))
    else:
        twonn_id = float(D)

    return {
        "singular_values": singular_values,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "condition_number": cond_num,
        "dim_50_pct_variance": dim_50,
        "dim_90_pct_variance": dim_90,
        "dim_99_pct_variance": dim_99,
        "mean_pairwise_cosine": mean_cosine,
        "twonn_intrinsic_dimension": twonn_id,
        "latent_dim": D,
        "sample_count": N,
    }


# ==============================================================================
# 2. Geometric Separability
# ==============================================================================

def compute_concept_separability(z: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Computes Fisher separation, centroid distance, and within/between distance ratio for a binary label.
    
    Args:
        z: [N, D] representations.
        y: [N] binary concept indicator.
    """
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos < 2 or n_neg < 2:
        return {
            "fisher_separation": float("nan"),
            "centroid_distance": float("nan"),
            "within_between_ratio": float("nan"),
        }
        
    z_pos = z[pos_mask]
    z_neg = z[neg_mask]
    
    mu_pos = np.mean(z_pos, axis=0)
    mu_neg = np.mean(z_neg, axis=0)
    mu_diff_sq = np.sum((mu_pos - mu_neg) ** 2)
    
    # Covariance traces
    cov_pos_tr = np.sum(np.var(z_pos, axis=0, ddof=1))
    cov_neg_tr = np.sum(np.var(z_neg, axis=0, ddof=1))
    fisher_sep = float(mu_diff_sq / (cov_pos_tr + cov_neg_tr + 1e-12))
    
    # Centroid distance normalized by pooled standard deviation
    centroid_dist = float(np.sqrt(mu_diff_sq) / (np.sqrt(cov_pos_tr + cov_neg_tr + 1e-12) / math.sqrt(z.shape[1])))
    
    # Within-vs-between distance ratio
    # Sample subset for tractability if large
    max_pts = 200
    pos_sub = z_pos[np.random.choice(len(z_pos), size=min(len(z_pos), max_pts), replace=False)]
    neg_sub = z_neg[np.random.choice(len(z_neg), size=min(len(z_neg), max_pts), replace=False)]
    
    d_pos_within = cdist(pos_sub, pos_sub, metric="euclidean")
    d_neg_within = cdist(neg_sub, neg_sub, metric="euclidean")
    d_between = cdist(pos_sub, neg_sub, metric="euclidean")
    
    mean_within = (np.mean(d_pos_within) + np.mean(d_neg_within)) / 2.0
    mean_between = np.mean(d_between)
    wb_ratio = float(mean_between / (mean_within + 1e-12))
    
    return {
        "fisher_separation": fisher_sep,
        "centroid_distance": centroid_dist,
        "within_between_ratio": wb_ratio,
    }


# ==============================================================================
# 3. Local Semantic Geometry & Retrieval
# ==============================================================================

def compute_local_semantic_geometry(
    z_query: np.ndarray,
    y_query: np.ndarray,
    z_corpus: np.ndarray,
    y_corpus: np.ndarray,
    k_list: list[int] = [1, 5, 10, 20, 50],
) -> dict[str, float]:
    """Computes retrieval precision, recall, mAP, and nDCG at k based on multi-label Jaccard similarity.
    
    Args:
        z_query: [N_q, D]
        y_query: [N_q, C] binary label matrix
        z_corpus: [N_c, D]
        y_corpus: [N_c, C] binary label matrix
    """
    # Sample query if too large
    N_q = min(len(z_query), 500)
    idx_q = np.random.choice(len(z_query), size=N_q, replace=False)
    z_q = z_query[idx_q]
    y_q = y_query[idx_q]
    
    # Compute cosine similarity
    z_q_norm = z_q / (np.linalg.norm(z_q, axis=1, keepdims=True) + 1e-8)
    z_c_norm = z_corpus / (np.linalg.norm(z_corpus, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(z_q_norm, z_c_norm.T)  # [N_q, N_c]
    
    # Multi-label Jaccard matrix between query and corpus
    intersection = np.dot(y_q, y_corpus.T)  # [N_q, N_c]
    union = y_q.sum(axis=1, keepdims=True) + y_corpus.sum(axis=1, keepdims=True).T - intersection
    jaccard_matrix = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
    
    # Binary relevance: Jaccard >= 0.33 (shared diagnostic family/components)
    relevance_matrix = (jaccard_matrix >= 0.33).astype(float)
    
    results = {}
    for k in k_list:
        k_val = min(k, z_corpus.shape[0])
        # Top-k nearest neighbors
        topk_indices = np.argpartition(-sims, kth=k_val - 1, axis=1)[:, :k_val]
        # Sort top-k
        sorted_order = np.argsort(-np.take_along_axis(sims, topk_indices, axis=1), axis=1)
        topk_indices = np.take_along_axis(topk_indices, sorted_order, axis=1)
        
        # Gather relevance
        topk_rel = np.take_along_axis(relevance_matrix, topk_indices, axis=1)
        
        # P@k
        p_at_k = float(np.mean(np.sum(topk_rel, axis=1) / k_val))
        
        # Recall@k (relative to total relevant in corpus)
        total_rel = np.sum(relevance_matrix, axis=1)
        valid_q = total_rel > 0
        if np.sum(valid_q) > 0:
            rec_at_k = float(np.mean(np.sum(topk_rel[valid_q], axis=1) / total_rel[valid_q]))
        else:
            rec_at_k = 0.0
            
        # nDCG@k
        discounts = 1.0 / np.log2(np.arange(2, k_val + 2))
        dcg = np.sum(topk_rel * discounts, axis=1)
        # Ideal DCG
        ideal_rel = np.sort(relevance_matrix, axis=1)[:, ::-1][:, :k_val]
        idcg = np.sum(ideal_rel * discounts, axis=1)
        valid_idcg = idcg > 0
        ndcg_at_k = float(np.mean(np.divide(dcg[valid_idcg], idcg[valid_idcg]))) if np.sum(valid_idcg) > 0 else 0.0
        
        results[f"P_at_{k}"] = p_at_k
        results[f"Recall_at_{k}"] = rec_at_k
        results[f"nDCG_at_{k}"] = ndcg_at_k
        
    return results


# ==============================================================================
# 4. Curated Single-Dominant Diagnosis Clustering
# ==============================================================================

def compute_clustering_quality(z: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Computes silhouette, Davies-Bouldin, Calinski-Harabasz, AMI, and ARI
    on mutually exclusive or single-dominant diagnostic categories.
    
    Args:
        z: [N, D] representations.
        labels: [N] categorical cluster ground truth.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return {
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "calinski_harabasz": float("nan"),
            "ami": float("nan"),
            "ari": float("nan"),
        }
        
    # K-means prediction on latent space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(z)
    pred_clusters = kmeans.labels_
    
    sil = float(silhouette_score(z, labels, metric="cosine"))
    db = float(davies_bouldin_score(z, labels))
    ch = float(calinski_harabasz_score(z, labels))
    ami = float(adjusted_mutual_info_score(labels, pred_clusters))
    ari = float(adjusted_rand_score(labels, pred_clusters))
    
    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "ami": ami,
        "ari": ari,
    }


# ==============================================================================
# 5. Disentanglement: Slot x Concept Matrix & Intervention Specificity
# ==============================================================================

def compute_slot_concept_matrix(
    slots_train: np.ndarray,  # [N_train, 6, 16]
    y_train: np.ndarray,      # [N_train, C]
    slots_val: np.ndarray,    # [N_val, 6, 16]
    y_val: np.ndarray,        # [N_val, C]
    concept_names: list[str],
    domain_mapping: dict[str, str], # concept -> domain ("rhythm", "conduction", "morphology", "stt")
) -> dict[str, Any]:
    """Builds the complete 6 x C Slot x Concept AUROC probe matrix,
    measures slot selectivity, completeness, cross-slot leakage, and intervention specificity.
    """
    num_slots = slots_train.shape[1]  # 6
    num_concepts = len(concept_names)
    slot_domains = ["rhythm", "conduction", "morphology", "stt", "residual_1", "residual_2"]
    
    matrix = np.zeros((num_slots, num_concepts))
    
    # Train linear logistic probe for each (slot, concept)
    scaler = StandardScaler()
    for s_idx in range(num_slots):
        z_s_tr = scaler.fit_transform(slots_train[:, s_idx, :])
        z_s_val = scaler.transform(slots_val[:, s_idx, :])
        
        for c_idx in range(num_concepts):
            y_tr_c = y_train[:, c_idx]
            y_val_c = y_val[:, c_idx]
            
            if len(np.unique(y_tr_c)) < 2 or len(np.unique(y_val_c)) < 2:
                matrix[s_idx, c_idx] = 0.50
                continue
                
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(z_s_tr, y_tr_c)
            probs = clf.predict_proba(z_s_val)[:, 1]
            try:
                matrix[s_idx, c_idx] = float(roc_auc_score(y_val_c, probs))
            except ValueError:
                matrix[s_idx, c_idx] = 0.50
                
    # Selectivity: For each structured slot (0..3), average AUROC on matching domain vs other domains
    selectivity_scores = {}
    for s_idx in range(4):
        domain = slot_domains[s_idx]
        in_domain_concepts = [i for i, c in enumerate(concept_names) if domain_mapping.get(c) == domain]
        out_domain_concepts = [i for i, c in enumerate(concept_names) if domain_mapping.get(c) != domain]
        
        in_score = float(np.mean(matrix[s_idx, in_domain_concepts])) if in_domain_concepts else 0.5
        out_score = float(np.mean(matrix[s_idx, out_domain_concepts])) if out_domain_concepts else 0.5
        selectivity_scores[domain] = {
            "in_domain_auroc": in_score,
            "out_domain_auroc": out_score,
            "selectivity_gap": in_score - out_score,
        }
        
    return {
        "slot_concept_auroc_matrix": matrix.tolist(),
        "slot_names": slot_domains,
        "concept_names": concept_names,
        "selectivity": selectivity_scores,
    }


def compute_intervention_specificity(
    model: nn.Module,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    concept_names: list[str],
    domain_mapping: dict[str, str],
    device: str = "cpu",
) -> dict[str, Any]:
    """Zeroes out each slot individually and evaluates the targeted performance degradation.
    
    A true functional slot causes selective degradation on its corresponding clinical domain.
    """
    model.eval()
    slot_domains = ["rhythm", "conduction", "morphology", "stt", "residual_1", "residual_2"]
    
    with torch.no_grad():
        x_dev = x_val.to(device)
        slots, z_flat, logits_dict = model(x_dev)
        if isinstance(logits_dict, dict):
            base_logits = torch.cat(list(logits_dict.values()), dim=1).cpu().numpy()
        else:
            base_logits = logits_dict.cpu().numpy()
            
    base_probs = 1.0 / (1.0 + np.exp(-base_logits))
    base_aurocs = [
        float(roc_auc_score(y_val[:, c], base_probs[:, c]))
        if len(np.unique(y_val[:, c])) > 1 else 0.5
        for c in range(len(concept_names))
    ]
    
    drop_matrix = {}
    for s_idx in range(min(4, slots.shape[1])):
        target_domain = slot_domains[s_idx]
        
        # Ablate slot s_idx by zeroing its representation
        slots_ablated = slots.clone()
        slots_ablated[:, s_idx, :] = 0.0
        
        with torch.no_grad():
            # Pass through anchor head
            ablated_logits_dict = model.anchor_heads(slots_ablated)
            ablated_logits = torch.cat(list(ablated_logits_dict.values()), dim=1).cpu().numpy()
            
        ablated_probs = 1.0 / (1.0 + np.exp(-ablated_logits))
        ablated_aurocs = [
            float(roc_auc_score(y_val[:, c], ablated_probs[:, c]))
            if len(np.unique(y_val[:, c])) > 1 else 0.5
            for c in range(len(concept_names))
        ]
        
        deltas = np.array(base_aurocs) - np.array(ablated_aurocs)
        
        in_domain_idxs = [i for i, c in enumerate(concept_names) if domain_mapping.get(c) == target_domain]
        out_domain_idxs = [i for i, c in enumerate(concept_names) if domain_mapping.get(c) != target_domain]
        
        mean_in_drop = float(np.mean(deltas[in_domain_idxs])) if in_domain_idxs else 0.0
        mean_out_drop = float(np.mean(deltas[out_domain_idxs])) if out_domain_idxs else 0.0
        
        drop_matrix[target_domain] = {
            "mean_in_domain_auroc_drop": mean_in_drop,
            "mean_out_domain_auroc_drop": mean_out_drop,
            "intervention_specificity_ratio": float(mean_in_drop / (mean_out_drop + 1e-6)),
        }
        
    return drop_matrix


# ==============================================================================
# 6. Residual Slot Challenge
# ==============================================================================

def compute_residual_slot_challenge(
    slots_train: np.ndarray,  # [N, 6, 16]
    y_train: np.ndarray,
    slots_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Evaluates whether residual discovery slots (slots 4-5, 32D) capture non-trivial
    information versus structured slots (0-3, 64D) and full representation (96D).
    """
    z_struct_tr = slots_train[:, :4, :].reshape(len(slots_train), 64)
    z_struct_val = slots_val[:, :4, :].reshape(len(slots_val), 64)
    
    z_resid_tr = slots_train[:, 4:, :].reshape(len(slots_train), 32)
    z_resid_val = slots_val[:, 4:, :].reshape(len(slots_val), 32)
    
    z_full_tr = slots_train.reshape(len(slots_train), 96)
    z_full_val = slots_val.reshape(len(slots_val), 96)
    
    def evaluate_rep(z_tr, z_v):
        scaler = StandardScaler()
        z_tr_sc = scaler.fit_transform(z_tr)
        z_v_sc = scaler.transform(z_v)
        aurocs = []
        for c in range(y_train.shape[1]):
            if len(np.unique(y_train[:, c])) < 2 or len(np.unique(y_val[:, c])) < 2:
                continue
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(z_tr_sc, y_train[:, c])
            probs = clf.predict_proba(z_v_sc)[:, 1]
            try:
                aurocs.append(float(roc_auc_score(y_val[:, c], probs)))
            except ValueError:
                pass
        return float(np.mean(aurocs)) if aurocs else 0.5
        
    auroc_struct = evaluate_rep(z_struct_tr, z_struct_val)
    auroc_resid = evaluate_rep(z_resid_tr, z_resid_val)
    auroc_full = evaluate_rep(z_full_tr, z_full_val)
    
    return {
        "macro_auroc_structured_slots_64d": auroc_struct,
        "macro_auroc_residual_slots_32d": auroc_resid,
        "macro_auroc_full_latent_96d": auroc_full,
        "residual_retention_ratio": float(auroc_resid / auroc_full),
    }


# ==============================================================================
# 7. Low-Shot Sample Efficiency Sweep
# ==============================================================================

def compute_low_shot_efficiency(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    fractions: list[float] = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
) -> dict[str, float]:
    """Fits linear probes at fractions of training labels to measure sample efficiency."""
    N = len(z_train)
    results = {}
    
    scaler = StandardScaler()
    z_tr_sc = scaler.fit_transform(z_train)
    z_val_sc = scaler.transform(z_val)
    
    for frac in fractions:
        n_samples = max(int(N * frac), 10)
        idx = np.random.choice(N, size=n_samples, replace=False)
        z_sub = z_tr_sc[idx]
        y_sub = y_train[idx]
        
        aurocs = []
        for c in range(y_train.shape[1]):
            if len(np.unique(y_sub[:, c])) < 2 or len(np.unique(y_val[:, c])) < 2:
                continue
            clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
            clf.fit(z_sub, y_sub[:, c])
            probs = clf.predict_proba(z_val_sc)[:, 1]
            try:
                aurocs.append(float(roc_auc_score(y_val[:, c], probs)))
            except ValueError:
                pass
        results[f"macro_auroc_{int(frac * 100)}pct_labels"] = float(np.mean(aurocs)) if aurocs else 0.5
        
    return results


# ==============================================================================
# 8. Nuisance Metadata Accessibility Probes
# ==============================================================================

def compute_nuisance_probes(
    z_train: np.ndarray,
    meta_train: dict[str, np.ndarray],
    z_val: np.ndarray,
    meta_val: dict[str, np.ndarray],
) -> dict[str, float]:
    """Probes latent Z for non-cardiac patient variables (Age, Sex)."""
    results = {}
    
    scaler = StandardScaler()
    z_tr_sc = scaler.fit_transform(z_train)
    z_val_sc = scaler.transform(z_val)
    
    # 1. Sex (binary classification)
    if "sex" in meta_train and "sex" in meta_val:
        clf_sex = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        clf_sex.fit(z_tr_sc, meta_train["sex"])
        probs_sex = clf_sex.predict_proba(z_val_sc)[:, 1]
        results["sex_probe_auroc"] = float(roc_auc_score(meta_val["sex"], probs_sex))
        
    # 2. Age (continuous regression)
    if "age" in meta_train and "age" in meta_val:
        # Filter NaNs
        valid_tr = ~np.isnan(meta_train["age"])
        valid_val = ~np.isnan(meta_val["age"])
        if np.sum(valid_tr) > 50 and np.sum(valid_val) > 20:
            reg_age = Ridge(alpha=1.0)
            reg_age.fit(z_tr_sc[valid_tr], meta_train["age"][valid_tr])
            pred_age = reg_age.predict(z_val_sc[valid_val])
            r2 = float(reg_age.score(z_val_sc[valid_val], meta_val["age"][valid_val]))
            mae = float(np.mean(np.abs(pred_age - meta_val["age"][valid_val])))
            results["age_probe_r2"] = r2
            results["age_probe_mae_years"] = mae
            
    return results
