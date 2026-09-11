r"""Exhaustive 4,095 Subset Lattice Evaluation & Lead Information Theory Engine.

Implements Stage B requirements from PRD v2 Sections 32-42:
1. All 4,095 unique non-empty subsets of the 12 displayed leads.
2. Exact independent measurement rank r(S) relative to {I, II, V1-V6} and limb algebra rank.
3. Fast token-caching subset evaluation: O(N * 12) temporal encoding, O(N * 4095) cross-attention.
4. Latent distance to Z*: Cosine similarity, normalized L2, coordinate drift gap.
5. Coordinate-preserving (fixed full-head) vs Information-preserving (reprobed) evaluation.
6. Exact Lead Shapley values:
       phi_i = sum_{S subset N \ {i}} [ |S|!(12-|S|-1)! / 12! ] * [ v(S u {i}) - v(S) ]
7. Pairwise lead synergy / redundancy (Harsanyi/Möbius interaction):
       I_{ij} = v(S u {i, j}) - v(S u {i}) - v(S u {j}) + v(S)
8. Best / median / worst Pareto information frontiers vs displayed count k and rank r.
9. Disease-specific minimal sufficient lead sets.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LIMB_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF"]
PRECORDIAL_LEADS = ["V1", "V2", "V3", "V4", "V5", "V6"]

# Linear projection from independent basis [I, II] to all 6 frontal limb leads
# III = II - I
# aVR = -0.5 * I - 0.5 * II
# aVL = I - 0.5 * II
# aVF = -0.5 * I + II
LIMB_BASIS_MATRIX = {
    "I": np.array([1.0, 0.0]),
    "II": np.array([0.0, 1.0]),
    "III": np.array([-1.0, 1.0]),
    "aVR": np.array([-0.5, -0.5]),
    "aVL": np.array([1.0, -0.5]),
    "aVF": np.array([-0.5, 1.0]),
}


class SubsetRegistry:
    """Complete catalog of all 2^12 - 1 = 4,095 non-empty unique lead subsets."""

    def __init__(self):
        self.subsets = []
        for mask in range(1, 1 << 12):
            leads = [STANDARD_12_LEADS[i] for i in range(12) if (mask & (1 << i))]
            k = len(leads)
            n_limb = sum(1 for l in leads if l in LIMB_LEADS)
            n_precordial = sum(1 for l in leads if l in PRECORDIAL_LEADS)

            # Limb algebraic rank (max 2)
            if n_limb == 0:
                limb_rank = 0
            elif n_limb == 1:
                limb_rank = 1
            else:
                limb_mat = np.stack([LIMB_BASIS_MATRIX[l] for l in leads if l in LIMB_LEADS], axis=0)
                limb_rank = int(np.linalg.matrix_rank(limb_mat, tol=1e-5))

            # Total independent rank relative to {I, II, V1-V6} (max 8)
            indep_rank = limb_rank + n_precordial

            self.subsets.append({
                "subset_bitmask": mask,
                "lead_names": leads,
                "k_displayed": k,
                "n_limb": n_limb,
                "limb_rank": limb_rank,
                "n_precordial": n_precordial,
                "independent_rank": indep_rank,
            })

    def __len__(self) -> int:
        return len(self.subsets)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.subsets)
        df["lead_names_str"] = df["lead_names"].apply(lambda x: ",".join(x))
        return df


def compute_exact_lead_shapley(
    val_func: Callable[[int], float],
    num_players: int = 12,
) -> np.ndarray:
    """Computes exact Shapley values for all players over 2^N - 1 game outcomes.
    
    Args:
        val_func: callable mapping subset bitmask (int) -> performance value v(S).
        num_players: total players (12 leads).
        
    Returns:
        shapley_values: [12] array of marginal contributions.
    """
    n = num_players
    shapley = np.zeros(n, dtype=float)

    # Precompute factorials
    weights = {}
    for s in range(n):
        weights[s] = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)

    # Evaluate game value for all 2^n subsets
    values = np.zeros(1 << n, dtype=float)
    values[0] = 0.0
    for mask in range(1, 1 << n):
        values[mask] = val_func(mask)

    # Exact Shapley formula
    for i in range(n):
        i_bit = 1 << i
        phi_i = 0.0
        # Iterate over all subsets S not containing i
        for mask in range(1 << n):
            if not (mask & i_bit):
                s_size = bin(mask).count("1")
                w = weights[s_size]
                marginal = values[mask | i_bit] - values[mask]
                phi_i += w * marginal
        shapley[i] = phi_i

    return shapley


def compute_pairwise_lead_interactions(
    val_func: Callable[[int], float],
    num_players: int = 12,
) -> np.ndarray:
    """Computes Harsanyi/Möbius pairwise synergy/redundancy interaction matrix.
    
    I_{ij} > 0: Synergy (leads cooperate to provide more info than sum of individuals).
    I_{ij} < 0: Redundancy (leads share overlapping info).
    """
    n = num_players
    matrix = np.zeros((n, n), dtype=float)

    # Precompute values
    values = np.zeros(1 << n, dtype=float)
    for mask in range(1, 1 << n):
        values[mask] = val_func(mask)

    for i in range(n):
        for j in range(i + 1, n):
            i_bit = 1 << i
            j_bit = 1 << j
            interaction_sum = 0.0
            count = 0
            # Average 2nd order difference across conditioning contexts
            for mask in range(1 << n):
                if not (mask & i_bit) and not (mask & j_bit):
                    s_size = bin(mask).count("1")
                    # Focus on small/medium contexts for statistical stability
                    if s_size <= 4:
                        v_none = values[mask]
                        v_i = values[mask | i_bit]
                        v_j = values[mask | j_bit]
                        v_ij = values[mask | i_bit | j_bit]
                        delta_2 = (v_ij - v_i) - (v_j - v_none)
                        interaction_sum += delta_2
                        count += 1
            avg_interaction = interaction_sum / count if count > 0 else 0.0
            matrix[i, j] = avg_interaction
            matrix[j, i] = avg_interaction

    return matrix


class TokenCachedSubsetEvaluator:
    """Evaluates all 4,095 subsets efficiently using cached spatio-temporal lead tokens."""

    def __init__(
        self,
        model: nn.Module,
        linear_head: nn.Linear | None = None,
        canonical_angles_12: torch.Tensor | None = None,
        device: str = "cuda",
    ):
        self.model = model
        self.linear_head = linear_head
        self.device = torch.device(device)
        self.canonical_angles_12 = canonical_angles_12

    def cache_all_lead_tokens(
        self,
        ecg_12l: torch.Tensor,  # [B, 12, 5000]
    ) -> torch.Tensor:
        """Precomputes spatio-temporal tokens for all 12 leads: [B, 12, 32, hidden_dim]."""
        self.model.eval()
        B = ecg_12l.shape[0]
        tokens_list = []

        with torch.no_grad():
            for lead_idx in range(12):
                single_lead = ecg_12l[:, lead_idx:lead_idx + 1, :].to(self.device)  # [B, 1, 5000]
                if self.canonical_angles_12 is not None:
                    angle = self.canonical_angles_12[lead_idx:lead_idx + 1].expand(B, -1).to(self.device)
                else:
                    angle = None

                # Forward single lead through view encoder
                if hasattr(self.model, "view_encoder"):
                    tok = self.model.view_encoder(single_lead, custom_angles=angle)  # [B, 32, hidden_dim]
                else:
                    x_flat = single_lead.reshape(B, 1, 5000)
                    h_flat = self.model.temporal_encoder(x_flat)
                    h = h_flat.reshape(B, 1, self.model.hidden_dim, 32).permute(0, 1, 3, 2)
                    tok = (h + self.model.temporal_pos_embed).reshape(B, 32, self.model.hidden_dim)

                tokens_list.append(tok)

        # [B, 12, 32, hidden_dim]
        return torch.stack(tokens_list, dim=1)

    def evaluate_subset_from_cached_tokens(
        self,
        cached_tokens: torch.Tensor,  # [B, 12, 32, hidden_dim]
        subset_lead_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Runs slot cross-attention queries over subset tokens to produce Z_S and predictions."""
        B = cached_tokens.shape[0]
        # Gather subset tokens: [B, |S|, 32, hidden_dim] -> [B, |S| * 32, hidden_dim]
        sub_tokens = cached_tokens[:, subset_lead_indices, :, :]
        flat_tokens = sub_tokens.reshape(B, len(subset_lead_indices) * 32, -1)

        with torch.no_grad():
            if hasattr(self.model, "slot_aggregator"):
                slots, z_flat = self.model.slot_aggregator(flat_tokens)
            else:
                q = self.model.query.expand(B, -1, -1)
                attn_out, _ = self.model.cross_attn(query=q, key=flat_tokens, value=flat_tokens)
                out = self.model.norm(q + attn_out).squeeze(1)
                z_flat = self.model.proj(out)

            logits = self.linear_head(z_flat) if self.linear_head is not None else None

        return z_flat, logits


def compute_information_frontiers(
    subset_results_df: pd.DataFrame,
    metric_col: str = "fixed_head_auroc",
) -> pd.DataFrame:
    """Computes Pareto information frontier: min, median, max performance vs k and rank r."""
    records = []
    # By displayed count k (1 to 12)
    for k in range(1, 13):
        sub = subset_results_df[subset_results_df["k_displayed"] == k]
        if len(sub) > 0:
            vals = sub[metric_col].values
            records.append({
                "grouping": "k_displayed",
                "value": k,
                "p_min": float(np.min(vals)),
                "p_med": float(np.median(vals)),
                "p_max": float(np.max(vals)),
                "best_subset": sub.loc[sub[metric_col].idxmax(), "lead_names_str"],
            })

    # By independent rank r (1 to 8)
    for r in range(1, 9):
        sub = subset_results_df[subset_results_df["independent_rank"] == r]
        if len(sub) > 0:
            vals = sub[metric_col].values
            records.append({
                "grouping": "independent_rank",
                "value": r,
                "p_min": float(np.min(vals)),
                "p_med": float(np.median(vals)),
                "p_max": float(np.max(vals)),
                "best_subset": sub.loc[sub[metric_col].idxmax(), "lead_names_str"],
            })

    return pd.DataFrame(records)


def compute_disease_minimal_sufficient_sets(
    subset_df: pd.DataFrame,
    full_aurocs: dict[str, float],
    delta: float = 0.02,
) -> pd.DataFrame:
    """Identifies the minimal sufficient lead set for each disease satisfying P(S) >= P(full) - delta."""
    records = []
    for disease, full_val in full_aurocs.items():
        col = f"auroc_{disease}"
        if col not in subset_df.columns:
            continue

        threshold = full_val - delta
        sufficient = subset_df[subset_df[col] >= threshold]

        if len(sufficient) > 0:
            min_k = int(sufficient["k_displayed"].min())
            min_k_subs = sufficient[sufficient["k_displayed"] == min_k]
            best_at_min_k = min_k_subs.loc[min_k_subs[col].idxmax()]

            records.append({
                "disease": disease,
                "full_auroc": full_val,
                "min_sufficient_k": min_k,
                "min_sufficient_rank": int(best_at_min_k["independent_rank"]),
                "best_minimal_subset": best_at_min_k["lead_names_str"],
                "minimal_auroc": float(best_at_min_k[col]),
                "auroc_difference": float(best_at_min_k[col] - full_val),
            })

    return pd.DataFrame(records)
