"""Builds the PTB-XL concept hierarchy, class imbalance weights, and transfer probe tiers.

Partitions concepts into:
1. Anchor Concepts (C_anchor): Supervised shaping targets.
2. Probe Concepts (C_probe):
   - Tier P1: Directly related (parent/child or strongly coupled)
   - Tier P2: Related but distinct (same physiological family)
   - Tier P3: Ontology-distinct (no direct hierarchy edge, minimal empirical co-occurrence)

Computes class imbalance positive weights:
    pos_weight_c = N_{-, c} / N_{+, c}
strictly from training folds 1-7.
Emits configs/ptbxl_concept_tiers.yaml.
"""

from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "ptb_xl"
OUTPUT_YAML = PROJECT_ROOT / "configs" / "ptbxl_concept_tiers.yaml"
OUTPUT_YAML.parent.mkdir(parents=True, exist_ok=True)


def load_ptbxl_metadata():
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes_dict"] = df["scp_codes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})
    scp_df = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
    return df, scp_df


def build_tiers():
    df, scp_df = load_ptbxl_metadata()
    train_df = df[df["strat_fold"].isin(range(1, 8))]
    n_train = len(train_df)

    # All codes present with at least 50 occurrences in training folds
    train_code_counts = {}
    for scps in train_df["scp_codes_dict"]:
        for code in scps.keys():
            train_code_counts[code] = train_code_counts.get(code, 0) + 1

    populated_codes = {c: count for c, count in train_code_counts.items() if count >= 50 and c in scp_df.index}
    print(f"Discovered {len(populated_codes)} populated codes (>=50 occurrences in Folds 1-7).")

    # Construct one-hot binary label matrix for training folds
    sorted_codes = sorted(populated_codes.keys())
    code_to_idx = {c: i for i, c in enumerate(sorted_codes)}
    Y_train = np.zeros((n_train, len(sorted_codes)), dtype=np.float32)

    for i, scps in enumerate(train_df["scp_codes_dict"]):
        for c in scps.keys():
            if c in code_to_idx:
                Y_train[i, code_to_idx[c]] = 1.0

    # Compute positive imbalance weights: N_- / N_+
    pos_counts = Y_train.sum(axis=0)
    neg_counts = n_train - pos_counts
    pos_weights = {sorted_codes[i]: float(neg_counts[i] / max(pos_counts[i], 1.0)) for i in range(len(sorted_codes))}

    # Compute Co-occurrence Jaccard matrix
    # J(i, j) = |i and j| / |i or j|
    intersection = Y_train.T @ Y_train
    col_sums = Y_train.sum(axis=0, keepdims=True)
    union = col_sums.T + col_sums - intersection
    jaccard = intersection / np.maximum(union, 1.0)

    # Existing established 25 Anchor Concepts from P0
    anchor_concepts = [
        # Rhythm (z1)
        "PVC", "AFLT",
        # Conduction (z2)
        "ILBBB", "LPFB", "1AVB", "IVCD", "CLBBB", "LAFB",
        # Morphology / Infarction (z3)
        "RAO/RAE", "ASMI", "RVH", "LVH", "IMI", "ALMI", "LAO/LAE", "LMI", "AMI",
        # ST-T / Repolarization (z4)
        "ISCIN", "DIG", "EL", "LNGQT", "ISCAL", "ISCLA", "NST_", "ISCIL"
    ]
    # Filter to those available
    anchor_set = set(anchor_concepts).intersection(sorted_codes)
    anchor_list = [c for c in anchor_concepts if c in anchor_set]

    # Candidate probe concepts
    probe_candidates = [c for c in sorted_codes if c not in anchor_set]

    # Classify probe concepts into Tier P1, P2, P3 based on SCP metadata & co-occurrence
    tier_p1 = []  # Directly related (parent/child, strongly coupled, Jaccard >= 0.15 with an anchor)
    tier_p2 = []  # Related but distinct (same diagnostic class/family, 0.03 <= Jaccard < 0.15)
    tier_p3 = []  # Ontology-distinct (no direct hierarchy edge, Jaccard < 0.03)

    anchor_indices = [code_to_idx[a] for a in anchor_list]

    for p in probe_candidates:
        p_idx = code_to_idx[p]
        p_class = scp_df.loc[p, "diagnostic_class"] if "diagnostic_class" in scp_df.columns else ""
        p_subclass = scp_df.loc[p, "diagnostic_subclass"] if "diagnostic_subclass" in scp_df.columns else ""

        # Max Jaccard with any anchor
        max_jaccard = float(np.max(jaccard[p_idx, anchor_indices]))

        # Check hierarchy overlap
        same_subclass_anchor = any(
            (scp_df.loc[a, "diagnostic_subclass"] == p_subclass and pd.notna(p_subclass) and p_subclass != "")
            for a in anchor_list
        )
        same_class_anchor = any(
            (scp_df.loc[a, "diagnostic_class"] == p_class and pd.notna(p_class) and p_class != "")
            for a in anchor_list
        )

        entry = {
            "code": p,
            "description": str(scp_df.loc[p, "description"]) if "description" in scp_df.columns else p,
            "diagnostic_class": str(p_class),
            "diagnostic_subclass": str(p_subclass),
            "train_pos_count": int(pos_counts[p_idx]),
            "pos_weight": float(pos_weights[p]),
            "max_anchor_jaccard": round(max_jaccard, 4),
        }

        if max_jaccard >= 0.12 or same_subclass_anchor:
            tier_p1.append(entry)
        elif max_jaccard >= 0.02 or same_class_anchor:
            tier_p2.append(entry)
        else:
            tier_p3.append(entry)

    # Format anchor metadata
    anchors_meta = []
    for a in anchor_list:
        a_idx = code_to_idx[a]
        anchors_meta.append({
            "code": a,
            "description": str(scp_df.loc[a, "description"]) if "description" in scp_df.columns else a,
            "diagnostic_class": str(scp_df.loc[a, "diagnostic_class"]) if "diagnostic_class" in scp_df.columns else "",
            "diagnostic_subclass": str(scp_df.loc[a, "diagnostic_subclass"]) if "diagnostic_subclass" in scp_df.columns else "",
            "train_pos_count": int(pos_counts[a_idx]),
            "pos_weight": float(pos_weights[a]),
        })

    # Domain partition for the 25 anchor concepts
    rhythm_codes = [c for c in ["PVC", "AFLT"] if c in anchor_list]
    conduction_codes = [c for c in ["ILBBB", "LPFB", "1AVB", "IVCD", "CLBBB", "LAFB"] if c in anchor_list]
    morphology_codes = [c for c in ["RAO/RAE", "ASMI", "RVH", "LVH", "IMI", "ALMI", "LAO/LAE", "LMI", "AMI"] if c in anchor_list]
    stt_codes = [c for c in ["ISCIN", "DIG", "EL", "LNGQT", "ISCAL", "ISCLA", "NST_", "ISCIL"] if c in anchor_list]

    domain_mapping = {}
    for c in rhythm_codes: domain_mapping[c] = "rhythm"
    for c in conduction_codes: domain_mapping[c] = "conduction"
    for c in morphology_codes: domain_mapping[c] = "morphology"
    for c in stt_codes: domain_mapping[c] = "stt"

    p1_codes = [p["code"] for p in tier_p1]
    p2_codes = [p["code"] for p in tier_p2]
    p3_codes = [p["code"] for p in tier_p3]
    all_probe_codes = p1_codes + p2_codes + p3_codes

    registry = {
        "version": 2,
        "description": "GRAIL-ECG v2 Concept Hierarchy, Class Imbalance, and Transfer Tiers",
        "dataset": "PTB-XL",
        "total_populated_concepts": len(sorted_codes),
        "anchor_concepts_count": len(anchors_meta),
        "tier_p1_directly_related_count": len(tier_p1),
        "tier_p2_related_distinct_count": len(tier_p2),
        "tier_p3_ontology_distinct_count": len(tier_p3),
        "all_anchor_codes": anchor_list,
        "all_probe_codes": all_probe_codes,
        "tier_p1_codes": p1_codes,
        "tier_p2_codes": p2_codes,
        "tier_p3_codes": p3_codes,
        "anchor_counts_per_domain": {
            "rhythm": len(rhythm_codes),
            "conduction": len(conduction_codes),
            "morphology": len(morphology_codes),
            "stt": len(stt_codes),
        },
        "domain_mapping": domain_mapping,
        "anchors": anchors_meta,
        "probes": {
            "tier_p1_directly_related": tier_p1,
            "tier_p2_related_distinct": tier_p2,
            "tier_p3_ontology_distinct": tier_p3,
        },
    }

    with open(OUTPUT_YAML, "w") as f:
        yaml.dump(registry, f, sort_keys=False)

    print(f"Registry written to {OUTPUT_YAML}")
    print(f"Anchors: {len(anchors_meta)} | Tier P1: {len(tier_p1)} | Tier P2: {len(tier_p2)} | Tier P3: {len(tier_p3)}")


if __name__ == "__main__":
    build_tiers()
