"""Patient-disjoint 5-fold cross-validation split generator and verification."""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def assign_patient_disjoint_folds(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    target_col: str = "is_positive",
    n_splits: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assigns deterministic patient-disjoint folds (0 to n_splits-1).
    
    Guarantees no patient appears in more than one validation fold.
    Stratifies patient assignment by patient-level positive event rate.
    """
    df = df.copy()
    patients = sorted(df[patient_col].unique())
    n_patients = len(patients)

    if n_patients < n_splits:
        # Fallback if cohort has fewer patients than splits
        folds = {p: i % n_patients for i, p in enumerate(patients)}
        df["fold"] = df[patient_col].map(folds)
        return df

    # Compute patient-level event count for balanced stratification
    pt_summary = df.groupby(patient_col).agg(
        n_rows=(target_col, "count"),
        n_pos=(target_col, "sum")
    ).reset_index()
    pt_summary["pos_rate"] = pt_summary["n_pos"] / pt_summary["n_rows"]
    pt_summary = pt_summary.sort_values(by=["pos_rate", "n_rows"], ascending=False).reset_index(drop=True)

    # Round-robin balanced assignment to folds
    patient_to_fold = {}
    for i, row in pt_summary.iterrows():
        fold_id = i % n_splits
        patient_to_fold[row[patient_col]] = fold_id

    df["fold"] = df[patient_col].map(patient_to_fold)

    # Verification: strict patient disjointness
    for f in range(n_splits):
        val_pts = set(df[df["fold"] == f][patient_col])
        train_pts = set(df[df["fold"] != f][patient_col])
        overlap = val_pts.intersection(train_pts)
        assert len(overlap) == 0, f"Patient leakage detected in fold {f}: {overlap}"

    return df


def generate_split_manifest(
    p01_df: pd.DataFrame,
    p02_df: pd.DataFrame,
    p03_df: pd.DataFrame,
    p04_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combines patient-fold assignments across all 4 tasks into a unified audit manifest."""
    records = []
    for task_name, df in [("P01", p01_df), ("P02", p02_df), ("P03", p03_df), ("P04", p04_df)]:
        for f in sorted(df["fold"].unique()):
            sub = df[df["fold"] == f]
            records.append({
                "task": task_name,
                "fold": int(f),
                "n_samples": len(sub),
                "n_patients": sub["patient_id"].nunique(),
                "n_positive": int(sub["is_positive"].sum()),
                "positive_rate": float(sub["is_positive"].mean()),
                "cohorts": "|".join(sorted(sub["dataset"].unique())),
            })

    return pd.DataFrame(records)
