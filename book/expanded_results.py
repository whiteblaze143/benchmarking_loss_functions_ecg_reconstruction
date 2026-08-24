"""Live, read-only reconciliation helpers for the expanded experiment book.

The module intentionally distinguishes checkpoint availability, training-screen
metrics, and downstream clinical evaluation.  None of those states implies the
others.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

LOSS_COLUMNS = ["mse", "correlation", "derivative", "vcg", "energy", "lead_consistency"]
MMD_LEVELS = {
    0: "none",
    1: "global_rbf",
    2: "anatomical_laplacian",
    3: "anatomical_imq_multiscale",
    4: "temporal_kmeans_imq_multiscale",
}


def decode_mask(mask: str) -> dict:
    mask = str(mask)
    if not re.fullmatch(r"[01]{6}[0-4]", mask):
        raise ValueError(f"Expected six binary digits plus one MMD level, got {mask!r}")
    row = {name: int(value) for name, value in zip(LOSS_COLUMNS, mask[:6])}
    row.update(mask=mask, mmd_level=int(mask[6]), mmd_kernel=MMD_LEVELS[int(mask[6])])
    return row


def read_sql(path: str | Path, query: str, params=()) -> pd.DataFrame:
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not path.exists():
        return pd.DataFrame()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


def checkpoint_catalog() -> pd.DataFrame:
    frame = read_sql("results/checkpoint_store/catalog.sqlite", "select * from checkpoints")
    if frame.empty:
        return frame
    metadata = frame.metadata_json.fillna("{}").map(json.loads)
    frame["architecture"] = metadata.map(lambda x: x.get("family", "unknown"))
    frame["selector"] = metadata.map(lambda x: x.get("checkpoint_selector"))
    frame["best_metrics"] = metadata.map(lambda x: x.get("best_metrics", {}))
    frame["val_missing_mse"] = frame.best_metrics.map(lambda x: x.get("val_missing_mse"))
    frame["val_missing_pearson"] = frame.best_metrics.map(lambda x: x.get("val_missing_pearson"))
    return pd.concat([frame, frame.factorial_mask.map(decode_mask).apply(pd.Series)], axis=1)


def checkpoint_completeness(seed: int = 42) -> pd.DataFrame:
    frame = checkpoint_catalog()
    expected = {"unet": 160, "msvae": 160, "ecg_aim": 160}
    rows = []
    for architecture, n_expected in expected.items():
        block = frame[(frame.seed == seed) & (frame.architecture == architecture)]
        rows.append({
            "architecture": architecture,
            "expected": n_expected,
            "registered": block.model_id.nunique(),
            "missing": n_expected - block.model_id.nunique(),
            "coverage_pct": 100 * block.model_id.nunique() / n_expected,
            "remote_verified": int(block.status.eq("remote_verified").sum()),
            "training_metric_rows": int(block.val_missing_pearson.notna().sum()),
        })
    return pd.DataFrame(rows)


def clinical_coverage(version: str = "missing_leads_v2") -> pd.DataFrame:
    metrics = read_sql(
        "results/clinical_biomarkers_multids/clinical_metrics.db",
        "select dataset, model_id, target, evaluation_version from clinical_metrics where evaluation_version=?",
        (version,),
    )
    catalog = checkpoint_completeness()
    if metrics.empty:
        return catalog.assign(clinical_models=0, clinical_coverage_pct=0.0)
    metrics["architecture"] = np.select(
        [metrics.model_id.str.startswith("f_"),
         metrics.model_id.str.startswith("factorial_msvae_"),
         metrics.model_id.str.startswith("factorial_ecg_aim_")],
        ["unet", "msvae", "ecg_aim"], default="reference")
    counts = (metrics.query("architecture != 'reference'")
              .groupby("architecture").model_id.nunique().rename("clinical_models"))
    out = catalog.merge(counts, on="architecture", how="left").fillna({"clinical_models": 0})
    out["clinical_models"] = out.clinical_models.astype(int)
    out["clinical_coverage_pct"] = 100 * out.clinical_models / out.expected
    return out


def clinical_metrics(version: str = "missing_leads_v2") -> pd.DataFrame:
    frame = read_sql(
        "results/clinical_biomarkers_multids/clinical_metrics.db",
        "select * from clinical_metrics where evaluation_version=?", (version,))
    if frame.empty:
        return frame
    frame["architecture"] = np.select(
        [frame.model_id.str.startswith("f_"),
         frame.model_id.str.startswith("factorial_msvae_"),
         frame.model_id.str.startswith("factorial_ecg_aim_")],
        ["unet", "msvae", "ecg_aim"], default="reference")
    frame["mask"] = frame.model_id.str.extract(r"(\d{7})", expand=False)
    return frame


def onelead_catalog() -> pd.DataFrame:
    frame = read_sql("results/onelead_checkpoint_store/catalog.sqlite", "select * from checkpoints")
    if frame.empty:
        return frame
    frame["observed_lead"] = frame.observed_leads_json.map(lambda x: json.loads(x)[0])
    return frame


def wavelet_queue() -> pd.DataFrame:
    jobs = read_sql("refine-logs/wavelet_ssl_1110000/full/queue.sqlite", "select * from jobs order by ordinal")
    if jobs.empty:
        return jobs
    cells = jobs.cell_json.fillna("{}").map(json.loads).apply(pd.Series).add_prefix("cell.")
    summaries = jobs.summary_json.fillna("{}").map(json.loads).apply(pd.Series).add_prefix("metric.")
    return pd.concat([jobs.drop(columns=["cell_json", "summary_json"]), cells, summaries], axis=1)


def evidence_matrix() -> pd.DataFrame:
    checkpoints = checkpoint_completeness().set_index("architecture")
    clinical = clinical_coverage().set_index("architecture")
    rows = []
    for architecture in ["unet", "msvae", "ecg_aim"]:
        rows.append({
            "track": "three-lead seven-mask",
            "architecture": architecture,
            "trained_or_registered": int(checkpoints.loc[architecture, "registered"]),
            "target": 160,
            "missing_leads_v2_models": int(clinical.loc[architecture, "clinical_models"]),
            "claim_status": "complete clinical grid" if clinical.loc[architecture, "clinical_models"] == 160
                            else "coverage-limited; no architecture-wide clinical claim",
        })
    one = onelead_catalog()
    if not one.empty:
        for architecture, block in one.groupby("architecture"):
            rows.append({"track": "one-lead development", "architecture": architecture,
                         "trained_or_registered": block.model_id.nunique(), "target": np.nan,
                         "missing_leads_v2_models": 0,
                         "claim_status": "development evidence; not the three-lead clinical grid"})
    return pd.DataFrame(rows)
