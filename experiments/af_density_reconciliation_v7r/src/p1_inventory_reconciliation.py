"""Priority 1: Independent Dataset Inventory & Source Parity Reconciliation (V7-R)."""
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_raw_ltafdb():
    ltaf_dir = "/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference"
    recs = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(ltaf_dir, "*.hea")))
    rows = []
    for r in recs:
        hdr = wfdb.rdheader(os.path.join(ltaf_dir, r))
        ann = wfdb.rdann(os.path.join(ltaf_dir, r), "atr")
        fs = float(hdr.fs)
        dur_s = hdr.sig_len / fs
        changes = []
        for s, aux in zip(ann.sample, ann.aux_note):
            c = aux.strip().replace("\x00", "")
            if c.startswith("("):
                changes.append((int(s), c[1:]))
        af_runs_raw = sum(1 for _, lbl in changes if lbl.upper().startswith("AFIB"))
        rows.append({
            "dataset": "LTAFDB", "record_id": r, "patient_id": f"ltafdb_{r}",
            "is_annotated": len(changes) > 0, "raw_duration_h": dur_s / 3600.0,
            "raw_af_transitions": af_runs_raw,
        })
    return pd.DataFrame(rows)


def inspect_raw_shdb():
    shdb_dir = "/data/mithunmanivannan/datasets/shdb_af"
    meta = pd.read_csv(os.path.join(shdb_dir, "AdditionalData.csv"))
    recs = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(shdb_dir, "*.hea")))
    rows = []
    for r in recs:
        hdr = wfdb.rdheader(os.path.join(shdb_dir, r))
        dur_s = hdr.sig_len / float(hdr.fs)
        m_row = meta[meta["Data_ID"].astype(str) == str(int(r))]
        subj = str(m_row["Subject_ID"].values[0]) if not m_row.empty else r
        is_ann = False
        af_runs_raw = 0
        try:
            ann = wfdb.rdann(os.path.join(shdb_dir, r), "atr")
            changes = [aux.strip().replace("\x00", "")[1:] for aux in ann.aux_note if aux.strip().replace("\x00", "").startswith("(")]
            is_ann = len(changes) > 0
            af_runs_raw = sum(1 for lbl in changes if lbl.upper().startswith("AFIB"))
        except Exception:
            pass
        rows.append({
            "dataset": "SHDB-AF", "record_id": r, "patient_id": f"shdb_{subj}",
            "is_annotated": is_ann, "raw_duration_h": dur_s / 3600.0,
            "raw_af_transitions": af_runs_raw,
        })
    return pd.DataFrame(rows)


def inspect_raw_iridia():
    base = "/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1"
    meta = pd.read_csv(os.path.join(base, "iridia-af-metadata-v1.0.1.csv"))
    iv = pd.read_csv(os.path.join(base, "timeline_intervals.csv"))
    run_counts = iv.groupby("record_id").size().to_dict()
    rows = []
    for r_info in meta.itertuples():
        rid = str(r_info.record_id)
        pid = str(r_info.patient_id)
        # Check duration column cleanly
        dur_s = float(getattr(r_info, "record_duration_seconds", getattr(r_info, "duration_seconds", 0.0)))
        if dur_s == 0.0 and hasattr(r_info, "record_n_samples") and hasattr(r_info, "sampling_frequency"):
            dur_s = float(r_info.record_n_samples) / float(r_info.sampling_frequency)
        rows.append({
            "dataset": "IRIDIA-AF", "record_id": rid, "patient_id": pid,
            "is_annotated": True, "raw_duration_h": dur_s / 3600.0,
            "raw_af_transitions": int(run_counts.get(rid, 0)),
        })
    return pd.DataFrame(rows)


def main():
    print("=== Priority 1: Independent Source Dataset Inventory Reconciliation ===")
    raw_df = pd.concat([inspect_raw_ltafdb(), inspect_raw_shdb(), inspect_raw_iridia()], ignore_index=True)
    rhy = pd.read_parquet("experiments/af_density_continuity_v5/manifests/canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet("experiments/af_density_continuity_v5/manifests/canonical_af_runs.parquet")
    c5 = pd.read_csv("experiments/af_density_continuity_v5/manifests/cohort_inventory.csv")

    # Merge canonical duration summaries per (dataset, record_id)
    rhy["record_id_str"] = rhy["record_id"].astype(str)
    runs["record_id_str"] = runs["record_id"].astype(str)
    raw_df["record_id_str"] = raw_df["record_id"].astype(str)

    # Known vs Unknown rhythm durations per record
    rec_known = rhy[rhy["rhythm_known"]].groupby(["dataset", "record_id_str"])["duration_s"].sum() / 3600.0
    rec_unk = rhy[~rhy["rhythm_known"]].groupby(["dataset", "record_id_str"])["duration_s"].sum() / 3600.0
    rec_af = runs.groupby(["dataset", "record_id_str"])["duration_s"].sum() / 3600.0
    rec_n_runs = runs.groupby(["dataset", "record_id_str"]).size()
    rec_nonaf_known = rhy[rhy["rhythm_known"] & (rhy["rhythm_label_harmonized"] != "AF")].groupby(["dataset", "record_id_str"])["duration_s"].sum() / 3600.0

    # Left/right censoring per record
    rec_lc = runs.groupby(["dataset", "record_id_str"])["is_left_censored"].sum()
    rec_rc = runs.groupby(["dataset", "record_id_str"])["is_right_censored"].sum()

    # Fill raw_duration_h from c5 if metadata lacked duration column
    c5["record_id_str"] = c5["record_id"].astype(str)
    c5_dur = c5.set_index(["dataset", "record_id_str"])["duration_h"].to_dict()

    records = []
    for row in raw_df.itertuples():
        k = (row.dataset, row.record_id_str)
        raw_h = c5_dur.get(k, row.raw_duration_h)
        records.append({
            "dataset": row.dataset,
            "record_id": row.record_id_str,
            "patient_id": row.patient_id,
            "is_annotated": bool(row.is_annotated),
            "raw_recording_h": float(raw_h),
            "known_rhythm_h": float(rec_known.get(k, 0.0)),
            "unknown_rhythm_h": float(rec_unk.get(k, 0.0)),
            "af_duration_h": float(rec_af.get(k, 0.0)),
            "known_non_af_h": float(rec_nonaf_known.get(k, 0.0)),
            "raw_af_transitions": int(row.raw_af_transitions),
            "canonical_af_runs": int(rec_n_runs.get(k, 0)),
            "left_censored_runs": int(rec_lc.get(k, 0)),
            "right_censored_runs": int(rec_rc.get(k, 0)),
        })

    rec_inv = pd.DataFrame(records)
    rec_inv.to_parquet(OUT_DIR / "V7R_recording_inventory.parquet", index=False)

    # Cohort-level summary table
    summary_rows = []
    for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF", "COMBINED"]:
        sub = rec_inv if ds == "COMBINED" else rec_inv[rec_inv["dataset"] == ds]
        ann = sub[sub["is_annotated"]]
        af_pos = ann[ann["canonical_af_runs"] > 0]
        multi_af = ann[ann["canonical_af_runs"] >= 2]
        summary_rows.append({
            "dataset": ds,
            "raw_recordings": len(sub),
            "annotated_recordings": len(ann),
            "unannotated_recordings": len(sub) - len(ann),
            "unique_patients_raw": sub["patient_id"].nunique(),
            "unique_patients_annotated": ann["patient_id"].nunique(),
            "patients_with_af": af_pos["patient_id"].nunique(),
            "patients_with_multi_af_runs": multi_af["patient_id"].nunique(),
            "raw_recording_h": round(sub["raw_recording_h"].sum(), 2),
            "annotated_known_rhythm_h": round(ann["known_rhythm_h"].sum(), 2),
            "unknown_rhythm_h_total": round(sub["unknown_rhythm_h"].sum(), 2),
            "af_duration_h": round(ann["af_duration_h"].sum(), 2),
            "known_non_af_h": round(ann["known_non_af_h"].sum(), 2),
            "raw_af_transitions_sum": int(sub["raw_af_transitions"].sum()),
            "canonical_af_runs_sum": int(sub["canonical_af_runs"].sum()),
            "left_censored_runs": int(sub["left_censored_runs"].sum()),
            "right_censored_runs": int(sub["right_censored_runs"].sum()),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUT_DIR / "V7R_cohort_reconciliation.csv", index=False)
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
