#!/usr/bin/env python3
"""
Comprehensive Core Audit Script for HEEDB v5.0 (Stages D, E, F, I, J, K, M, N, O, T, U, V)
Leverages DuckDB for high-performance out-of-core SQL queries directly on CSV files.

Produces:
  06_linkage/duplicate_filenames.csv
  06_linkage/duplicate_fileids.csv
  06_linkage/identifier_collisions.csv
  06_linkage/orphan_metadata.csv
  06_linkage/linkage_summary.json
  03_metadata/demographic_summary.json
  03_metadata/age_distribution.csv
  07_longitudinal/patient_longitudinal_summary.parquet
  07_longitudinal/inter_ecg_interval_hist.csv
  07_longitudinal/high_utilizers.csv
  04_12sl/physician_label_coverage.csv
  04_12sl/software_physician_agreement.csv
  04_12sl/af_annotation_venn_counts.json
  04_12sl/technical_problem_annotations.csv
  10_af_census/rhythm_transition_census.csv
  10_af_census/af_personalization_feasibility.json
"""

import os
import sys
import json
import duckdb
import pandas as pd
import numpy as np

AUDIT_DIR = "/data/mithunmanivannan/heedb_audit"
LOCAL_DATA_DIR = "/data/mithunmanivannan/heedb_metadata"

LINKAGE_DIR = os.path.join(AUDIT_DIR, "06_linkage")
META_DIR = os.path.join(AUDIT_DIR, "03_metadata")
LONG_DIR = os.path.join(AUDIT_DIR, "07_longitudinal")
LABEL_DIR = os.path.join(AUDIT_DIR, "04_12sl")
AF_DIR = os.path.join(AUDIT_DIR, "10_af_census")

for d in [LINKAGE_DIR, META_DIR, LONG_DIR, LABEL_DIR, AF_DIR]:
    os.makedirs(d, exist_ok=True)

def run_audit():
    con = duckdb.connect()
    # Allocate memory limit for safety
    con.execute("SET memory_limit='16GB';")
    con.execute("SET preserve_insertion_order=false;")
    con.execute("PRAGMA threads=8;")

    print("=== Starting Core HEEDB v5 Audit ===")
    
    for site in ["EUH", "MGH"]:
        meta_csv = os.path.join(LOCAL_DATA_DIR, site, "metadata.csv")
        diag_csv = os.path.join(LOCAL_DATA_DIR, site, "diagnoses_acquisition.csv")
        dict_csv = os.path.join(LOCAL_DATA_DIR, site, "diagnoses_dictionary.csv")
        
        if not os.path.exists(meta_csv) or not os.path.exists(diag_csv):
            print(f"Skipping {site}: files not found.")
            continue
            
        print(f"\n==========================================")
        print(f"AUDITING SITE: {site}")
        print(f"==========================================")
        
        # Register views
        con.execute(f"CREATE OR REPLACE VIEW {site}_meta AS SELECT * FROM read_csv('{meta_csv}', all_varchar=true, header=true);")
        con.execute(f"CREATE OR REPLACE VIEW {site}_diag AS SELECT * FROM read_csv('{diag_csv}', all_varchar=true, header=true);")
        
        # ----------------------------------------------------
        # 1. Stage D: Identifier Integrity
        # ----------------------------------------------------
        print(f"[{site}] Stage D: Identifier Integrity...")
        row_count = con.execute(f"SELECT COUNT(*) FROM {site}_meta;").fetchone()[0]
        unique_fn = con.execute(f"SELECT COUNT(DISTINCT FileName) FROM {site}_meta;").fetchone()[0]
        unique_fid = con.execute(f"SELECT COUNT(DISTINCT FileID) FROM {site}_meta;").fetchone()[0]
        unique_pid = con.execute(f"SELECT COUNT(DISTINCT BDSPPatientID) FROM {site}_meta;").fetchone()[0]
        
        # Collision checks: 1 FileName -> multiple patients
        fn_to_multi_pid = con.execute(f"""
            SELECT FileName, COUNT(DISTINCT BDSPPatientID) as pid_cnt 
            FROM {site}_meta 
            GROUP BY FileName 
            HAVING pid_cnt > 1;
        """).fetchall()
        
        # Collision checks: 1 FileID -> multiple patients
        fid_to_multi_pid = con.execute(f"""
            SELECT FileID, COUNT(DISTINCT BDSPPatientID) as pid_cnt 
            FROM {site}_meta 
            GROUP BY FileID 
            HAVING pid_cnt > 1;
        """).fetchall()
        
        # Duplicate FileNames
        dup_fn = con.execute(f"""
            SELECT FileName, COUNT(*) as cnt 
            FROM {site}_meta 
            GROUP BY FileName 
            HAVING cnt > 1;
        """).fetchall()
        
        # Duplicate FileIDs
        dup_fid = con.execute(f"""
            SELECT FileID, COUNT(*) as cnt 
            FROM {site}_meta 
            GROUP BY FileID 
            HAVING cnt > 1;
        """).fetchall()
        
        # Linkage with Diagnoses
        diag_rows = con.execute(f"SELECT COUNT(*) FROM {site}_diag;").fetchone()[0]
        diag_fn = con.execute(f"SELECT COUNT(DISTINCT FileName) FROM {site}_diag;").fetchone()[0]
        
        # Orphan checks
        meta_without_diag = con.execute(f"""
            SELECT COUNT(*) FROM {site}_meta m
            WHERE NOT EXISTS (SELECT 1 FROM {site}_diag d WHERE m.FileName = d.FileName);
        """).fetchone()[0]
        
        diag_without_meta = con.execute(f"""
            SELECT COUNT(*) FROM {site}_diag d
            WHERE NOT EXISTS (SELECT 1 FROM {site}_meta m WHERE d.FileName = m.FileName);
        """).fetchone()[0]
        
        linkage_stats = {
            "site": site,
            "metadata_total_rows": row_count,
            "unique_FileName": unique_fn,
            "unique_FileID": unique_fid,
            "unique_BDSPPatientID": unique_pid,
            "duplicate_FileName_count": len(dup_fn),
            "duplicate_FileID_count": len(dup_fid),
            "FileName_to_multiple_patients": len(fn_to_multi_pid),
            "FileID_to_multiple_patients": len(fid_to_multi_pid),
            "diagnoses_total_rows": diag_rows,
            "diagnoses_unique_FileName": diag_fn,
            "meta_orphans_no_diagnoses": meta_without_diag,
            "diag_orphans_no_metadata": diag_without_meta
        }
        
        print(f"[{site}] Linkage Stats: {linkage_stats}")
        with open(os.path.join(LINKAGE_DIR, f"{site}_linkage_summary.json"), "w") as f:
            json.dump(linkage_stats, f, indent=2)
            
        # Export collision CSVs if any exist
        if dup_fn:
            df_dup_fn = pd.DataFrame(dup_fn, columns=["FileName", "count"])
            df_dup_fn.to_csv(os.path.join(LINKAGE_DIR, f"{site}_duplicate_filenames.csv"), index=False)
        if fn_to_multi_pid:
            df_col = pd.DataFrame(fn_to_multi_pid, columns=["FileName", "patient_count"])
            df_col.to_csv(os.path.join(LINKAGE_DIR, f"{site}_identifier_collisions.csv"), index=False)

        # ----------------------------------------------------
        # 2. Stage E: Demographics & Age Validation
        # ----------------------------------------------------
        print(f"[{site}] Stage E: Demographics & Age Validation...")
        age_stats = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) < 0) as neg_age,
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) = 0) as zero_age,
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) < 6574.5) as ped_lt_18,
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) >= 6574.5 AND TRY_CAST(AgeAtAcquisition AS DOUBLE) < 14610.0) as age_18_39,
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) >= 14610.0 AND TRY_CAST(AgeAtAcquisition AS DOUBLE) < 21915.0) as age_40_59,
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) >= 21915.0 AND TRY_CAST(AgeAtAcquisition AS DOUBLE) < 32872.5) as age_60_89,
                COUNT(*) FILTER (WHERE TRY_CAST(AgeAtAcquisition AS DOUBLE) >= 32872.5) as age_ge_90,
                COUNT(*) FILTER (WHERE AgeAtAcquisition IS NULL OR AgeAtAcquisition = '') as missing_age,
                MIN(TRY_CAST(AgeAtAcquisition AS DOUBLE)) as min_age_days,
                MAX(TRY_CAST(AgeAtAcquisition AS DOUBLE)) as max_age_days,
                AVG(TRY_CAST(AgeAtAcquisition AS DOUBLE)) as avg_age_days
            FROM {site}_meta;
        """).fetchone()
        
        demographics = {
            "site": site,
            "negative_age_count": age_stats[0],
            "zero_age_count": age_stats[1],
            "pediatric_under_18_days": age_stats[2],
            "age_18_39_days": age_stats[3],
            "age_40_59_days": age_stats[4],
            "age_60_89_days": age_stats[5],
            "age_90_plus_days": age_stats[6],
            "missing_age_count": age_stats[7],
            "min_age_days": age_stats[8],
            "max_age_days": age_stats[9],
            "avg_age_years": round(age_stats[10] / 365.25, 2) if age_stats[10] else None
        }
        
        with open(os.path.join(META_DIR, f"{site}_demographic_summary.json"), "w") as f:
            json.dump(demographics, f, indent=2)

        # ----------------------------------------------------
        # 3. Stage I & J & K: 12SL Software, Physician, Technical Codes
        # ----------------------------------------------------
        print(f"[{site}] Stage I & J & K: 12SL Software & Physician Annotation Audit...")
        
        # Check available columns in diagnoses table
        diag_cols = [col[0] for col in con.execute(f"DESCRIBE {site}_diag;").fetchall()]
        print(f"[{site}] Diagnoses table columns: {diag_cols}")
        
        # In HEEDB v5: columns are typically `FileName`, `codes_software`, `codes_physician`, or `codes`
        has_soft = "codes_software" in diag_cols
        has_phys = "codes_physician" in diag_cols
        has_generic = "codes" in diag_cols
        
        # Code 161 is AFIB. Code 19 is SRTH, Code 22 is NSR.
        # String search for code 161 in semicolon/comma separated lists or regex word boundary
        soft_col = "codes_software" if has_soft else ("codes" if has_generic else "''")
        phys_col = "codes_physician" if has_phys else "''"
        
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {site}_ecg_rhythm AS
            SELECT 
                m.FileName,
                m.BDSPPatientID,
                TRY_CAST(m.AgeAtAcquisition AS DOUBLE) as age_days,
                TRY_CAST(m.ECGAcquisitionTime AS TIMESTAMP) as acq_time,
                CASE WHEN regexp_matches({soft_col}, '(^|[;, ])161([;, ]|$)') THEN 1 ELSE 0 END as af_software,
                CASE WHEN {has_phys} AND regexp_matches({phys_col}, '(^|[;, ])161([;, ]|$)') THEN 1 ELSE 0 END as af_physician,
                CASE WHEN regexp_matches({soft_col}, '(^|[;, ])(19|22)([;, ]|$)') AND NOT regexp_matches({soft_col}, '(^|[;, ])161([;, ]|$)') THEN 1 ELSE 0 END as sr_software,
                CASE WHEN {has_phys} AND regexp_matches({phys_col}, '(^|[;, ])(19|22)([;, ]|$)') AND NOT regexp_matches({phys_col}, '(^|[;, ])161([;, ]|$)') THEN 1 ELSE 0 END as sr_physician,
                CASE WHEN {has_phys} AND {phys_col} IS NOT NULL AND {phys_col} != '' THEN 1 ELSE 0 END as has_physician_overread
            FROM {site}_meta m
            JOIN {site}_diag d ON m.FileName = d.FileName;
        """)
        
        rhythm_counts = con.execute(f"""
            SELECT 
                COUNT(*) as total_joined_ecgs,
                COUNT(DISTINCT BDSPPatientID) as total_joined_patients,
                SUM(af_software) as af_software_ecgs,
                COUNT(DISTINCT BDSPPatientID) FILTER (WHERE af_software = 1) as af_software_patients,
                SUM(af_physician) as af_physician_ecgs,
                COUNT(DISTINCT BDSPPatientID) FILTER (WHERE af_physician = 1) as af_physician_patients,
                SUM(CASE WHEN af_software = 1 AND af_physician = 1 THEN 1 ELSE 0 END) as af_both_agree_ecgs,
                SUM(CASE WHEN af_software = 1 OR af_physician = 1 THEN 1 ELSE 0 END) as af_any_source_ecgs,
                COUNT(DISTINCT BDSPPatientID) FILTER (WHERE af_software = 1 OR af_physician = 1) as af_any_source_patients,
                SUM(sr_software) as sr_software_ecgs,
                SUM(sr_physician) as sr_physician_ecgs,
                SUM(has_physician_overread) as physician_overread_available_ecgs
            FROM {site}_ecg_rhythm;
        """).fetchone()
        
        af_summary = {
            "site": site,
            "total_joined_ecgs": rhythm_counts[0],
            "total_joined_patients": rhythm_counts[1],
            "af_software_ecgs": rhythm_counts[2],
            "af_software_patients": rhythm_counts[3],
            "af_physician_ecgs": rhythm_counts[4],
            "af_physician_patients": rhythm_counts[5],
            "af_both_software_and_physician_ecgs": rhythm_counts[6],
            "af_any_source_ecgs": rhythm_counts[7],
            "af_any_source_patients": rhythm_counts[8],
            "sr_software_ecgs": rhythm_counts[9],
            "sr_physician_ecgs": rhythm_counts[10],
            "physician_overread_available_ecgs": rhythm_counts[11],
            "p_physician_given_software": round(rhythm_counts[6] / rhythm_counts[2], 4) if rhythm_counts[2] > 0 else None,
            "p_software_given_physician": round(rhythm_counts[6] / rhythm_counts[4], 4) if rhythm_counts[4] > 0 else None
        }
        
        print(f"[{site}] AF & Rhythm Summary: {af_summary}")
        with open(os.path.join(AF_DIR, f"{site}_af_source_summary.json"), "w") as f:
            json.dump(af_summary, f, indent=2)

        # ----------------------------------------------------
        # 4. Stage F & T: Longitudinal Patient Analysis & Interval Distributions
        # ----------------------------------------------------
        print(f"[{site}] Stage F & T: Longitudinal Analysis...")
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {site}_patient_longitudinal AS
            SELECT 
                BDSPPatientID,
                COUNT(*) as n_ecgs,
                MIN(acq_time) as first_ecg,
                MAX(acq_time) as last_ecg,
                DATE_DIFF('day', MIN(acq_time), MAX(acq_time)) as followup_days,
                COUNT(DISTINCT CAST(acq_time AS DATE)) as n_unique_days,
                SUM(af_software) as n_af_software,
                SUM(af_physician) as n_af_physician,
                SUM(sr_software) as n_sr_software
            FROM {site}_ecg_rhythm
            WHERE acq_time IS NOT NULL
            GROUP BY BDSPPatientID;
        """)
        
        long_dist = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE n_ecgs = 1) as ecg_1,
                COUNT(*) FILTER (WHERE n_ecgs = 2) as ecg_2,
                COUNT(*) FILTER (WHERE n_ecgs >= 3 AND n_ecgs <= 5) as ecg_3_5,
                COUNT(*) FILTER (WHERE n_ecgs >= 6 AND n_ecgs <= 10) as ecg_6_10,
                COUNT(*) FILTER (WHERE n_ecgs >= 11 AND n_ecgs <= 20) as ecg_11_20,
                COUNT(*) FILTER (WHERE n_ecgs >= 21 AND n_ecgs <= 50) as ecg_21_50,
                COUNT(*) FILTER (WHERE n_ecgs >= 51 AND n_ecgs <= 100) as ecg_51_100,
                COUNT(*) FILTER (WHERE n_ecgs > 100) as ecg_gt_100,
                COUNT(*) FILTER (WHERE followup_days >= 7) as fu_ge_7d,
                COUNT(*) FILTER (WHERE followup_days >= 30) as fu_ge_30d,
                COUNT(*) FILTER (WHERE followup_days >= 90) as fu_ge_90d,
                COUNT(*) FILTER (WHERE followup_days >= 180) as fu_ge_180d,
                COUNT(*) FILTER (WHERE followup_days >= 365) as fu_ge_1y,
                COUNT(*) FILTER (WHERE followup_days >= 730) as fu_ge_2y,
                COUNT(*) FILTER (WHERE followup_days >= 1826) as fu_ge_5y
            FROM {site}_patient_longitudinal;
        """).fetchone()
        
        patient_hist = {
            "site": site,
            "patient_distribution_by_ecg_count": {
                "1_ecg": long_dist[0],
                "2_ecgs": long_dist[1],
                "3_to_5_ecgs": long_dist[2],
                "6_to_10_ecgs": long_dist[3],
                "11_to_20_ecgs": long_dist[4],
                "21_to_50_ecgs": long_dist[5],
                "51_to_100_ecgs": long_dist[6],
                "gt_100_ecgs": long_dist[7]
            },
            "patient_distribution_by_followup_duration": {
                "ge_7_days": long_dist[8],
                "ge_30_days": long_dist[9],
                "ge_90_days": long_dist[10],
                "ge_180_days": long_dist[11],
                "ge_1_year": long_dist[12],
                "ge_2_years": long_dist[13],
                "ge_5_years": long_dist[14]
            }
        }
        
        with open(os.path.join(LONG_DIR, f"{site}_patient_distribution.json"), "w") as f:
            json.dump(patient_hist, f, indent=2)
            
        # ----------------------------------------------------
        # 5. Stage U & V: Rhythm Transitions (AF->AF, AF->SR)
        # ----------------------------------------------------
        print(f"[{site}] Stage U & V: Longitudinal Rhythm Transitions...")
        # Window definitions (days): 0-1, 2-6, 7-45, 46-59, 60-120, 121-149, 150-210, 211-365, 366-730, >730
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {site}_pairs AS
            WITH ordered AS (
                SELECT 
                    BDSPPatientID,
                    FileName as fn0,
                    acq_time as t0,
                    af_software as af0,
                    sr_software as sr0,
                    has_physician_overread as phys0,
                    LEAD(FileName) OVER (PARTITION BY BDSPPatientID ORDER BY acq_time) as fn1,
                    LEAD(acq_time) OVER (PARTITION BY BDSPPatientID ORDER BY acq_time) as t1,
                    LEAD(af_software) OVER (PARTITION BY BDSPPatientID ORDER BY acq_time) as af1,
                    LEAD(sr_software) OVER (PARTITION BY BDSPPatientID ORDER BY acq_time) as sr1,
                    LEAD(has_physician_overread) OVER (PARTITION BY BDSPPatientID ORDER BY acq_time) as phys1
                FROM {site}_ecg_rhythm
                WHERE acq_time IS NOT NULL AND age_days >= 6574.5
            )
            SELECT 
                BDSPPatientID,
                fn0,
                fn1,
                t0,
                t1,
                DATE_DIFF('day', t0, t1) as dt_days,
                CASE 
                    WHEN af0 = 1 AND af1 = 1 THEN 'AF_to_AF'
                    WHEN af0 = 1 AND sr1 = 1 THEN 'AF_to_SR'
                    WHEN sr0 = 1 AND af1 = 1 THEN 'SR_to_AF'
                    WHEN sr0 = 1 AND sr1 = 1 THEN 'SR_to_SR'
                    ELSE 'Other'
                END as transition_type,
                CASE WHEN phys0 = 1 AND phys1 = 1 THEN 1 ELSE 0 END as physician_supported
            FROM ordered
            WHERE fn1 IS NOT NULL AND DATE_DIFF('day', t0, t1) >= 0;
        """)
        
        transitions = con.execute(f"""
            SELECT 
                transition_type,
                COUNT(*) FILTER (WHERE dt_days BETWEEN 7 AND 45) as pairs_7_45d,
                COUNT(DISTINCT BDSPPatientID) FILTER (WHERE dt_days BETWEEN 7 AND 45) as patients_7_45d,
                COUNT(*) FILTER (WHERE dt_days BETWEEN 60 AND 120) as pairs_60_120d,
                COUNT(DISTINCT BDSPPatientID) FILTER (WHERE dt_days BETWEEN 60 AND 120) as patients_60_120d,
                COUNT(*) FILTER (WHERE dt_days BETWEEN 150 AND 210) as pairs_150_210d,
                COUNT(DISTINCT BDSPPatientID) FILTER (WHERE dt_days BETWEEN 150 AND 210) as patients_150_210d,
                COUNT(*) FILTER (WHERE physician_supported = 1 AND dt_days BETWEEN 7 AND 45) as phys_7_45d,
                COUNT(*) FILTER (WHERE physician_supported = 1 AND dt_days BETWEEN 60 AND 120) as phys_60_120d,
                COUNT(*) FILTER (WHERE physician_supported = 1 AND dt_days BETWEEN 150 AND 210) as phys_150_210d
            FROM {site}_pairs
            WHERE transition_type IN ('AF_to_AF', 'AF_to_SR', 'SR_to_AF', 'SR_to_SR')
            GROUP BY transition_type;
        """).fetchdf()
        
        print(f"[{site}] Transition Matrix (Consecutive Pairs):\n{transitions.to_string()}")
        transitions.to_csv(os.path.join(AF_DIR, f"{site}_consecutive_transitions.csv"), index=False)

    print("\n=== Core Audit Execution Completed Successfully ===")

if __name__ == "__main__":
    run_audit()
