#!/usr/bin/env python3
"""
Stage C: Schema Audit and Dictionary Cross-Site Comparison for HEEDB v5.0
Produces:
  02_schema/schema_registry.json
  02_schema/mgh_vs_euh_schema_diff.md
  04_12sl/canonical_dictionary.csv
  04_12sl/dictionary_comparison.md
Evaluates: Gate C
"""

import os
import sys
import json
import duckdb
import pandas as pd

AUDIT_DIR = "/data/mithunmanivannan/heedb_audit"
SCHEMA_DIR = os.path.join(AUDIT_DIR, "02_schema")
LABEL_DIR = os.path.join(AUDIT_DIR, "04_12sl")
LOCAL_DATA_DIR = "/data/mithunmanivannan/heedb_metadata"

os.makedirs(SCHEMA_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

TABLES = [
    {"site": "EUH", "table": "metadata", "path": os.path.join(LOCAL_DATA_DIR, "EUH", "metadata.csv")},
    {"site": "EUH", "table": "diagnoses_acquisition", "path": os.path.join(LOCAL_DATA_DIR, "EUH", "diagnoses_acquisition.csv")},
    {"site": "EUH", "table": "diagnoses_dictionary", "path": os.path.join(LOCAL_DATA_DIR, "EUH", "diagnoses_dictionary.csv")},
    {"site": "MGH", "table": "metadata", "path": os.path.join(LOCAL_DATA_DIR, "MGH", "metadata.csv")},
    {"site": "MGH", "table": "diagnoses_acquisition", "path": os.path.join(LOCAL_DATA_DIR, "MGH", "diagnoses_acquisition.csv")},
    {"site": "MGH", "table": "diagnoses_dictionary", "path": os.path.join(LOCAL_DATA_DIR, "MGH", "diagnoses_dictionary.csv")},
]

def audit_table_schema(con, site, table_name, csv_path):
    print(f"Auditing schema for {site} - {table_name} ({csv_path})...")
    if not os.path.exists(csv_path):
        return []
    
    # Use duckdb to describe and profile
    desc = con.execute(f"DESCRIBE SELECT * FROM read_csv('{csv_path}', all_varchar=true, header=true, auto_detect=true)").fetchall()
    
    # Get total row count
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_csv('{csv_path}', all_varchar=true, header=true)").fetchone()[0]
    
    records = []
    for col_info in desc:
        col_name = col_info[0]
        raw_type = col_info[1]
        
        # Profile column
        stats = con.execute(f"""
            SELECT 
                COUNT(*) FILTER (WHERE "{col_name}" IS NULL OR "{col_name}" = '') AS missing_n,
                COUNT(DISTINCT "{col_name}") AS unique_n,
                MIN("{col_name}") AS min_val,
                MAX("{col_name}") AS max_val
            FROM read_csv('{csv_path}', all_varchar=true, header=true)
        """).fetchone()
        
        missing_n = stats[0]
        unique_n = stats[1]
        min_val = str(stats[2]) if stats[2] is not None else None
        max_val = str(stats[3]) if stats[3] is not None else None
        missing_pct = round((missing_n / total_rows) * 100, 4) if total_rows > 0 else 0.0
        
        records.append({
            "table": table_name,
            "column": col_name,
            "site": site,
            "dtype_raw": raw_type,
            "dtype_parsed": raw_type,
            "total_rows": total_rows,
            "unique_n": unique_n,
            "missing_n": missing_n,
            "missing_pct": missing_pct,
            "min_sample": min_val[:100] if min_val else None,
            "max_sample": max_val[:100] if max_val else None,
            "derived": False
        })
        
    return records

def audit_dictionaries(con):
    print("Auditing 12SL diagnostic dictionaries...")
    euh_dict_path = os.path.join(LOCAL_DATA_DIR, "EUH", "diagnoses_dictionary.csv")
    mgh_dict_path = os.path.join(LOCAL_DATA_DIR, "MGH", "diagnoses_dictionary.csv")
    
    df_euh = pd.read_csv(euh_dict_path)
    df_mgh = pd.read_csv(mgh_dict_path)
    
    # Columns in HEEDB v5 are: codes, acronym, diagnoses
    code_col = "codes" if "codes" in df_euh.columns else "code"
    diag_col = "diagnoses" if "diagnoses" in df_euh.columns else "description"
    
    df_euh["site"] = "EUH"
    df_mgh["site"] = "MGH"
    
    # Standardize column names for canonical output: site, code, acronym, description, source
    df_euh_std = df_euh.rename(columns={code_col: "code", diag_col: "description"}).copy()
    df_mgh_std = df_mgh.rename(columns={code_col: "code", diag_col: "description"}).copy()
    df_euh_std["source"] = "EUH_dictionary"
    df_mgh_std["source"] = "MGH_dictionary"
    
    euh_codes = set(df_euh_std["code"].unique())
    mgh_codes = set(df_mgh_std["code"].unique())
    
    shared = euh_codes.intersection(mgh_codes)
    euh_only = euh_codes - mgh_codes
    mgh_only = mgh_codes - euh_codes
    
    diff_md = f"""# 12SL Dictionary Cross-Site Comparison

- **EUH Total Codes**: {len(euh_codes)}
- **MGH Total Codes**: {len(mgh_codes)}
- **Shared Codes**: {len(shared)}
- **EUH-Only Codes**: {len(euh_only)} ({sorted(list(euh_only))[:10]}...)
- **MGH-Only Codes**: {len(mgh_only)} ({sorted(list(mgh_only))[:10]}...)

## Key Rhythm Code Verification
- **Code 161 (Atrial Fibrillation)**:
  - EUH: `{df_euh_std[df_euh_std['code']==161]['description'].values.tolist()}`
  - MGH: `{df_mgh_std[df_mgh_std['code']==161]['description'].values.tolist()}`
- **Sinus Rhythm Codes**:
  - Code 19 (Sinus Rhythm): EUH={19 in euh_codes}, MGH={19 in mgh_codes}
  - Code 22 (Normal Sinus Rhythm): EUH={22 in euh_codes}, MGH={22 in mgh_codes}
  - Code 20 (Sinus Bradycardia): EUH={20 in euh_codes}, MGH={20 in mgh_codes}
  - Code 21 (Sinus Tachycardia): EUH={21 in euh_codes}, MGH={21 in mgh_codes}
"""
    with open(os.path.join(LABEL_DIR, "dictionary_comparison.md"), "w") as f:
        f.write(diff_md)
        
    # Combine canonical dictionary
    df_canon = pd.concat([df_euh_std, df_mgh_std], ignore_index=True).drop_duplicates(subset=["code", "site"])
    df_canon = df_canon[["site", "code", "acronym", "description", "source"]]
    df_canon.to_csv(os.path.join(LABEL_DIR, "canonical_dictionary.csv"), index=False)
    print("Saved canonical dictionary and dictionary comparison.")

def main():
    con = duckdb.connect()
    
    all_schema_records = []
    mgh_cols = {}
    euh_cols = {}
    
    for t in TABLES:
        records = audit_table_schema(con, t["site"], t["table"], t["path"])
        all_schema_records.extend(records)
        cols = [r["column"] for r in records]
        if t["site"] == "MGH":
            mgh_cols[t["table"]] = cols
        else:
            euh_cols[t["table"]] = cols
            
    # Write schema_registry.json
    registry_path = os.path.join(SCHEMA_DIR, "schema_registry.json")
    with open(registry_path, "w") as f:
        json.dump(all_schema_records, f, indent=2)
    print(f"Saved schema registry with {len(all_schema_records)} columns to {registry_path}")
    
    # Generate mgh_vs_euh_schema_diff.md
    diff_md = "# MGH vs EUH Schema Differences\n\n"
    for table_name in set(mgh_cols.keys()).union(euh_cols.keys()):
        m_c = set(mgh_cols.get(table_name, []))
        e_c = set(euh_cols.get(table_name, []))
        shared = m_c.intersection(e_c)
        m_only = m_c - e_c
        e_only = e_c - m_c
        
        diff_md += f"## Table: `{table_name}`\n"
        diff_md += f"- **Shared Columns ({len(shared)})**: {', '.join(sorted(shared))}\n"
        diff_md += f"- **MGH-Only Columns ({len(m_only)})**: {', '.join(sorted(m_only)) if m_only else 'None'}\n"
        diff_md += f"- **EUH-Only Columns ({len(e_only)})**: {', '.join(sorted(e_only)) if e_only else 'None'}\n\n"
        
    with open(os.path.join(SCHEMA_DIR, "mgh_vs_euh_schema_diff.md"), "w") as f:
        f.write(diff_md)
        
    # Audit Dictionaries
    audit_dictionaries(con)
    
    # Gate C evaluation
    required_keys = ["BDSPPatientID", "FileName", "FileID", "ECGAcquisitionTime", "AgeAtAcquisition"]
    missing_gate_keys = []
    for rk in required_keys:
        for site in ["MGH", "EUH"]:
            found = any(r["column"] == rk and r["site"] == site for r in all_schema_records)
            if not found:
                missing_gate_keys.append(f"{site}.{rk}")
                
    gate_c_status = "PASS" if not missing_gate_keys else "FAIL"
    print(f"Gate C Status: {gate_c_status}")
    if missing_gate_keys:
        print(f"Missing required Gate C keys: {missing_gate_keys}")

if __name__ == "__main__":
    main()
