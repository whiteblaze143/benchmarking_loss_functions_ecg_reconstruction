import sqlite3

db_path = "results/clinical_biomarkers_multids/clinical_metrics.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

# 1. Detailed Lead-by-Lead, Feature-by-Feature Presacan Metrics Table
c.execute("""
CREATE TABLE IF NOT EXISTS presacan_clinical_metrics (
    model_id TEXT,
    dataset TEXT DEFAULT 'ptb_xl',
    lead TEXT,
    feature TEXT,
    real_mean REAL,
    real_sd REAL,
    recon_mean REAL,
    recon_sd REAL,
    p_mean REAL,
    p_var REAL,
    var_ratio REAL,
    var_ret_pct REAL,
    bland_bias REAL,
    bland_sd REAL,
    loa_low REAL,
    loa_high REAL,
    presacan_r2 REAL,
    presacan_slope REAL,
    direct_r2 REAL,
    direct_slope REAL,
    err_p5 REAL,
    err_p10 REAL,
    err_p90 REAL,
    err_p95 REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_id, dataset, lead, feature)
)
""")

# 2. Consolidated Model-Level Presacan Summary Table
c.execute("""
CREATE TABLE IF NOT EXISTS presacan_model_summary (
    model_id TEXT PRIMARY KEY,
    dataset TEXT DEFAULT 'ptb_xl',
    v3_r_presacan_r2 REAL,
    v3_r_presacan_slope REAL,
    v3_r_var_ret_pct REAL,
    v3_r_direct_r2 REAL,
    v3_r_direct_slope REAL,
    v6_r_presacan_r2 REAL,
    v6_r_presacan_slope REAL,
    v6_r_var_ret_pct REAL,
    v6_r_direct_r2 REAL,
    v6_r_direct_slope REAL,
    v3_t_presacan_r2 REAL,
    v3_t_presacan_slope REAL,
    v3_t_var_ret_pct REAL,
    interlead_r2_real_I_V3 REAL,
    interlead_r2_recon_I_V3 REAL,
    interlead_r2_real_I_V6 REAL,
    interlead_r2_recon_I_V6 REAL,
    interlead_t_r2_real_I_V3 REAL,
    interlead_t_r2_recon_I_V3 REAL,
    spurious_coupling_ratio_v3 REAL,
    avg_precordial_var_ret_pct REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
print("Successfully created presacan_clinical_metrics and presacan_model_summary tables!")
