import sqlite3

conn = sqlite3.connect("results/clinical_biomarkers_multids/clinical_metrics.db")
cur = conn.cursor()

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EVALUATION_VERSION = "missing_leads_v2"

# 1. Add PTB-XL missing lead signal targets for reference (MAE=0, r=1, r2=1, bias=0)
for l in LEAD_NAMES:
    cur.execute("""
        INSERT OR REPLACE INTO clinical_metrics (
            dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
            auroc, auprc, f1, sens, spec, ppv, npv, evaluation_version
        ) VALUES ('ptb_xl', 'reference', ?, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
    """, (f"Signal_Lead_{l}", EVALUATION_VERSION))

cur.execute("""
    INSERT OR REPLACE INTO clinical_metrics (
        dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
        auroc, auprc, f1, sens, spec, ppv, npv, evaluation_version
    ) VALUES ('ptb_xl', 'reference', 'Delineation_Missing_Lead_Coverage', 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
""", (EVALUATION_VERSION,))

# 2. Add LUDB targets for reference (27 targets, matching Sunnybrook)
ludb_targets = [
    ("QRS_Overall", 0.0, 1.0, 1.0, 0.0),
    ("LVH_SokolowLyon", 0.0, 1.0, 1.0, 0.0),
    ("Signal_Missing_Leads_Pearson", 1.0, 1.0, 1.0, 0.0),
    ("Signal_Missing_Leads_MSE", 0.0, 1.0, 1.0, 0.0),
    ("Signal_Missing_Leads_SNR_dB", 100.0, 1.0, 1.0, 0.0),
    ("Signal_Missing_Leads_DTW", 0.0, 1.0, 1.0, 0.0),
    ("Morphology_P_Wave_Dice", 1.0, 1.0, 1.0, 0.0),
    ("Morphology_QRS_Wave_Dice", 1.0, 1.0, 1.0, 0.0),
    ("Morphology_T_Wave_Dice", 1.0, 1.0, 1.0, 0.0),
    ("Boundary_P_Onset_MAE_ms", 0.0, 1.0, 1.0, 0.0),
    ("Boundary_P_Offset_MAE_ms", 0.0, 1.0, 1.0, 0.0),
    ("Boundary_R_Onset_MAE_ms", 0.0, 1.0, 1.0, 0.0),
    ("Boundary_R_Offset_MAE_ms", 0.0, 1.0, 1.0, 0.0),
    ("Boundary_T_Onset_MAE_ms", 0.0, 1.0, 1.0, 0.0),
    ("Boundary_T_Offset_MAE_ms", 0.0, 1.0, 1.0, 0.0),
]
for t, mae, pr, r2, bias in ludb_targets:
    cur.execute("""
        INSERT OR REPLACE INTO clinical_metrics (
            dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
            auroc, auprc, f1, sens, spec, ppv, npv, evaluation_version
        ) VALUES ('ludb', 'reference', ?, ?, ?, ?, ?, 0.0, 0.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
    """, (t, mae, pr, r2, bias, EVALUATION_VERSION))

for l in LEAD_NAMES:
    cur.execute("""
        INSERT OR REPLACE INTO clinical_metrics (
            dataset, model_id, target, mae, pearson_r, r2, bland_bias, loa_low, loa_high,
            auroc, auprc, f1, sens, spec, ppv, npv, evaluation_version
        ) VALUES ('ludb', 'reference', ?, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, ?)
    """, (f"Signal_Lead_{l}", EVALUATION_VERSION))

conn.commit()

# Print target counts
cur.execute("SELECT dataset, COUNT(DISTINCT target) FROM clinical_metrics WHERE model_id = 'reference' GROUP BY dataset;")
print("Reference targets per dataset:", cur.fetchall())

cur.execute("SELECT COUNT(DISTINCT target) FROM clinical_metrics WHERE model_id = 'reference';")
print("Total distinct reference targets across all datasets:", cur.fetchone()[0])
conn.close()
