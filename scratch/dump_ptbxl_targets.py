import sqlite3

conn = sqlite3.connect('results/clinical_biomarkers_multids/clinical_metrics.db')
c = conn.cursor()
c.execute('SELECT DISTINCT target FROM clinical_metrics WHERE dataset="ptb_xl" AND evaluation_version="missing_leads_v2"')
targets = sorted([r[0] for r in c.fetchall()])
conn.close()

print(f"Total PTB-XL targets: {len(targets)}")
print("\nFoundation Model / Diagnostic Targets:")
for t in targets:
    if not t.startswith('Signal_') and not t.startswith('R_') and not t.startswith('S_') and not t.startswith('T_') and not t.startswith('ST_'):
        print(f"  - {t}")

print("\nBiomarker / Signal Targets:")
for t in targets:
    if t.startswith('Signal_') or t.startswith('R_') or t.startswith('S_') or t.startswith('T_') or t.startswith('ST_') or t in ['QRS_Overall', 'LVH_SokolowLyon']:
        print(f"  - {t}")
