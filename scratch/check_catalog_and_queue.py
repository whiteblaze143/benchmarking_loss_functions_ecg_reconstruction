import sqlite3
import pandas as pd
import json
from pathlib import Path
from collections import Counter

# 1. catalog.sqlite
cat_path = Path('results/checkpoint_store/catalog.sqlite')
if cat_path.exists():
    conn = sqlite3.connect(str(cat_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(checkpoints)")
    cols = [r[1] for r in cursor.fetchall()]
    print("Columns in catalog.sqlite checkpoints table:", cols)
    
    df = pd.read_sql_query("SELECT * FROM checkpoints", conn)
    print(f"\n=== Checkpoints Catalog ({len(df)} total records) ===")
    
    def classify_id(mid):
        if 'msvae' in mid: return 'msvae'
        elif 'aim' in mid or 'ecg_aim' in mid: return 'ecg_aim'
        else: return 'unet'
        
    df['arch'] = df['model_id'].apply(classify_id)
    print(df['arch'].value_counts())
    conn.close()

# 2. queue_state.json
q_path = Path('refine-logs/queue/queue_state.json')
if q_path.exists():
    with open(q_path) as f:
        q = json.load(f)
    jobs = q.get("jobs", [])
    print(f"\n=== Training Queue State (refine-logs/queue/queue_state.json) ===")
    print(f"Total jobs in queue: {len(jobs)}")
    arch_status = Counter((j.get('architecture', 'unet'), j.get('status')) for j in jobs)
    for k, v in sorted(arch_status.items()):
        print(f"  Arch: {k[0]:<10} | Status: {k[1]:<12} | Count: {v}")

# 3. Model registry
reg_path = Path('results/clinical_biomarkers_model_registry.json')
if reg_path.exists():
    with open(reg_path) as f:
        reg = json.load(f)
    r_models = reg.get("models", [])
    print(f"\n=== Model Registry (results/clinical_biomarkers_model_registry.json) ===")
    print(f"Total models registered: {len(r_models)}")
    r_counts = Counter(m.get('kind') for m in r_models)
    for k, v in r_counts.items():
        print(f"  Kind: {k:<10} | Registered Count: {v}")
