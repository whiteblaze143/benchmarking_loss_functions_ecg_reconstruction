import json
import sqlite3
from pathlib import Path
import os

ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
invalid_job_ids = ["ecg_aim_f_1000001_s42", "ecg_aim_f_1000002_s42"]

def clean_db(db_path):
    if not db_path.exists():
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cur.fetchall()]
        print(f"{db_path.name} tables: {tables}")
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = [r[1] for r in cur.fetchall()]
            if 'model_name' in columns:
                for job_id in invalid_job_ids:
                    cur.execute(f"DELETE FROM {table} WHERE model_name LIKE ?", (f"{job_id}%",))
                    print(f"Deleted {job_id} from {db_path.name}.{table} (rows affected: {cur.rowcount})")
            elif 'id' in columns:
                for job_id in invalid_job_ids:
                    cur.execute(f"DELETE FROM {table} WHERE id LIKE ?", (f"%{job_id}%",))
                    print(f"Deleted {job_id} from {db_path.name}.{table} (rows affected: {cur.rowcount})")
        conn.commit()
    except Exception as e:
        print(f"Error handling {db_path.name}: {e}")
    finally:
        conn.close()

clean_db(ROOT / "results/clinical_biomarkers_multids/clinical_metrics.db")
clean_db(ROOT / "results/checkpoint_store/catalog.sqlite")

print("Done resetting.")
