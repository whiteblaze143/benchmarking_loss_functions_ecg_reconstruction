import sqlite3, json, os, glob
from pathlib import Path

print("=== 1. WAVELET SSL QUEUE (refine-logs/wavelet_ssl_1110000/full/queue.sqlite) ===")
db = "refine-logs/wavelet_ssl_1110000/full/queue.sqlite"
if os.path.exists(db):
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT status, count(*) FROM jobs GROUP BY status").fetchall()
    total = conn.execute("SELECT count(*) FROM jobs").fetchone()[0]
    l0_done = conn.execute("SELECT count(*) FROM jobs WHERE id LIKE '%_l0' AND status='completed'").fetchone()[0]
    l1_done = conn.execute("SELECT count(*) FROM jobs WHERE id LIKE '%_l1' AND status='completed'").fetchone()[0]
    running = conn.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
    conn.close()
    print(f"Total jobs: {total} | Status breakdown: {dict(rows)}")
    print(f"Lead 1 (_l0): {l0_done}/60 completed")
    print(f"Lead 2 (_l1): {l1_done}/60 completed")
    print(f"Currently running: {[r[0] for r in running]}")
else:
    print("Not found.")

print("\n=== 2. 3-ARCH QUEUE (refine-logs/queue_3arch) ===")
for p in ["refine-logs/queue_3arch/queue.sqlite", "refine-logs/queue_3arch/queue_state.json"]:
    if os.path.exists(p):
        print(f"Found {p}:")
        if p.endswith(".json"):
            with open(p) as f:
                d = json.load(f)
                print(f"State keys: {list(d.keys())[:10]}")
                if "status_counts" in d:
                    print("Status counts:", d["status_counts"])
        elif p.endswith(".sqlite"):
            conn = sqlite3.connect(p)
            rows = conn.execute("SELECT status, count(*) FROM jobs GROUP BY status").fetchall()
            conn.close()
            print("Status breakdown:", dict(rows))

print("\n=== 3. ONELEAD RDB EVALUATION (results/onelead_rdb_semiseg_blinded/compact.sqlite) ===")
db_rdb = "results/onelead_rdb_semiseg_blinded/compact.sqlite"
if os.path.exists(db_rdb):
    conn = sqlite3.connect(db_rdb)
    completed = conn.execute("SELECT count(*) FROM evaluations WHERE status='complete'").fetchone()[0]
    models = conn.execute("SELECT model_id FROM evaluations WHERE status='complete'").fetchall()
    conn.close()
    print(f"Evaluations in DB: {completed}")
    print(f"Models: {[m[0] for m in models]}")
else:
    print("Not found.")

print("\n=== 4. OTHER QUEUES IN refine-logs/ ===")
for q in glob.glob("refine-logs/**/queue.sqlite", recursive=True):
    if "wavelet_ssl_1110000" in q or "queue_3arch" in q: continue
    conn = sqlite3.connect(q)
    rows = conn.execute("SELECT status, count(*) FROM jobs GROUP BY status").fetchall()
    conn.close()
    print(f"{q} -> {dict(rows)}")
