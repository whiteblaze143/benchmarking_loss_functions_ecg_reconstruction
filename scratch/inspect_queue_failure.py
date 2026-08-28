import sqlite3, json

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
cursor = conn.cursor()

# Overall status counts
counts = cursor.execute("SELECT status, count(*) FROM jobs GROUP BY status").fetchall()
print("=== QUEUE STATUS BREAKDOWN ===")
for st, cnt in counts:
    print(f"  {st:<15}: {cnt}")

# Inspect failed jobs
failed_jobs = cursor.execute("SELECT id, attempts, returncode, error, log_path FROM jobs WHERE status LIKE '%fail%'").fetchall()
print(f"\n=== FAILED JOBS ({len(failed_jobs)}) ===")
for jid, att, rc, err, lpath in failed_jobs:
    print(f"ID: {jid}")
    print(f"  Attempts: {att} | ReturnCode: {rc}")
    print(f"  Error: {err}")
    print(f"  Log: {lpath}")
    if lpath:
        try:
            with open(lpath) as f:
                lines = f.readlines()
                print(f"  Last 5 lines of log:")
                for l in lines[-5:]:
                    print(f"    {l.strip()}")
        except Exception as e:
            print(f"  Could not read log: {e}")
    print("-" * 60)

# Completed jobs breakdown
completed_l0 = cursor.execute("SELECT count(*) FROM jobs WHERE status='completed' AND id LIKE '%_l0'").fetchone()[0]
completed_l1 = cursor.execute("SELECT count(*) FROM jobs WHERE status='completed' AND id LIKE '%_l1'").fetchone()[0]
print(f"\n=== COMPLETED BREAKDOWN ===")
print(f"  Lead-I  (_l0): {completed_l0}/60")
print(f"  Lead-II (_l1): {completed_l1}/60")

# Pending jobs breakdown
pending_jobs = cursor.execute("SELECT id FROM jobs WHERE status='pending'").fetchall()
print(f"\n=== PENDING JOBS ({len(pending_jobs)}) ===")
for p in pending_jobs:
    print(f"  {p[0]}")

conn.close()
