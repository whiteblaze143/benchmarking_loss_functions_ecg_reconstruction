import sqlite3, datetime

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
rows = conn.execute("""
    SELECT id, started_at, completed_at, status 
    FROM jobs 
    WHERE id LIKE '%_l1' AND status='completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL
    ORDER BY completed_at DESC
""").fetchall()

pending_count = conn.execute("SELECT count(*) FROM jobs WHERE id LIKE '%_l1' AND status='pending'").fetchone()[0]
running_jobs = conn.execute("SELECT id, started_at FROM jobs WHERE id LIKE '%_l1' AND status='running'").fetchall()
total_l1 = conn.execute("SELECT count(*) FROM jobs WHERE id LIKE '%_l1'").fetchone()[0]
completed_l1 = len(rows)

durations = []
for r in rows:
    try:
        t_start = datetime.datetime.fromisoformat(r[1].replace("Z", "+00:00"))
        t_end = datetime.datetime.fromisoformat(r[2].replace("Z", "+00:00"))
        durations.append((t_end - t_start).total_seconds())
    except Exception:
        pass

conn.close()

print(f"Lead 2 (_l1) Status: {completed_l1}/{total_l1} completed | {len(running_jobs)} running | {pending_count} pending")
if durations:
    avg_sec = sum(durations) / len(durations)
    last_5_avg = sum(durations[:5]) / min(5, len(durations))
    print(f"Overall average job duration: {avg_sec/60:.2f} min ({avg_sec:.1f}s)")
    print(f"Recent (last 5) job duration: {last_5_avg/60:.2f} min ({last_5_avg:.1f}s)")
    
    remaining_jobs = pending_count + len(running_jobs)
    est_sec = remaining_jobs * last_5_avg
    est_hours = est_sec / 3600
    est_minutes = (est_sec % 3600) / 60
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    eta_utc = now_utc + datetime.timedelta(seconds=est_sec)
    print(f"\nRemaining jobs: {remaining_jobs}")
    print(f"Estimated time remaining: {int(est_hours)}h {int(est_minutes):02d}m (~{est_sec/60:.1f} minutes)")
    print(f"Current UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Estimated Completion (ETA): {eta_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
