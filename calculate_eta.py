import json
from datetime import datetime

with open("refine-logs/queue/queue_state.json", "r") as f:
    state = json.load(f)

completed_jobs = [j for j in state["jobs"] if j["status"] == "completed" and j.get("started") and j.get("completed")]

durations = []
for j in completed_jobs:
    try:
        t0 = datetime.fromisoformat(j["started"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(j["completed"].replace("Z", "+00:00"))
        durations.append((t1 - t0).total_seconds())
    except Exception:
        pass

if durations:
    avg_duration_sec = sum(durations) / len(durations)
    print(f"Completed jobs: {len(durations)}")
    print(f"Average duration per job: {avg_duration_sec / 60:.2f} minutes")
    
    remaining = 353
    # 4 jobs run concurrently
    total_hours = (remaining / 4) * (avg_duration_sec / 3600)
    print(f"Estimated time for remaining {remaining} jobs (with 4x concurrency): {total_hours:.1f} hours ({total_hours / 24:.2f} days)")
