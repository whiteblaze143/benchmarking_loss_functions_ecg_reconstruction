import json

with open("refine-logs/queue/queue_state.json", "r") as f:
    state = json.load(f)

fixed = 0
for job in state["jobs"]:
    if job["status"] != "completed":
        job["status"] = "pending"
        job["error"] = None
        job["attempts"] = 0
        fixed += 1

with open("refine-logs/queue/queue_state.json", "w") as f:
    json.dump(state, f, indent=2)

print(f"Fixed {fixed} jobs in queue_state.json")
