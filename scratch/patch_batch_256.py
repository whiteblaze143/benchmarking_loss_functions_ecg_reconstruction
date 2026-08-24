import json
import shutil
import re

manifest_path = "refine-logs/factorial_manifest.json"
shutil.copy(manifest_path, manifest_path + ".bak")

with open(manifest_path, "r") as f:
    manifest = json.load(f)

for phase in manifest.get("phases", []):
    for job in phase.get("jobs", []):
        cmd = job["cmd"]
        job["cmd"] = re.sub(r'--batch_size\s+\d+', '--batch_size 256', cmd)

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("Updated factorial_manifest.json to batch_size 256")

state_path = "refine-logs/queue/queue_state.json"
shutil.copy(state_path, state_path + ".bak")

with open(state_path, "r") as f:
    state = json.load(f)

for job in state.get("jobs", []):
    cmd = job.get("cmd", "")
    if cmd:
        job["cmd"] = re.sub(r'--batch_size\s+\d+', '--batch_size 256', cmd)
    if job["status"] != "completed":
        job["status"] = "pending"
        job["error"] = None
        job["attempts"] = 0

with open(state_path, "w") as f:
    json.dump(state, f, indent=2)

print("Updated queue_state.json to batch_size 256 and reset non-completed jobs to pending")
