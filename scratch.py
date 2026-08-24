import json

path = "experiment_queue/factorial_v2/queue_state.json"
with open(path) as f:
    state = json.load(f)

for phase in state["phases"]:
    if phase["name"] in ["smoke_factorial", "train_primary", "validation_evaluation", "confirmation_seeds", "full_robustness_evaluation", "factorial_statistics"]:
        phase["status"] = "pending"

with open(path, "w") as f:
    json.dump(state, f, indent=2)

print("Reset phase statuses.")
