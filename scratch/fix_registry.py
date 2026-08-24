import json

with open("results/pareto_models_registry.json", "r") as f:
    data = json.load(f)

for model in data["models"]:
    model["observed_leads"] = [0, 1, 7]

with open("results/pareto_models_registry.json", "w") as f:
    json.dump(data, f, indent=2)

print("Registry fixed.")
