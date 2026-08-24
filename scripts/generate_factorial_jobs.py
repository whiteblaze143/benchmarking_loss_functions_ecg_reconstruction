import json
import itertools

project = "ecg_reconstruction_factorial"
cwd = "/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction"

base_manifest = {
    "project": project,
    "cwd": cwd,
    "ssh": "localhost",
    "conda": "none",
    "gpus": [0, 1, 2, 3],
    "max_parallel": 4,
    "gpu_free_threshold_mib": 6000,
    "oom_retry": {
        "delay": 60,
        "max_attempts": 3
    },
    "phases": [{
        "name": "default",
        "depends_on": [],
        "jobs": []
    }]
}

seeds = [42, 200, 201]
corrs = [0, 1]
derivs = [0, 1]
vcgs = [0, 1]
eds = [0, 1]
leads = [0, 1]
mmds = [0, 1, 2, 3, 4]

for seed, corr, deriv, vcg, ed, lead, mmd in itertools.product(seeds, corrs, derivs, vcgs, eds, leads, mmds):
    mask = f"1{corr}{deriv}{vcg}{ed}{lead}{mmd}"
    job_id = f"f_{mask}_s{seed}"
    cmd = f"~/.venv/bin/python3 scripts/train_factorial.py --factorial_mask {mask} --seed {seed} --run_name {job_id} --checkpoint_path checkpoints/factorial_{mask}_s{seed}.pt"
    
    base_manifest["phases"][0]["jobs"].append({
        "id": job_id,
        "cmd": cmd
    })

with open("refine-logs/factorial_manifest.json", "w") as f:
    json.dump(base_manifest, f, indent=2)

print(f"Generated {len(base_manifest['phases'][0]['jobs'])} jobs in default phase.")
