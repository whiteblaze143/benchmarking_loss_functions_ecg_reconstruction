#!/usr/bin/env python3
"""
Dynamic Model Registry Generator.
Pulls completed s42 checkpoints from both the local checkpoints/ directory
and the catalog.sqlite database (to support remote_verified models that were locally pruned).
"""

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
REGISTRY_FILE = ROOT / "results" / "clinical_biomarkers_model_registry.json"
CATALOG_DB = ROOT / "results" / "checkpoint_store" / "catalog.sqlite"
QUEUE_STATE = ROOT / "refine-logs" / "queue_3arch" / "queue_state.json"


def completed_3arch_jobs():
    if not QUEUE_STATE.exists():
        return set()
    payload = json.loads(QUEUE_STATE.read_text())
    return {
        job["id"] for job in payload.get("jobs", [])
        if job.get("status") == "completed"
    }


def queue_id_for_checkpoint(stem, kind):
    if kind == "msvae" and stem.startswith("factorial_msvae_"):
        return "msvae_f_" + stem.removeprefix("factorial_msvae_")
    if kind == "alitok" and stem.startswith("factorial_ecg_aim_"):
        return "ecg_aim_f_" + stem.removeprefix("factorial_ecg_aim_")
    return None

def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)

    models_dict = {}
    completed_jobs = completed_3arch_jobs()

    # 1. Add models physically present in checkpoints/ (must be s42)
    for path in sorted(CHECKPOINT_DIR.glob("*_s42.pt")):
        name = path.stem
        if "msvae" in name:
            kind = "msvae"
        elif "ecg_aim" in name or "alitok" in name:
            kind = "alitok"
        else:
            kind = "unet"

        # A best-so-far checkpoint exists after the first epoch.  Do not expose
        # it to evaluation until the queue has recorded an exit-zero training
        # completion, otherwise partial models become permanently "evaluated".
        queue_id = queue_id_for_checkpoint(name, kind)
        if queue_id is not None and queue_id not in completed_jobs:
            continue
            
        # Standardize ID to remove 'factorial_' or 'unet_' prefixes for unet
        model_id = name
        if kind == "unet" and model_id.startswith("factorial_unet_"):
            model_id = model_id.replace("factorial_unet_", "f_")
        elif kind == "unet" and model_id.startswith("factorial_"):
            model_id = model_id.replace("factorial_", "f_")

        models_dict[model_id] = {
            "id": model_id,
            "kind": kind,
            "checkpoint": f"checkpoints/{path.name}",
            "observed_leads": [0, 1, 7]
        }

    # 2. Add models from catalog.sqlite (if they are remote_verified or local_only)
    if CATALOG_DB.exists():
        conn = sqlite3.connect(str(CATALOG_DB))
        cursor = conn.cursor()
        
        # Check if checkpoints table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
        if cursor.fetchone():
            cursor.execute("""
                SELECT model_id FROM checkpoints
                WHERE model_id LIKE '%s42'
                  AND status IN ('local', 'remote_verified', 'cached')
            """)
            for row in cursor.fetchall():
                model_id = row[0]
                if model_id not in models_dict:
                    if "msvae" in model_id:
                        kind = "msvae"
                    elif "ecg_aim" in model_id or "alitok" in model_id:
                        kind = "alitok"
                    else:
                        kind = "unet"
                        
                    models_dict[model_id] = {
                        "id": model_id,
                        "kind": kind,
                        "checkpoint": f"checkpoints/{model_id}.pt",
                        "observed_leads": [0, 1, 7]
                    }
        conn.close()

    arch_priority = {"unet": 0, "msvae": 1, "alitok": 2, "ecg_aim": 2}
    models = sorted(list(models_dict.values()), key=lambda x: (arch_priority.get(x["kind"], 99), x["id"]))

    registry_payload = {
        "version": 1,
        "models": models
    }

    temporary = REGISTRY_FILE.with_suffix(REGISTRY_FILE.suffix + ".tmp")
    temporary.write_text(json.dumps(registry_payload, indent=2) + "\n")
    temporary.replace(REGISTRY_FILE)

    print(f"Updated Model Registry at {REGISTRY_FILE}: {len(models)} models registered (s42).")

if __name__ == "__main__":
    main()
