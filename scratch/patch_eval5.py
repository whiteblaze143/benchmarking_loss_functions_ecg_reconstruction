import re

with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "r") as f:
    code = f.read()

# Replace the SQLite query logic for finding models
new_logic = """
    if checkpoint_db.is_file():
        conn_chk = connect_checkpoint_db(checkpoint_db)
        
        # Determine if it's the new queue format or the old catalog
        tables = [t[0] for t in conn_chk.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "jobs" in tables:
            # It's a queue.sqlite
            models = conn_chk.execute(
                "SELECT id as model_id, '{}' as factorial_mask, 'unknown' as sha256, '' as local_path, '' as observed_leads_json FROM jobs WHERE status='completed' ORDER BY id"
            ).fetchall()
            is_queue = True
        else:
            # It's catalog.sqlite
            models = conn_chk.execute(
                "SELECT model_id, factorial_mask, sha256, local_path, observed_leads_json FROM checkpoints WHERE status='remote_verified' ORDER BY model_id"
            ).fetchall()
            is_queue = False
            
        conn_chk.close()
        
        for row in models:
            if STOP_REQUESTED:
                break
            model_id = row["model_id"]
            mask = row["factorial_mask"]
            sha256 = row["sha256"]
            
            if is_queue:
                obs_lead = int(model_id.split('_l')[-1]) if '_l' in model_id else 0
                ckpt_path = checkpoint_db.parent / "runs" / model_id / "resume.pt"
            else:
                obs_json = row["observed_leads_json"]
                obs_lead = json.loads(obs_json)[0] if obs_json else 0
                local_path = row["local_path"]
                ckpt_path = Path(local_path) if local_path and Path(local_path).is_file() else cache_dir / f"{model_id}.pt"
"""

code = re.sub(r'    if checkpoint_db\.is_file\(\):\n\s+conn_chk = .*?\n\s+for row in models:\n\s+if STOP_REQUESTED:\n\s+break\n\s+model_id, mask.*?obs_lead =.*?\n', new_logic, code, flags=re.DOTALL)

with open("scripts/evaluate_1lead_rdb_semiseg_blinded.py", "w") as f:
    f.write(code)
