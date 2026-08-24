#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Regexes
# cell id: family__c?m?d?__s42 -> family__e1c?m?d?__s42
# Matches something like: unet__c0m0d0__s42
PATTERN_ID = re.compile(r"([a-zA-Z0-9_]+)__c([01])m([01])d([01])__s(\d+)")

def replace_id(match):
    family, c, m, d, seed = match.groups()
    return f"{family}__e1c{c}m{m}d{d}__s{seed}"

def update_file_content(path: Path):
    if not path.is_file():
        return
    
    try:
        content = path.read_text()
    except Exception:
        return # Skip binaries
        
    new_content = PATTERN_ID.sub(replace_id, content)
    
    # Also fix factorial_mask in JSON
    if path.suffix == ".json" or path.name.endswith(".json"):
        try:
            data = json.loads(new_content)
            modified = False
            
            def recurse(d):
                nonlocal modified
                if isinstance(d, dict):
                    if "factorial_mask" in d and isinstance(d["factorial_mask"], str) and len(d["factorial_mask"]) == 3:
                        d["factorial_mask"] = "1" + d["factorial_mask"]
                        modified = True
                    for v in d.values():
                        recurse(v)
                elif isinstance(d, list):
                    for v in d:
                        recurse(v)
                        
            recurse(data)
            if modified:
                new_content = json.dumps(data, indent=2)
        except json.JSONDecodeError:
            pass # Maybe not a standard JSON file or just let the regex do its thing

    if new_content != content:
        path.write_text(new_content)
        print(f"Updated content in {path.relative_to(PROJECT_ROOT)}")

def rename_and_update(root_dir: Path):
    if not root_dir.exists():
        return
        
    # Bottom-up so renaming parent doesn't break children iteration
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            path = Path(root) / name
            update_file_content(path)
            
            new_name = PATTERN_ID.sub(replace_id, name)
            if new_name != name:
                new_path = Path(root) / new_name
                path.rename(new_path)
                print(f"Renamed file {path.relative_to(PROJECT_ROOT)} -> {new_name}")
                
        for name in dirs:
            new_name = PATTERN_ID.sub(replace_id, name)
            if new_name != name:
                path = Path(root) / name
                new_path = Path(root) / new_name
                path.rename(new_path)
                print(f"Renamed dir {path.relative_to(PROJECT_ROOT)} -> {new_name}")

if __name__ == "__main__":
    rename_and_update(PROJECT_ROOT / "checkpoints" / "factorial_v2")
    rename_and_update(PROJECT_ROOT / "results" / "factorial_v2")
    
    # Update state file specifically if it exists
    queue_state = PROJECT_ROOT / "experiment_queue" / "factorial_v2" / "queue_state.json"
    if queue_state.exists():
        update_file_content(queue_state)
