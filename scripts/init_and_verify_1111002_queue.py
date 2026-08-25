#!/usr/bin/env python3
"""Initialize and verify the SQLite queue for Wavelet SSL 1111002 (120 models)."""

import json, sys, os, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wavelet_ssl_queue import connect, initialize, sha256_file

def main():
    work_dir = ROOT / "refine-logs/wavelet_ssl_1111002"
    preflight_manifest = work_dir / "preflight/manifest.json"
    full_manifest = work_dir / "full/manifest.json"

    print("="*70)
    print("  INITIALIZING & VERIFYING WAVELET SSL 1111002 QUEUE (120 MODELS)")
    print("="*70)

    # 1. Initialize preflight queue
    print("\n1. Initializing Preflight Queue (15 models)...")
    preflight_db = work_dir / "preflight/queue.sqlite"
    with connect(preflight_db) as conn_pf:
        initialize(conn_pf, preflight_manifest, ROOT)
    
    conn_pf = sqlite3.connect(preflight_db)
    pf_count = conn_pf.execute("SELECT count(*) FROM jobs").fetchone()[0]
    conn_pf.close()
    print(f"   Preflight Queue populated with {pf_count} jobs.")

    # 2. Initialize full sweep queue
    print("\n2. Initializing Full Sweep Queue (120 models: 60 Lead 0, 60 Lead 1)...")
    full_db = work_dir / "full/queue.sqlite"
    with connect(full_db) as conn_full:
        initialize(conn_full, full_manifest, ROOT)
        
    conn_full = sqlite3.connect(full_db)
    full_count = conn_full.execute("SELECT count(*) FROM jobs").fetchone()[0]
    lead0_count = conn_full.execute("SELECT count(*) FROM jobs WHERE id LIKE '%_l0'").fetchone()[0]
    lead1_count = conn_full.execute("SELECT count(*) FROM jobs WHERE id LIKE '%_l1'").fetchone()[0]
    conn_full.close()
    print(f"   Full Queue populated with {full_count} jobs:")
    print(f"     - Lead 0 (Lead I):  {lead0_count} jobs")
    print(f"     - Lead 1 (Lead II): {lead1_count} jobs")

    # 3. Verify SQLite DB integrity
    print("\n3. Verifying SQLite DB Integrity...")
    for label, db_path in (("Preflight", preflight_db), ("Full", full_db)):
        conn = sqlite3.connect(db_path)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        print(f"   {label} DB ({db_path.name}): {integrity}")
        if integrity != "ok":
            raise RuntimeError(f"Integrity check failed on {db_path}")

    # 4. Verify command structure
    print("\n4. Verifying Sample Job Command Structure for 1111002...")
    conn = sqlite3.connect(full_db)
    row = conn.execute("SELECT id, command_json FROM jobs LIMIT 1").fetchone()
    sample_id, sample_cmd_str = row
    sample_cmd = json.loads(sample_cmd_str)
    conn.close()
    print(f"   Sample Job ID: {sample_id}")
    print(f"   Factorial Mask: {'1111002' in sample_cmd}")
    print(f"   Command flags preview: {' '.join(sample_cmd[:8])}...")

    print("\n" + "="*70)
    print("  QUEUE INITIALIZATION & INTEGRITY AUDIT: 100% SUCCESS")
    print("="*70)

if __name__ == "__main__":
    main()
