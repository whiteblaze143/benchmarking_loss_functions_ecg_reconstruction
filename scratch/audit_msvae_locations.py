import sqlite3
import json
from pathlib import Path

with open('refine-logs/queue_3arch/queue_state.json') as f:
    q = json.load(f)

completed_msvae = [j for j in q['jobs'] if j['id'].startswith('msvae_') and j['status'] == 'completed']
print(f"Total completed MS-VAE jobs in queue_3arch: {len(completed_msvae)}")

conn = sqlite3.connect('results/checkpoint_store/catalog.sqlite')
cursor = conn.cursor()

in_catalog = []
not_in_catalog = []
on_disk = []
not_on_disk = []

for j in completed_msvae:
    mid = j['id'].replace('msvae_f_', 'factorial_msvae_')
    cursor.execute('SELECT model_id, status, asset_id, uploaded_at, remote_verified_at FROM checkpoints WHERE model_id=?', (mid,))
    row = cursor.fetchone()
    if row:
        in_catalog.append((mid, row[1], row[2], row[3]))
    else:
        not_in_catalog.append(mid)
        
    p_disk = Path('checkpoints') / f'{mid}.pt'
    if p_disk.exists():
        on_disk.append(mid)
    else:
        not_on_disk.append(mid)

print(f"\nCatalog Summary:")
print(f"  In catalog.sqlite: {len(in_catalog)} / {len(completed_msvae)}")
print(f"  Not in catalog.sqlite: {len(not_in_catalog)} / {len(completed_msvae)}")
if not_in_catalog:
    print(f"  Sample not in catalog: {not_in_catalog[:5]}")

print(f"\nDisk Summary (checkpoints/*.pt):")
print(f"  On disk: {len(on_disk)} / {len(completed_msvae)}")
print(f"  Not on disk (remote/archived): {len(not_on_disk)} / {len(completed_msvae)}")
if not_on_disk:
    print(f"  Sample not on disk: {not_on_disk[:5]}")

# Also check checkpoint_archiver status
conn.close()
