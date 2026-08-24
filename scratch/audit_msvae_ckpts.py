import json
import sqlite3
from pathlib import Path

# 1. Models in registry
with open('results/clinical_biomarkers_model_registry.json') as f:
    reg = json.load(f)
reg_msvae = {m['id'] for m in reg['models'] if m.get('kind') == 'msvae'}

# 2. Checkpoints on disk
disk_msvae = {p.stem for p in Path('checkpoints').glob('*msvae*.pt')}

# 3. Checkpoints in catalog
conn = sqlite3.connect('results/checkpoint_store/catalog.sqlite')
cursor = conn.cursor()
cursor.execute("SELECT model_id FROM checkpoints WHERE model_id LIKE '%msvae%'")
cat_msvae = {r[0] for r in cursor.fetchall()}
conn.close()

print(f"MS-VAE models in clinical_biomarkers_model_registry.json: {len(reg_msvae)}")
print(f"MS-VAE checkpoints on disk (checkpoints/*.pt): {len(disk_msvae)}")
print(f"MS-VAE checkpoints in catalog.sqlite: {len(cat_msvae)}")

# Check disk vs registry
on_disk_not_in_reg = disk_msvae - reg_msvae
print(f"\nOn disk but not in registry ({len(on_disk_not_in_reg)}):", sorted(list(on_disk_not_in_reg))[:10])

# Check catalog vs disk
in_cat_not_on_disk = cat_msvae - disk_msvae
print(f"\nIn catalog.sqlite but not on disk ({len(in_cat_not_on_disk)}):", sorted(list(in_cat_not_on_disk))[:10])
