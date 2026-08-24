#!/usr/bin/env python3
"""Create transactionally consistent SQLite inputs for one book render."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, sqlite3
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATABASES=[
 'results/clinical_biomarkers_multids/clinical_metrics.db','results/checkpoint_store/catalog.sqlite',
 'results/ecgaim_ludb_semiseg_blinded/compact.sqlite','results/ecgaim_rdb_semiseg_blinded/compact.sqlite',
 'results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite','refine-logs/wavelet_ssl_1110000/full/queue.sqlite',
 'results/onelead_checkpoint_store/catalog.sqlite','results/checkpoint_embeddings/compact.sqlite',
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for rel in DATABASES:
  src=ROOT/rel;dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if not src.is_file():rows.append({'path':rel,'status':'missing_optional'});continue
  with sqlite3.connect(f'file:{src}?mode=ro',uri=True,timeout=60) as source, sqlite3.connect(dest) as target:
   source.backup(target);check=target.execute('pragma quick_check').fetchone()[0]
  if check!='ok':raise RuntimeError(f'Snapshot integrity failed: {rel}: {check}')
  rows.append({'path':rel,'status':'snapshotted','bytes':dest.stat().st_size,'sha256':sha(dest)})
 manifest={'schema_version':1,'snapshot_id':dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
           'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'databases':rows}
 tmp=out/'SNAPSHOT_MANIFEST.json.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(out/'SNAPSHOT_MANIFEST.json')
 print(json.dumps(manifest))
if __name__=='__main__':main()
