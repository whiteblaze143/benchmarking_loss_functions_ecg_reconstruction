"""Download pinned public LTAFDB headers and references, verifying publisher hashes."""
import concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
P=Path('/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference');P.mkdir(exist_ok=True)
BASE='https://physionet.org/files/ltafdb/1.0.0/'
def fetch(name):
    b=urllib.request.urlopen(BASE+name,timeout=60).read()
    return b
manifest=fetch('SHA256SUMS.txt');(P/'SHA256SUMS.txt').write_bytes(manifest)
expected={}
for line in manifest.decode().splitlines():
    digest,name=line.split(maxsplit=1);expected[name.lstrip('*')]=digest
records=fetch('RECORDS');assert hashlib.sha256(records).hexdigest()==expected['RECORDS'];(P/'RECORDS').write_bytes(records)
names=[f'{rid}.{ext}' for rid in records.decode().split() for ext in ['hea','atr']]
assert len(names)==168 and all(n in expected for n in names)
def download(name):
    path=P/name
    if path.exists():b=path.read_bytes()
    else:b=fetch(name)
    digest=hashlib.sha256(b).hexdigest()
    assert digest==expected[name],name
    if not path.exists():path.write_bytes(b)
    print(name,len(b),'verified',flush=True)
    return {'file':name,'bytes':len(b),'sha256':digest,'verified':True}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(download,names))
(P/'reference_download_receipt.json').write_text(json.dumps({'version':'1.0.0','source':BASE,'complete':True,'scope':'RECORDS, headers and atr annotations only; waveforms not downloaded','files':checks},indent=2)+'\n')
print('COMPLETE_REFERENCE_FILES_VERIFIED',flush=True)
