"""Stage unchanged ECG HDF5 members for random-access observed-channel analysis."""
import hashlib,json,os,re,time,zipfile
from pathlib import Path
from rr_protocol import P
D=P/'ecg_hdf5';D.mkdir(exist_ok=True)
manifest=P/'ecg_staging_manifest.jsonl'
assert not manifest.exists(),'Staging already started; inspect existing worker before any restart'
start=time.time()
with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z,manifest.open('w') as f:
    names=sorted(n for n in z.namelist() if '_ecg_' in n and n.endswith('.h5'))
    assert len(names)==289
    for i,name in enumerate(names):
        assert re.fullmatch(r'iridia-af-records-v1\.0\.1/(record_\d{3})/\1_ecg_\d{2}\.h5',name)
        relative=Path(name).relative_to('iridia-af-records-v1.0.1');dest=D/relative;dest.parent.mkdir(exist_ok=True)
        assert not dest.exists(),dest
        temp=dest.with_suffix('.h5.partial');digest=hashlib.sha256();count=0
        with z.open(name) as source,temp.open('wb') as out:
            while data:=source.read(8*1024*1024):
                out.write(data);digest.update(data);count+=len(data)
        assert count==z.getinfo(name).file_size
        os.replace(temp,dest)
        row={'member':name,'path':str(dest),'bytes':count,'sha256':digest.hexdigest(),'archive_crc32':format(z.getinfo(name).CRC,'08x')}
        f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps({'completed':i+1,'total':289,'bytes_written':count,'elapsed_seconds':time.time()-start}),flush=True)
(P/'ecg_staging_status.json').write_text(json.dumps({'complete':True,'files':len(names),'elapsed_seconds':time.time()-start,'source_values_changed':False})+'\n')
