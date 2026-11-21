import csv,hashlib,io,json,time,zipfile
from pathlib import Path
import h5py,numpy as np
from rr_features import features,layout,NAMES
from rr_protocol import P,O
assert not (O/'windows.npz').exists(),'Refusing to replace prepared data'
protocol=json.loads((O/'protocol.json').read_text());assert hashlib.sha256((O/'patient_splits.csv').read_bytes()).hexdigest()==protocol['split_manifest_sha256']
patients=list(csv.DictReader((O/'patient_splits.csv').open()));pi={r['patient_id']:i for i,r in enumerate(patients)}
records=list(csv.DictReader((P/'iridia-af-metadata-v1.0.1.csv').open()));parts=[];files=[];started=time.time()
with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
    for ri,m in enumerate(records):
        rid=m['record_id'];base=f'iridia-af-records-v1.0.1/{rid}/{rid}'
        labels=list(csv.DictReader(io.StringIO(z.read(base+'_rr_labels.csv').decode())))
        for day in range(int(m['record_files'])):
            with h5py.File(io.BytesIO(z.read(f'{base}_rr_{day:02d}.h5'))) as h:rr=h['rr'][:]
            truth=np.zeros(len(rr),dtype=np.int8)
            for e in labels:
                sf=int(e['start_file_index']);ef=int(e['end_file_index'])
                if sf<=day<=ef:
                    lo=int(e['start_rr_index']) if day==sf else 0
                    hi=int(e['end_rr_index']) if day==ef else len(rr)
                    assert 0<=lo<=hi<=len(rr);truth[lo:hi]=1
            excluded=31 if day==0 else 61
            starts,assigned,ends=layout(len(rr),excluded)
            f,valid=features(rr[starts[:,None]+np.arange(60)])
            prefix=np.r_[0,np.cumsum(truth,dtype=np.int64)]
            n=len(starts)
            parts.append({'X':f.astype(np.float32),'y':((prefix[starts+60]-prefix[starts])>=30).astype(np.int8),'valid':valid,'assigned_positive':(prefix[ends]-prefix[assigned]).astype(np.int16),'assigned_count':(ends-assigned).astype(np.int16),'patient_index':np.full(n,pi[m['patient_id']],dtype=np.int16),'record_index':np.full(n,ri,dtype=np.int16),'day':np.full(n,day,dtype=np.int8),'input_start':starts.astype(np.int32),'assigned_start':assigned.astype(np.int32),'assigned_end':ends.astype(np.int32)})
            assert (ends-assigned).sum()==len(rr)-excluded
            files.append({'patient_id':m['patient_id'],'record_id':rid,'record_index':ri,'day':day,'source_rr_count':len(rr),'excluded_calibration_proxy_rr':excluded,'declared_windows':n,'usable_windows':int(valid.sum()),'assigned_rr':len(rr)-excluded,'assigned_af_rr':int(truth[excluded:].sum())})
        print(rid,'completed',len(parts),'daily files',flush=True)
a={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
np.savez_compressed(O/'windows.npz',**a)
with (O/'file_manifest.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(files[0]));w.writeheader();w.writerows(files)
summary={'feature_names':NAMES,'daily_files':len(files),'recordings':len(records),'patient_ids':len(patients),'windows':len(a['X']),'usable_windows':int(a['valid'].sum()),'assigned_rr':int(a['assigned_count'].sum()),'calibration_proxy_excluded_rr':sum(r['excluded_calibration_proxy_rr'] for r in files),'elapsed_seconds':time.time()-started,'partitions':{s:{'patients':sum(r['split']==s for r in patients),'windows':int(sum(len(a['X'][a['patient_index']==i]) for i,r in enumerate(patients) if r['split']==s))} for s in ['train','validation','test']}}
summary['data_sha256']=hashlib.sha256((O/'windows.npz').read_bytes()).hexdigest();(O/'preparation_receipt.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
