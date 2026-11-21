"""Measure periodic device-prefix candidates; preserve samples for visual audit."""
import csv,io,json,time,zipfile
from pathlib import Path
import h5py,numpy as np
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
D=P/'ecg_prefix_120s';D.mkdir(exist_ok=True)
assert not (P/'ecg_prefix_audit.csv').exists(),'Refusing to overwrite prefix audit'
rows=[];start=time.time()
with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
    names=sorted(n for n in z.namelist() if '_ecg_' in n and n.endswith('.h5'))
    for k,name in enumerate(names):
        with h5py.File(io.BytesIO(z.read(name))) as h:a=h['ecg'][:24000]
        np.save(D/(Path(name).stem+'.npy'),a,allow_pickle=False)
        seconds=len(a)//200-1
        equal=(a[:seconds*200]==a[200:(seconds+1)*200]).reshape(seconds,200,2).mean(axis=(1,2))
        run=next((i for i,v in enumerate(equal) if v<0.99),len(equal))
        rows.append({'record_id':Path(name).parent.name,'daily_file_index':int(Path(name).stem.rsplit('_',1)[1]),'initial_periodic_candidate_seconds':run+1 if run else 0,'candidate_capped_at_prefix_limit':run==len(equal),'first_second_repeat_fraction':float(equal[0]),'repeat_fraction_after_30s':float(equal[30]),'repeat_fraction_after_60s':float(equal[60])})
        print(json.dumps({'completed':k+1,'total':len(names),'member':name,'elapsed_seconds':time.time()-start}),flush=True)
with (P/'ecg_prefix_audit.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'ecg_prefix_audit_status.json').write_text(json.dumps({'complete':True,'files':len(rows),'seconds':time.time()-start,'criterion':'Consecutive seconds with >=99% exact sample equality to a one-second shift; candidate calibration only, not independently adjudicated.'})+'\n')
