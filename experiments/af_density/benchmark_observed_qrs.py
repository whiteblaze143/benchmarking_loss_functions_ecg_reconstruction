"""Training-patient-only observed-channel detector preflight; no classifier fit."""
import csv,json,time
from pathlib import Path
import h5py,numpy as np
from wfdb.processing import xqrs_detect
from rr_protocol import P,O
D=O.parent/'observed_qrs_preflight';D.mkdir(exist_ok=True)
train={r['patient_id'] for r in csv.DictReader((O/'patient_splits.csv').open()) if r['split']=='train'}
meta=[r for r in csv.DictReader((P/'iridia-af-metadata-v1.0.1.csv').open()) if r['patient_id'] in train][:3]
rows=[]
for m in meta:
    path=P/'ecg_hdf5'/m['record_id']/(m['record_id']+'_ecg_00.h5')
    with h5py.File(path) as h:signal=h['ecg'][6000:66000]
    for channel in [0,1]:
        start=time.time();peaks=np.asarray(xqrs_detect(signal[:,channel],fs=200,verbose=False));seconds=time.time()-start
        rr=np.diff(peaks)/200;valid=(rr>=.3)&(rr<=2.)
        rows.append({'patient_id':m['patient_id'],'record_id':m['record_id'],'partition':'train','observed_channel_index':channel,'input_seconds':len(signal)/200,'detected_peaks':len(peaks),'accepted_rr_300_2000ms':int(valid.sum()),'fraction_input_represented_by_accepted_rr':float(rr[valid].sum()/(len(signal)/200)),'detector_seconds':seconds,'real_time_speedup':(len(signal)/200)/seconds,'peak_sample_indices':peaks.tolist()})
        print(m['record_id'],channel,'peaks',len(peaks),'seconds',round(seconds,3),flush=True)
summary={'configurations':len(rows),'all_train_only':True,'signal_seconds':sum(r['input_seconds'] for r in rows),'detector_seconds':sum(r['detector_seconds'] for r in rows),'rows':rows,'limits':'Three training records, five minutes each, two channels independently. Shape/runtime preflight only; no accuracy or whole-dataset extraction success claim.'}
summary['aggregate_real_time_speedup']=summary['signal_seconds']/summary['detector_seconds']
(D/'detector_preflight.json').write_text(json.dumps(summary,indent=2)+'\n');print('AGGREGATE_SPEEDUP',summary['aggregate_real_time_speedup'])
