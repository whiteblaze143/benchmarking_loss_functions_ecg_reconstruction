"""Compare released RR-index endpoints with ECG-index endpoints; no fitted shifts."""
import csv,io,json,zipfile
from pathlib import Path
import h5py,numpy as np
from rr_protocol import P,O
splits={r['patient_id']:r['split'] for r in csv.DictReader((O/'patient_splits.csv').open())}
meta=list(csv.DictReader((P/'iridia-af-metadata-v1.0.1.csv').open()));rows=[];duration=[]
lengths={}
for line in (P/'waveform_audit.jsonl').read_text().splitlines():
 r=json.loads(line)
 if r['key']=='ecg':lengths[r['member']]=r['shape'][0]
with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
 for m in meta:
  rid=m['record_id'];base=f'iridia-af-records-v1.0.1/{rid}/{rid}'
  ecg=list(csv.DictReader(io.StringIO(z.read(base+'_ecg_labels.csv').decode())))
  labels=list(csv.DictReader(io.StringIO(z.read(base+'_rr_labels.csv').decode())))
  rr=[]
  for day in range(int(m['record_files'])):
   with h5py.File(io.BytesIO(z.read(f'{base}_rr_{day:02d}.h5'))) as h:v=h['rr'][:]
   rr.append(np.cumsum(v,dtype=np.float64)/1000)
   duration.append({'patient_id':m['patient_id'],'record_id':rid,'day':day,'rr_sum_seconds':float(rr[-1][-1]),'ecg_seconds':lengths[f'{base}_ecg_{day:02d}.h5']/200,'rr_sum_minus_ecg_seconds':float(rr[-1][-1])-lengths[f'{base}_ecg_{day:02d}.h5']/200})
  assert len(ecg)==len(labels)
  for j,(a,b) in enumerate(zip(ecg,labels)):
   for endpoint in ['start','end']:
    day=int(b[f'{endpoint}_file_index']);index=int(b[f'{endpoint}_rr_index'])
    assert day==int(a[f'{endpoint}_file_index'])
    actual=float(a[f'{endpoint}_qrs_index'])/200
    inferred=float(rr[day][index]) if index<len(rr[day]) else None
    rows.append({'patient_id':m['patient_id'],'split':splits[m['patient_id']],'record_id':rid,'episode_row':j,'endpoint':endpoint,'day':day,'rr_index':index,'ecg_endpoint_seconds':actual,'rr_cumulative_endpoint_seconds':inferred,'rr_minus_ecg_seconds':inferred-actual if inferred is not None else None})
for name,data in [('rr_ecg_clock_endpoint_audit.csv',rows),('rr_ecg_duration_audit.csv',duration)]:
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
s={'endpoint_count':len(rows),'unavailable_rr_endpoint_indices':sum(r['rr_minus_ecg_seconds'] is None for r in rows),'no_offsets_fitted':True,'partitions':{}}
for split in ['train','validation','test']:
 v=np.array([r['rr_minus_ecg_seconds'] for r in rows if r['split']==split and r['rr_minus_ecg_seconds'] is not None])
 s['partitions'][split]={'endpoints':len(v),'median_raw_rr_minus_ecg_seconds':float(np.median(v)),'min':float(v.min()),'max':float(v.max()),'absolute_error_95_percentile_seconds':float(np.quantile(abs(v),.95))}
s['daily_rr_sum_minus_ecg_seconds_range']=[min(r['rr_sum_minus_ecg_seconds'] for r in duration),max(r['rr_sum_minus_ecg_seconds'] for r in duration)]
(P/'rr_ecg_clock_alignment_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
