"""Count actual AF patterns within fixed windows without compressing exclusions."""
import csv,json
from collections import defaultdict
from pathlib import Path
from metrics import charitos_density

P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
records=list(csv.DictReader((P/'timeline_record_audit.csv').open()))
events=defaultdict(list)
for r in csv.DictReader((P/'timeline_intervals.csv').open()):
    events[r['record_id']].append((int(r['start_sample']),int(r['end_sample_exclusive'])))
lengths=defaultdict(list)
for line in (P/'waveform_audit.jsonl').read_text().splitlines():
    r=json.loads(line)
    if r['key']=='ecg':lengths[Path(r['member']).parent.name].append((r['member'],r['shape'][0]))
rows=[]
for record in records:
    rid=record['record_id'];total=int(record['waveform_samples']);offset=0;calibration=[]
    for _,length in sorted(lengths[rid]):
        calibration.append((offset,offset+6000));offset+=length
    assert offset==total
    windows=[(f'prefix_{h}h',6000,6000+h*3600*200) for h in (6,12,24)]
    windows.append(('full_after_initial_calibration',6000,total))
    windows.extend((f'fixed_24h_block_{d}',6000+d*86400*200,6000+(d+1)*86400*200) for d in range((total-6000)//(86400*200)))
    for name,start,end in windows:
        available=end<=total and end>start
        clipped=[(max(a,start),min(b,end)) for a,b in events[rid] if available and a<end and b>start]
        duration=(end-start)/200
        af=sum(b-a for a,b in clipped)/200
        internal_calibration=sum(max(0,min(end,b)-max(start,a)) for a,b in calibration)/200 if available else None
        row={'patient_id':record['patient_id'],'record_id':rid,'window':name,'start_sample':start,'end_sample_exclusive':end,'available':available,'observation_seconds':duration if available else None,'af_intervals_in_window':len(clipped) if available else None,'af_seconds':af if available else None,'af_burden':af/duration if available else None,'left_censored_af':bool(clipped and clipped[0][0]==start),'right_censored_af':bool(clipped and clipped[-1][1]==end),'density_first_record_calibration_only':charitos_density([((a-start)/200,(b-start)/200) for a,b in clipped],duration) if available else None,'calibration_seconds_if_each_daily_file':internal_calibration,'uninterrupted_under_each_file_calibration_policy':available and internal_calibration==0,'density_under_each_file_calibration_policy':None}
        if row['uninterrupted_under_each_file_calibration_policy']:row['density_under_each_file_calibration_policy']=row['density_first_record_calibration_only']
        rows.append(row)
with (P/'observation_horizon_audit.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for name in ['prefix_6h','prefix_12h','prefix_24h','full_after_initial_calibration','fixed_24h_blocks']:
    rs=[r for r in rows if (r['window']==name or name=='fixed_24h_blocks' and r['window'].startswith('fixed_24h_block_')) and r['available']]
    valid=[r for r in rs if r['uninterrupted_under_each_file_calibration_policy']]
    multi=[r for r in rs if r['af_intervals_in_window']>1 and r['density_first_record_calibration_only'] is not None]
    validmulti=[r for r in valid if r['af_intervals_in_window']>1 and r['density_under_each_file_calibration_policy'] is not None]
    summary.append({'window':name,'available_windows':len(rs),'available_patient_ids':len({r['patient_id'] for r in rs}),'multi_interval_windows_first_record_calibration_only':len(multi),'multi_interval_patient_ids_first_record_calibration_only':len({r['patient_id'] for r in multi}),'uninterrupted_windows_if_each_file_calibration':len(valid),'multi_interval_patient_ids_if_each_file_calibration':len({r['patient_id'] for r in validmulti})})
(P/'observation_horizon_summary.json').write_text(json.dumps({'calibration_policy':'Unresolved. Two explicit scenarios; no excluded time is compressed or labeled non-AF. Density withheld when an internal calibration gap is excluded.','quality_qualified':False,'counts':summary},indent=2)+'\n')
print(json.dumps(summary,indent=2))
