"""Apply measured device-prefix exclusions without joining daily segments."""
import csv,json
from collections import defaultdict
from pathlib import Path
from metrics import charitos_density
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
assert json.loads((P/'ecg_prefix_audit_status.json').read_text())['complete']
lengths={}
for line in (P/'waveform_audit.jsonl').read_text().splitlines():
    x=json.loads(line)
    if x['key']=='ecg':lengths[(Path(x['member']).parent.name,int(Path(x['member']).stem.rsplit('_',1)[1]))]=x['shape'][0]
meta={x['record_id']:x for x in csv.DictReader((P/'timeline_record_audit.csv').open())}
events=defaultdict(list)
for e in csv.DictReader((P/'timeline_intervals.csv').open()):events[e['record_id']].append((int(e['start_sample']),int(e['end_sample_exclusive'])))
rows=[];offsets=defaultdict(int);masks=[]
for x in csv.DictReader((P/'ecg_prefix_audit.csv').open()):
    rid=x['record_id'];day=int(x['daily_file_index']);seconds=int(x['initial_periodic_candidate_seconds'])
    assert x['candidate_capped_at_prefix_limit']=='False'
    assert seconds==(30 if day==0 else 60),(rid,day,seconds)
    off=offsets[rid];end=off+lengths[(rid,day)];start=off+seconds*200
    clipped=[(max(a,start),min(b,end)) for a,b in events[rid] if a<end and b>start]
    cal_af=sum(max(0,min(b,start)-max(a,off)) for a,b in events[rid])/200
    masks.append({'patient_id':meta[rid]['patient_id'],'record_id':rid,'daily_file_index':day,'excluded_start_sample':off,'excluded_end_sample_exclusive':start,'excluded_seconds':seconds,'source_label_af_seconds_in_exclusion':cal_af,'reason':'device periodic prefix, empirical 1-second exact-repeat transition; not physiological ECG'})
    duration=(end-start)/200
    rows.append({'patient_id':meta[rid]['patient_id'],'record_id':rid,'daily_file_index':day,'start_sample':start,'end_sample_exclusive':end,'observation_seconds':duration,'af_intervals':len(clipped),'af_seconds':sum(b-a for a,b in clipped)/200,'density_contiguous_segment':charitos_density([((a-start)/200,(b-start)/200) for a,b in clipped],duration),'contains_full_24h':duration>=86400,'left_censored_af':bool(clipped and clipped[0][0]==start),'right_censored_af':bool(clipped and clipped[-1][1]==end),'clinical_quality_assessed':False})
    offsets[rid]=end
assert all(offsets[rid]==int(m['waveform_samples']) for rid,m in meta.items())
for name,rs in [('calibration_exclusion_intervals.csv',masks),('contiguous_segment_feasibility.csv',rows)]:
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
s={'daily_segments':len(rows),'excluded_device_prefix_seconds':sum(r['excluded_seconds'] for r in masks),'source_label_af_seconds_in_exclusions':sum(r['source_label_af_seconds_in_exclusion'] for r in masks),'remaining_observed_hours_before_quality_screen':sum(r['observation_seconds'] for r in rows)/3600,'multi_interval_contiguous_segments':sum(r['af_intervals']>1 for r in rows),'patient_ids_with_multi_interval_contiguous_segment':len({r['patient_id'] for r in rows if r['af_intervals']>1}),'uninterrupted_complete_24h_segments':sum(r['contains_full_24h'] for r in rows),'gap_policy':'Keep exclusions on original clock. Do not compute full-record gap-free density by concatenating segments. Fixed 24h windows require an explicit missing-data method.','quality_qualified':False}
(P/'contiguous_segment_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
