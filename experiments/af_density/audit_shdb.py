"""Audit existing SHDB-AF metadata and reference rhythm intervals only."""
import csv,json
from collections import Counter
from pathlib import Path
import wfdb
from metrics import charitos_density

p=Path('/data/mithunmanivannan/datasets/shdb_af')
out=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
metadata=list(csv.DictReader((p/'AdditionalData.csv').open()))
rows=[];intervals=[];tokens=Counter()
for m in metadata:
    uid=m['Data_ID'];h=wfdb.rdheader(str(p/uid));annotated=m['Annotated']=='True'
    assert (p/f'{uid}.atr').exists()==annotated
    af=[];unknown=0;rhythm_seconds=Counter()
    if annotated:
        a=wfdb.rdann(str(p/uid),'atr')
        transitions=[(int(s),note.strip()) for s,note in zip(a.sample,a.aux_note) if note.strip()]
        assert transitions and all(0<=s<h.sig_len for s,_ in transitions)
        assert all(x[0]<=y[0] for x,y in zip(transitions,transitions[1:]))
        unknown=transitions[0][0]/h.fs
        for j,(start,rhythm) in enumerate(transitions):
            end=transitions[j+1][0] if j+1<len(transitions) else h.sig_len
            assert rhythm in {'(AFIB','(AFL','(AT','(PAT','(NOD','(N','(AB'},rhythm
            tokens[rhythm]+=1
            if start==end:continue
            rhythm_seconds[rhythm]+=(end-start)/h.fs
            intervals.append({'record_id':uid,'subject_id':m['Subject_ID'],'rhythm':rhythm,'start_sample':start,'end_sample_exclusive':end,'sampling_rate':h.fs})
            if rhythm=='(AFIB':
                if af and af[-1][1]==start:af[-1]=(af[-1][0],end)
                else:af.append((start,end))
    duration=h.sig_len/h.fs
    rows.append({'record_id':uid,'subject_id':m['Subject_ID'],'annotated':annotated,'sampling_rate':h.fs,'channel_names':';'.join(h.sig_name),'duration_seconds':duration,'initial_unknown_seconds':unknown if annotated else duration,'af_intervals':len(af) if annotated else None,'af_seconds':sum(b-a for a,b in af)/h.fs if annotated else None,'afl_seconds':rhythm_seconds['(AFL'] if annotated else None,'at_seconds':rhythm_seconds['(AT'] if annotated else None,'continuous_charitos_density_raw':charitos_density([(s/h.fs,e/h.fs) for s,e in af],duration) if annotated and unknown==0 else None})
for name,data in [('shdb_record_audit.csv',rows),('shdb_reference_intervals.csv',intervals)]:
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
s={'recordings':len(rows),'subject_ids':len({r['subject_id'] for r in rows}),'annotated_recordings':sum(r['annotated'] for r in rows),'annotated_subject_ids':len({r['subject_id'] for r in rows if r['annotated']}),'af_intervals':sum(r['af_intervals'] or 0 for r in rows),'multi_af_interval_recordings':sum((r['af_intervals'] or 0)>1 for r in rows),'multi_af_interval_subject_ids':len({r['subject_id'] for r in rows if (r['af_intervals'] or 0)>1}),'annotation_tokens':dict(tokens),'sampling_rates':sorted({r['sampling_rate'] for r in rows}),'channel_names':sorted({r['channel_names'] for r in rows}),'limitations':['Existing local release audited; waveform checksums and signal quality have not been verified here.','Categorical ECG1/ECG2 names do not establish canonical q coefficients.','No-AF or continuous-AF density is undefined; raw metrics are descriptive, not held-out model evaluation.']}
(out/'shdb_audit_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
