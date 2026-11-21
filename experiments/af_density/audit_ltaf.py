"""Audit pinned LTAFDB reference-only release, preserving initial unknown time."""
import csv,json
from collections import Counter
from pathlib import Path
import wfdb
from metrics import charitos_density
P=Path('/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference')
assert json.loads((P/'reference_download_receipt.json').read_text())['complete']
rows=[];intervals=[];rhythms=Counter();comments=Counter()
for rid in (P/'RECORDS').read_text().split():
    h=wfdb.rdheader(str(P/rid));a=wfdb.rdann(str(P/rid),'atr')
    trans=[]
    for sample,symbol,note in zip(a.sample,a.symbol,a.aux_note):
        note=note.strip()
        if note.startswith('('):
            assert 0<=sample<h.sig_len
            trans.append((int(sample),note));rhythms[note]+=1
        elif note:comments[note]+=1
    assert trans and all(x[0]<=y[0] for x,y in zip(trans,trans[1:]))
    af=[]
    for j,(start,rhythm) in enumerate(trans):
        end=trans[j+1][0] if j+1<len(trans) else h.sig_len
        if start==end:continue
        intervals.append({'record_id':rid,'rhythm':rhythm,'start_sample':start,'end_sample_exclusive':end,'sampling_rate':h.fs})
        if rhythm=='(AFIB':
            if af and af[-1][1]==start:af[-1]=(af[-1][0],end)
            else:af.append((start,end))
    print(rid,'AF intervals',len(af),flush=True)
    first=trans[0][0];duration=(h.sig_len-first)/h.fs
    rows.append({'record_id':rid,'patient_id':'','patient_linkage_verified':False,'sampling_rate':h.fs,'channel_names':';'.join(h.sig_name),'raw_duration_seconds':h.sig_len/h.fs,'initial_unknown_seconds':first/h.fs,'rhythm_annotated_suffix_seconds':duration,'af_intervals':len(af),'af_seconds':sum(b-a for a,b in af)/h.fs,'density_on_annotated_suffix':charitos_density([((s-first)/h.fs,(e-first)/h.fs) for s,e in af],duration),'q_coefficients_known':False,'waveform_downloaded':False})
for name,data in [('record_reference_audit.csv',rows),('reference_rhythm_intervals.csv',intervals)]:
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
s={'records':len(rows),'af_intervals':sum(r['af_intervals'] for r in rows),'multi_interval_records':sum(r['af_intervals']>1 for r in rows),'no_af_records':sum(r['af_intervals']==0 for r in rows),'continuous_af_on_annotated_suffix_records':sum(r['af_seconds']==r['rhythm_annotated_suffix_seconds'] for r in rows),'rhythm_markers':dict(rhythms),'nonrhythm_aux_notes':dict(comments),'sampling_rates':sorted({r['sampling_rate'] for r in rows}),'maximum_initial_unknown_seconds':max(r['initial_unknown_seconds'] for r in rows),'limitations':['Patient IDs not supplied by this header/reference audit; do not claim verified patient-disjoint splitting.','Waveforms not downloaded or quality checked.','Rhythm before first explicit marker is retained as unknown, not silently normal.','Reference suffix endpoint uses recording end; terminal AF may be observation-censored.']}
(P/'reference_audit_summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
