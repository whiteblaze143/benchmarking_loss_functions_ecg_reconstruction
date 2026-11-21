"""Audit released intervals against all actual HDF5 lengths; no labels inferred."""
import csv
import io
import json
import zipfile
from pathlib import Path
from metrics import charitos_density, discrete_pattern_from_runs

ROOT=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')

def main():
    status=json.loads((ROOT/'waveform_audit_status.json').read_text())
    assert status['complete'] and status['files']==578
    scan=[json.loads(line) for line in (ROOT/'waveform_audit.jsonl').read_text().splitlines()]
    assert len(scan)==578 and len({r['member'] for r in scan})==578
    lengths={r['member']:r['shape'][0] for r in scan}
    metadata=list(csv.DictReader((ROOT/'iridia-af-metadata-v1.0.1.csv').open()))
    rows=[];interval_rows=[];failures=[]
    with zipfile.ZipFile(ROOT/'iridia-af-records-v1.0.1.zip') as z:
        for m in metadata:
            rid=m['record_id'];base=f'iridia-af-records-v1.0.1/{rid}/{rid}'
            nf=int(m['record_files']);ecg_lengths=[lengths[f'{base}_ecg_{i:02d}.h5'] for i in range(nf)]
            rr_lengths=[lengths[f'{base}_rr_{i:02d}.h5'] for i in range(nf)]
            def offsets(ls):
                o=[0]
                for n in ls:o.append(o[-1]+n)
                return o
            ecg_offsets=offsets(ecg_lengths);rr_offsets=offsets(rr_lengths)
            ecg_labels=list(csv.DictReader(io.StringIO(z.read(base+'_ecg_labels.csv').decode())))
            rr_labels=list(csv.DictReader(io.StringIO(z.read(base+'_rr_labels.csv').decode())))
            pairs={};errors=[]
            for kind,labels,ls,off,index_key in [('ecg',ecg_labels,ecg_lengths,ecg_offsets,'qrs'),('rr',rr_labels,rr_lengths,rr_offsets,'rr')]:
                runs=[]
                for j,label in enumerate(labels):
                    sf=int(label['start_file_index']);ef=int(label['end_file_index'])
                    a=int(label[f'start_{index_key}_index']);b=int(label[f'end_{index_key}_index'])
                    if not (0<=sf<=ef<nf and 0<=a<ls[sf] and 0<=b<=ls[ef]):
                        errors.append({'kind':kind,'row':j,'error':'boundary_out_of_range','label':label});continue
                    a+=off[sf];b+=off[ef]
                    if b<=a or (runs and a<=runs[-1][1]):
                        errors.append({'kind':kind,'row':j,'error':'reversed_overlapping_or_touching_interval'});continue
                    runs.append((a,b))
                    if kind=='ecg':
                        interval_rows.append({'patient_id':m['patient_id'],'record_id':rid,'row':j,'start_sample':a,'end_sample_exclusive':b,'start_seconds':a/200,'end_seconds':b/200,'index_duration_seconds':(b-a)/200,'reported_duration_seconds':float(label['af_duration']),'duration_difference_seconds':(b-a)/200-float(label['af_duration']),'left_observation_censored':a==0,'end_at_observation_boundary':b==off[-1]})
                pairs[kind]=runs
            failures.extend({'record_id':rid,**e} for e in errors)
            duration=ecg_offsets[-1]/200
            af=sum(b-a for a,b in pairs['ecg'])/200
            beat_metrics=discrete_pattern_from_runs(pairs['rr'],rr_offsets[-1]) if not errors else {'density':None,'aggregation':None}
            rows.append({'patient_id':m['patient_id'],'record_id':rid,'boundary_errors':len(errors),'daily_files':nf,'waveform_seconds':duration,'metadata_seconds':float(m['record_seconds']),'waveform_samples':ecg_offsets[-1],'metadata_samples':int(m['record_samples']),'af_intervals':len(pairs['ecg']),'index_af_seconds':af,'index_af_burden':af/duration,'longest_af_seconds':max((b-a)/200 for a,b in pairs['ecg']),'continuous_charitos_density_raw':charitos_density([(a/200,b/200) for a,b in pairs['ecg']],duration) if not errors else None,'beat_density_raw':beat_metrics['density'],'beat_aggregation_raw':beat_metrics['aggregation'],'potential_calibration_seconds_if_each_file':nf*30,'raw_complete_24h_blocks':int(duration//86400),'raw_6h_prefix_available':duration>=21600,'raw_12h_prefix_available':duration>=43200,'raw_24h_prefix_available':duration>=86400})
    for name,data in [('timeline_record_audit.csv',rows),('timeline_intervals.csv',interval_rows)]:
        with (ROOT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (ROOT/'timeline_boundary_errors.json').write_text(json.dumps(failures,indent=2)+'\n')
    summary={'recordings':len(rows),'patient_ids':len({r['patient_id'] for r in rows}),'boundary_errors':len(failures),'waveform_samples_match_metadata':all(r['waveform_samples']==r['metadata_samples'] for r in rows),'waveform_hours':sum(r['waveform_seconds'] for r in rows)/3600,'index_af_hours':sum(r['index_af_seconds'] for r in rows)/3600,'nonfinite_values':sum(r['nonfinite_values'] for r in scan),'raw_24h_prefix_recordings':sum(r['raw_24h_prefix_available'] for r in rows),'raw_24h_prefix_patient_ids':len({r['patient_id'] for r in rows if r['raw_24h_prefix_available']}),'raw_complete_24h_blocks':sum(r['raw_complete_24h_blocks'] for r in rows),'limitations':['All metrics are descriptive raw-timeline values including calibration time; not final quality-qualified endpoints.','RR labels index transitions; beat-indexed results must not be interpreted as time-weighted density.','Finite samples do not establish clinical waveform quality.','Missing or calibration time has not been removed or bridged.']}
    (ROOT/'timeline_audit_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
