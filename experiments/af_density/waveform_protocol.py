"""Prospectively freeze matched waveform inputs before observed-QRS extraction."""
import csv,hashlib,io,json,shutil
from pathlib import Path
import zipfile
import numpy as np
from rr_protocol import P,O as RR_ROOT
W=RR_ROOT.parent/'waveform_v1'
FS=200
SAMPLES=6000

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def window_layout(length,excluded):
    if length-excluded<SAMPLES:raise ValueError('File too short for actual 30-second input')
    starts=np.arange(excluded,length-SAMPLES+1,SAMPLES,dtype=np.int32)
    assigned=starts.copy();ends=starts+SAMPLES
    if ends[-1]<length:
        starts=np.r_[starts,length-SAMPLES].astype(np.int32)
        assigned=np.r_[assigned,ends[-1]].astype(np.int32)
        ends=np.r_[ends,length].astype(np.int32)
    return starts,assigned,ends

def count_overlap(starts,ends,intervals):
    result=np.zeros(len(starts),dtype=np.int64)
    for a,b in intervals:result+=np.maximum(0,np.minimum(ends,b)-np.maximum(starts,a))
    return result

def prepare():
    W.mkdir(exist_ok=True)
    assert not (W/'input_contract.json').exists(),'Input contract already frozen'
    staging=json.loads((P/'ecg_staging_verification.json').read_text());assert staging['all_pass'] and staging['files']==289
    old=json.loads((RR_ROOT/'protocol.json').read_text());assert sha(RR_ROOT/'patient_splits.csv')==old['split_manifest_sha256']
    shutil.copyfile(RR_ROOT/'patient_splits.csv',W/'patient_splits.csv')
    splits={r['patient_id']:r['split'] for r in csv.DictReader((W/'patient_splits.csv').open())}
    input_files={r['member']:r for r in map(json.loads,(P/'ecg_staging_manifest.jsonl').read_text().splitlines())}
    shapes={r['member']:r['shape'][0] for r in map(json.loads,(P/'waveform_audit.jsonl').read_text().splitlines()) if r['key']=='ecg'}
    metadata=list(csv.DictReader((P/'iridia-af-metadata-v1.0.1.csv').open()))
    files=[];arrays=[];label_hashes={};reference_rows=[];(W/'reference_ecg_labels').mkdir(exist_ok=True)
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
        for m in metadata:
            rid=m['record_id'];base=f'iridia-af-records-v1.0.1/{rid}/{rid}'
            raw=z.read(base+'_ecg_labels.csv');target=W/'reference_ecg_labels'/(rid+'.csv');target.write_bytes(raw);label_hashes[rid]=hashlib.sha256(raw).hexdigest()
            labels=list(csv.DictReader(io.StringIO(raw.decode())));offset=0
            for day in range(int(m['record_files'])):
                member=f'{base}_ecg_{day:02d}.h5';source=input_files[member];length=shapes[member];excluded=6000 if day==0 else 12000
                intervals=[]
                for event in labels:
                    sf=int(event['start_file_index']);ef=int(event['end_file_index'])
                    if sf<=day<=ef:
                        a=int(event['start_qrs_index']) if sf==day else 0;b=int(event['end_qrs_index']) if ef==day else length
                        assert 0<=a<b<=length
                        intervals.append((a,b))
                assert all(a[1]<b[0] for a,b in zip(intervals,intervals[1:]))
                starts,assigned,ends=window_layout(length,excluded);n=len(starts);index=len(files)
                af_input=count_overlap(starts,starts+SAMPLES,intervals);af_assigned=count_overlap(assigned,ends,intervals)
                assert (ends-assigned).sum()==length-excluded
                assert af_assigned.sum()==sum(max(0,b-max(a,excluded)) for a,b in intervals)
                files.append({'file_index':index,'patient_id':m['patient_id'],'record_id':rid,'partition':splits[m['patient_id']],'daily_file_index':day,'h5_path':source['path'],'h5_sha256':source['sha256'],'source_samples':length,'excluded_prefix_samples':excluded,'record_file_offset_samples':offset,'windows':n,'source_rr_annotation_disagreement_flag':rid=='record_023'})
                arrays.append({'file_index':np.full(n,index,dtype=np.int16),'input_start':starts,'assigned_start':assigned,'assigned_end':ends,'af_input_samples':af_input.astype(np.uint16),'af_assigned_samples':af_assigned.astype(np.uint16),'target':(af_input>=3000).astype(np.int8)})
                for a,b in intervals:reference_rows.append({'file_index':index,'start_sample':a,'end_sample_exclusive':b})
                offset+=length
    data={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]};np.savez_compressed(W/'windows.npz',**data)
    for name,rows in [('file_manifest.csv',files),('reference_intervals.csv',reference_rows)]:
        with (W/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts={s:{'patients':sum(v==s for v in splits.values()),'files':sum(r['partition']==s for r in files),'windows':sum(r['windows'] for r in files if r['partition']==s)} for s in ['train','validation','test']}
    contract={'stage':'Input/preprocessing only; no waveform classifier fitted','sampling_rate':FS,'input_samples':SAMPLES,'input_seconds':30,'stride_samples':SAMPLES,'tail':'Last actual 30-second input overlaps preceding window; assigns only uncovered tail, no padding or discarded samples.','target':'AF-positive iff at least 3000 of the 6000 actual input samples are source-labeled AF. Exact ties are positive.','reference':'Released ECG sample-index intervals, onset inclusive and first-NSR termination exclusive. Secondary RR label table is not substituted. record_023 disagreement retained as flag, not adjudicated or relabeled.','clock':'Original per-record source sample clock; daily file offsets and prefix exclusions preserved. Do not compress excluded time or join distinct monitoring records.','calibration':'Measured first 6000 samples in first daily file, 12000 in subsequent files excluded.','detector':'WFDB XQRS, fs=200, learn=True, independently on each actual 6000-sample observed channel. No other channel or outside-window context.','source_waveform_preprocessing_dtype':'float32 before feature/peak extraction, consistently in all stages','phase_interval_rule':'Authentic inclusive 300–2000 ms RR pairs; minimum two valid intervals; no manufactured beats.','phase_contract':'Original PCHIP 256-phase samples, 16 cells x 128 Nyström features; later whitening/landmarks/learned scaling training-only.','rr_features':'Same named 16 classical RR statistics, generalized to the actual variable observed interval count; 200–4000ms validity, min4 valid RR/min2 adjacent valid pairs; do not close gaps.','channel_identity':'Stored indices 0 and 1. Descriptor names I/II retained as provenance only; anatomical q, array-to-anatomy mapping and voltage units not assumed.','permitted_initial_set_modes':['categorical','q_ablated'],'geometry_modes':'continuous/projective not admitted without independently supported operators; no fabricated q coefficients.','failure':'Store peak errors, actual peak/accepted-beat counts, represented fractions and indeterminate flags. Retain every declared window in manifest.','training_weights_planned':'Exactly equal total per training patient via assigned-sample weights divided by that patient\'s total assigned samples; no subsequent class reweighting. Enforce by runtime assertion before fit.','availability_planned':'Named stored-channel subsets {0}, {1}, {0,1}; matched examples/availability for raw and set methods. No source-RR stream as primary matched input.','classifier_protocol':'Not yet frozen; no test threshold/model selection. Preprocessing contract is sealed before extraction.','prior_exposure':'Source-RR prototype and reference stress tests used these partitions; not an entirely untouched test cohort. No waveform model test predictions yet.','partitions':counts,'declared_windows':len(data['target']),'declared_files':len(files),'label_csv_sha256':label_hashes,'input_hashes':{n:sha(W/n) for n in ['patient_splits.csv','windows.npz','file_manifest.csv','reference_intervals.csv']},'staging_manifest_sha256':sha(P/'ecg_staging_manifest.jsonl'),'staging_verification_sha256':sha(P/'ecg_staging_verification.json')}
    (W/'input_contract.json').write_text(json.dumps(contract,indent=2)+'\n');print(json.dumps({'windows':len(data['target']),'partitions':counts},indent=2))
if __name__=='__main__':prepare()
