import csv,hashlib,io,json,zipfile
from pathlib import Path
import joblib,numpy as np
from rr_protocol import P,O
from rr_features import NAMES

def main():
    summary=json.loads((O/'summary.json').read_text());assert summary['complete']
    frozen=json.loads((O/'frozen_before_test.json').read_text());checks=[]
    assert hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest()==frozen['checkpoint_sha256'];checks.append('checkpoint hash unchanged')
    for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    checks.append('all frozen source files unchanged')
    patients=list(csv.DictReader((O/'patient_splits.csv').open()));assert len({r['patient_id'] for r in patients})==152
    split=np.array([r['split'] for r in patients]);assert dict(zip(*np.unique(split,return_counts=True)))=={'train':90,'validation':30,'test':32};checks.append('patient partitions disjoint and complete')
    archive=np.load(O/'windows.npz');a={k:archive[k] for k in archive.files};archive.close();pi=a['patient_index'];scores=np.load(O/'all_window_scores.npy');assert len(scores)==519083==len(a['X'])
    assert a['X'].shape[1]==len(NAMES)==16;assert a['valid'].all();assert summary['train_fitted_windows']==324367;checks.append('all 519083 windows represented and all 324367 training windows fitted')
    model=joblib.load(O/'model.joblib');assert model.n_iter_==200 and model.early_stopping is False
    test=(split[pi]=='test')&a['valid'];replay=model.predict_proba(a['X'][test])[:,1];np.testing.assert_array_equal(replay,scores[test]);checks.append('all held-out scores replay bitwise')
    files=list(csv.DictReader((O/'file_manifest.csv').open()));assert len(files)==289
    for f in files:
        index=np.flatnonzero((a['record_index']==int(f['record_index']))&(a['day']==int(f['day'])))
        lo=a['assigned_start'][index];hi=a['assigned_end'][index]
        assert lo[0]==int(f['excluded_calibration_proxy_rr']) and hi[-1]==int(f['source_rr_count'])
        np.testing.assert_array_equal(hi[:-1],lo[1:]);assert (a['input_start'][index]>=lo[0]).all()
        assert (a['input_start'][index]+60<=hi[-1]).all()
        assert np.sum(a['assigned_positive'][index],dtype=np.int64)==int(f['assigned_af_rr'])
        assert len(set(pi[index]))==1 and patients[int(pi[index[0]])]['patient_id']==f['patient_id']
    checks.append('all daily-file assignment ranges complete, nonoverlapping, within source, patient-correct')
    # Independent scalar feature replay on actual source windows from each split.
    records=list(csv.DictReader((P/'iridia-af-metadata-v1.0.1.csv').open()))
    with zipfile.ZipFile(P/'iridia-af-records-v1.0.1.zip') as z:
        import h5py
        for s in ['train','validation','test']:
            i=int(np.flatnonzero(split[pi]==s)[0]);rid=records[int(a['record_index'][i])]['record_id'];day=int(a['day'][i]);start=int(a['input_start'][i])
            with h5py.File(io.BytesIO(z.read(f'iridia-af-records-v1.0.1/{rid}/{rid}_rr_{day:02d}.h5'))) as h:rr=h['rr'][start:start+60]
            v=rr[(rr>=200)&(rr<=4000)];adj=[float(rr[j+1]-rr[j]) for j in range(59) if 200<=rr[j]<=4000 and 200<=rr[j+1]<=4000]
            np.testing.assert_allclose(a['X'][i,NAMES.index('mean_rr_ms')],np.mean(v),rtol=1e-6)
            np.testing.assert_allclose(a['X'][i,NAMES.index('median_rr_ms')],np.median(v),rtol=1e-6)
            np.testing.assert_allclose(a['X'][i,NAMES.index('rmssd_ms')],np.sqrt(np.mean(np.square(adj))),rtol=1e-6)
    checks.append('independent mean/median/RMSSD from real train, validation and test RR values')
    # Recompute threshold selection from validation rows only.
    m=split[pi]=='validation';positive=a['assigned_positive'][m].astype(float);negative=a['assigned_count'][m]-positive;ids=pi[m];score=scores[m]
    pp=np.bincount(ids,weights=positive,minlength=152);nn=np.bincount(ids,weights=negative,minlength=152);candidates=[]
    for t in np.arange(1,100)/100:
        tp=np.bincount(ids,weights=positive*(score>=t),minlength=152);tn=np.bincount(ids,weights=negative*(score<t),minlength=152)
        accuracy=((tp[pp>0]/pp[pp>0]).mean()+(tn[nn>0]/nn[nn>0]).mean())/2
        candidates.append((accuracy,-abs(t-.5),t))
    assert max(candidates)[2]==summary['threshold']==.48;checks.append('threshold independently recovered from validation only')
    pattern=list(csv.DictReader((O/'file_pattern_metrics.csv').open()));assert len(pattern)==289
    assert len({r['patient_id'] for r in pattern if r['split']=='test'})==32;checks.append('pattern denominators include all daily files and all held-out patients')
    out={'all_pass':True,'checks':checks,'held_out_scores_replayed':int(test.sum()),'max_held_out_score_difference':float(np.max(np.abs(replay-scores[test])))}
    (O/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
