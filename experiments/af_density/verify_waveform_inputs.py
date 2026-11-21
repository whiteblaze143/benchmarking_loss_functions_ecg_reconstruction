"""Independent per-sample label and assignment verification, before extraction."""
import csv,hashlib,json,time
import numpy as np
from waveform_protocol import W

def main():
    start=time.time();contract=json.loads((W/'input_contract.json').read_text())
    for name,digest in contract['input_hashes'].items():
        with (W/name).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,name
    files=list(csv.DictReader((W/'file_manifest.csv').open()));patients=list(csv.DictReader((W/'patient_splits.csv').open()));split={r['patient_id']:r['split'] for r in patients}
    with np.load(W/'windows.npz') as z:a={k:z[k] for k in z.files}
    assert len(a['target'])==contract['declared_windows']==802491
    assert len(files)==289 and len(split)==152
    for row in files:
        fi=int(row['file_index']);length=int(row['source_samples']);cut=int(row['excluded_prefix_samples']);day=int(row['daily_file_index']);rid=row['record_id']
        assert row['partition']==split[row['patient_id']]
        assert cut==(6000 if day==0 else 12000)
        m=np.flatnonzero(a['file_index']==fi);lo=a['assigned_start'][m];hi=a['assigned_end'][m];inputs=a['input_start'][m]
        assert len(m)==int(row['windows']);assert lo[0]==cut and hi[-1]==length
        np.testing.assert_array_equal(hi[:-1],lo[1:]);assert (inputs>=cut).all() and (inputs+6000<=length).all()
        assert (lo>=inputs).all() and (hi<=inputs+6000).all()
        label_path=W/'reference_ecg_labels'/(rid+'.csv')
        assert hashlib.sha256(label_path.read_bytes()).hexdigest()==contract['label_csv_sha256'][rid]
        truth=np.zeros(length,dtype=np.int8)
        for label in csv.DictReader(label_path.open()):
            sf=int(label['start_file_index']);ef=int(label['end_file_index'])
            if sf<=day<=ef:truth[int(label['start_qrs_index']) if sf==day else 0:int(label['end_qrs_index']) if ef==day else length]=1
        prefix=np.r_[0,np.cumsum(truth,dtype=np.int64)]
        np.testing.assert_array_equal(a['af_input_samples'][m],prefix[inputs+6000]-prefix[inputs])
        np.testing.assert_array_equal(a['af_assigned_samples'][m],prefix[hi]-prefix[lo])
        np.testing.assert_array_equal(a['target'][m],(prefix[inputs+6000]-prefix[inputs]>=3000).astype(np.int8))
        assert int(a['af_assigned_samples'][m].sum(dtype=np.int64))==int(truth[cut:].sum(dtype=np.int64))
    out={'all_pass':True,'files':len(files),'patient_ids':len(split),'windows':len(a['target']),'all_input_targets_and_assigned_AF_sample_counts_match_independent_per_sample_reconstruction':True,'all_assignments_complete_and_nonoverlapping_after_calibration':True,'seconds':time.time()-start}
    (W/'input_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
