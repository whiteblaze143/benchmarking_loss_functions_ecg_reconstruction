"""Freeze source-patient splits and baseline contract before fitting."""
import csv,hashlib,json
from pathlib import Path
P=Path('/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1')
O=Path('/data/mithunmanivannan/codex_artifacts/af_density_20261120/rr_baseline_v1')

def freeze():
    O.mkdir(parents=True,exist_ok=True)
    assert not (O/'protocol.json').exists(),'Frozen protocol already exists'
    subjects=list(csv.DictReader((P/'subject_feasibility_ledger.csv').open()))
    rows=[]
    for multi in [False,True]:
        group=[r for r in subjects if (r['has_multi_interval_recording']=='True')==multi]
        group.sort(key=lambda r:hashlib.sha256(('af-density-20261120-v1:'+r['patient_id']).encode()).hexdigest())
        train=int(.6*len(group));val=int(.2*len(group))
        for i,r in enumerate(group):rows.append({'patient_id':r['patient_id'],'split':'train' if i<train else 'validation' if i<train+val else 'test','stratum_multi_interval_recording':multi})
    with (O/'patient_splits.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(sorted(rows,key=lambda r:r['patient_id']))
    protocol={'model':'sklearn HistGradientBoostingClassifier','parameters':{'learning_rate':.05,'max_iter':200,'max_leaf_nodes':31,'min_samples_leaf':50,'l2_regularization':1,'early_stopping':False,'random_state':42},'split_seed_string':'af-density-20261120-v1','split_counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},'window_beats':60,'tail':'Final overlapping actual 60-beat input assigns only previously uncovered tail; no padding or discarded tail.','calibration_proxy':'Exclude first 31 RR indices in first daily file and 61 in later files (one extra transition interval beyond measured 30/60 second device prefix). This is a conservative beat-index proxy, not exact elapsed-time alignment.','feature_range_ms':[200,4000],'feature_policy':'Statistics use valid intervals and originally adjacent valid pairs/triples only; invalid fraction retained. Fewer than four valid RR or two valid adjacent pairs is indeterminate.','fit_policy':'All eligible training windows; no convenience sample or cap; no early stopping on training-window splits.','weights':'Equal total weight per patient, with class balancing from training window targets only.','target':'Majority source AF RR labels in the full 60-beat feature window.','threshold':'Fixed candidates .01 through .99; maximize validation patient-mean beat sensitivity/specificity balanced accuracy on assigned intervals. Tie: closest to .5, then larger threshold.','primary_clock':'Native RR observed beat clock; per daily file after calibration proxy exclusion. No physical time-density claim.','indeterminate':'Keep coverage and denominators; file pattern metrics undefined if any assigned interval has no model representation.','source_label_convention':'RR onset index inclusive, termination index exclusive, per released tables.','provenance_limits':['Source RR detector channel availability unknown; not a measurement-matched single-lead baseline.','Dataset labels and reference patterns examined before this split; no AF detector test predictions examined.','Published source labels are used without new clinical adjudication.','Patient independence relies on released source patient identifiers.']}
    protocol['split_manifest_sha256']=hashlib.sha256((O/'patient_splits.csv').read_bytes()).hexdigest()
    (O/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(protocol['split_counts'])

if __name__=='__main__':freeze()
