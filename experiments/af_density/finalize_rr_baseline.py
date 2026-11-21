"""Resume frozen evaluation after fixing repeated NPZ decompression bookkeeping.

No model is refitted and no predictions or threshold are changed. The original
frozen evaluation block is executed with all NPZ arrays cached once in memory.
"""
import csv,hashlib,io,json,textwrap,time,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from rr_protocol import P,O
from metrics import discrete_pattern_from_runs
from train_rr_baseline import runs
assert not (O/'summary.json').exists(),'Refusing to overwrite completed results'
frozen=json.loads((O/'frozen_before_test.json').read_text())
for name,digest in frozen['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
started=time.time();patients=list(csv.DictReader((O/'patient_splits.csv').open()));split=np.array([r['split'] for r in patients])
with np.load(O/'windows.npz') as archive:a={k:archive[k] for k in archive.files}
pi=a['patient_index'];scores=np.load(O/'all_window_scores.npy');valid=a['valid'];positive=a['assigned_positive'].astype(float);covered=a['assigned_count'].astype(float);negative=covered-positive;y=a['y']
threshold=frozen['threshold']['threshold'];masks={s:split[pi]==s for s in ['train','validation','test']}
summary={'threshold':threshold,'train_seconds':None,'fit_timing_limitation':'Original fit timing was not persisted before evaluation-only reconstruction was stopped. No refit performed.','iterations':200,'train_declared_windows':int(masks['train'].sum()),'train_fitted_windows':int((masks['train']&valid).sum()),'partitions':{}}
for s,mask in masks.items():
    m=mask&valid;pred=scores>=threshold;finite=np.isfinite(scores);ids=pi[mask]
    pos=np.bincount(ids,weights=positive[mask],minlength=len(patients));neg=np.bincount(ids,weights=negative[mask],minlength=len(patients))
    tp=np.bincount(ids,weights=(positive*pred*finite)[mask],minlength=len(patients));tn=np.bincount(ids,weights=(negative*(~pred)*finite)[mask],minlength=len(patients))
    summary['partitions'][s]={'patients':int(sum(split==s)),'declared_windows':int(mask.sum()),'usable_windows':int(m.sum()),'indeterminate_windows':int((mask&~valid).sum()),'beat_coverage':float(covered[m].sum()/covered[mask].sum()),'sensitivity_patient_mean_worst_case_indeterminate':float((tp[pos>0]/pos[pos>0]).mean()),'specificity_patient_mean_worst_case_indeterminate':float((tn[neg>0]/neg[neg>0]).mean()),'window_auroc':float(roc_auc_score(y[m],scores[m])),'window_auprc':float(average_precision_score(y[m],scores[m])),'window_brier':float(brier_score_loss(y[m],scores[m]))}
# Preserve the actual frozen evaluation logic; change only its array container
# from repeatedly opened NPZ entries to the in-memory dictionary above.
source=Path('experiments/af_density/train_rr_baseline.py').read_text()
block=source[source.index('    # Reconstruct each native-RR daily piece'):source.index("\nif __name__=='__main__'")]
exec(compile(textwrap.dedent(block),'frozen_rr_evaluation_block','exec'))
(O/'finalization_provenance.json').write_text(json.dumps({'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'finalizer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_preserved':True,'threshold_preserved':True,'score_archive_sha256':hashlib.sha256((O/'all_window_scores.npy').read_bytes()).hexdigest(),'change':'Metadata arrays cached once; frozen evaluation block otherwise unchanged.','finalization_seconds':time.time()-started},indent=2)+'\n')
