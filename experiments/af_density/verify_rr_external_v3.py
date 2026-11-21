import csv,hashlib,json
from pathlib import Path
import joblib,numpy as np
from rr_protocol import O
E=O/'external_shdb_v3'
f=json.loads((E/'frozen_before_external.json').read_text());s=json.loads((E/'summary.json').read_text());assert s['complete']
assert s['threshold_unchanged']==f['threshold']==.48
assert hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest()==f['checkpoint_sha256']
assert hashlib.sha256(Path('experiments/af_density/evaluate_rr_shdb_dtype32.py').read_bytes()).hexdigest()==f['source_sha256']
with np.load(E/'window_predictions.npz') as z:a={k:z[k] for k in z.files}
model=joblib.load(O/'model.joblib');valid=a['valid'];actual=model.predict_proba(a['X'][valid])[:,1];expected=a['scores'][valid]
delta=np.abs(actual-expected)
rows=list(csv.DictReader((E/'record_metrics.csv').open()));assert len(rows)==98 and len({r['subject_id'] for r in rows})==93
assert len(a['scores'])==s['windows']==172542
for i,r in enumerate(rows):
 m=a['record_index']==i;lo=a['assigned_start'][m];hi=a['assigned_end'][m]
 assert lo[0]==0 and hi[-1]==int(r['declared_rr']);np.testing.assert_array_equal(hi[:-1],lo[1:])
 assert (a['input_start'][m]>=0).all() and (a['input_start'][m]+60<=hi[-1]).all()
out={'all_pass':bool(np.array_equal(actual,expected)),'held_out_external_scores':len(actual),'bitwise_equal_scores':int((actual==expected).sum()),'max_absolute_score_difference':float(delta.max()),'classification_decision_changes':int(((actual>=.48)!=(expected>=.48)).sum()),'checkpoint_threshold_source_unchanged':True,'all_98_record_assignments_complete':True,'exact_duplicate_qrs_events_removed':sum(int(r['qrs_exact_duplicate_events_removed']) for r in rows)}
(E/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['all_pass']
