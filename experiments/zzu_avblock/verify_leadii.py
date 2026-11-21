"""Independent saved-ledger and training-transform checks; no heldout tuning."""
import json
import numpy as np,pandas as pd,torch
from leadii import OUT,BASE,cohort,sha,save
from protocol import locked_threshold

def main():
 f=cohort();ex=pd.read_parquet(OUT/'extraction.parquet');checks={}
 for name in ['M0','M1','M1_FP32']:
  directory=OUT/name;l=pd.read_parquet(directory/'prediction_manifest.parquet');lock=json.loads((directory/'evaluation_lock.json').read_text());cp=torch.load(directory/'best.pt',weights_only=True)
  v=l[l.partition=='validation'];t=l[l.partition=='test'];completed=v.decision_status.eq('completed');threshold=locked_threshold(v.loc[completed,'label'].to_numpy(),v.loc[completed,'score'].to_numpy(dtype=np.float32))
  checks[name+'_threshold_reproduces']=threshold==lock['threshold']
  checks[name+'_validation_specificity_at_least95']=float((v.loc[completed&v.label.eq(0),'score']<threshold).mean())>=.95
  checks[name+'_best_checkpoint_hash']=sha(directory/'best.pt')==lock['checkpoint_sha256']
  checks[name+'_correct_checkpoint_selection']=cp['epoch']==pd.read_csv(directory/'history.csv').validation_auroc_completed.idxmax()
  checks[name+'_full_test_denominator']=len(t)==1865 and t.label.sum()==190 and t.record_uid.is_unique
  checks[name+'_failures_have_no_scores']=t.loc[t.decision_status.ne('completed'),'score'].isna().all()
  checks[name+'_selected_and_final_distinct']=sha(directory/'best.pt')!=sha(directory/'final.pt')
  checks[name+'_decisions_match_threshold']=bool(np.array_equal(l.decision.to_numpy(),(l.score.to_numpy()>=threshold)&l.decision_status.eq('completed').to_numpy()))
 x=np.load(OUT/'leadII_mV.npy');tr=f.partition.eq('train').to_numpy();cp=torch.load(OUT/'M0/best.pt',weights_only=True)
 checks['M0_train_only_mean']=bool(np.isclose(x[tr].mean(dtype=np.float64),float(cp['mean'].squeeze()),atol=1e-7));checks['M0_train_only_std']=bool(np.isclose(x[tr].std(dtype=np.float64),float(cp['std'].squeeze()),atol=1e-7))
 fit=np.load(OUT/'phase_fit.npz');checks['M1_train_only_voltage_scale']=bool(np.isclose(fit['voltage_scale'],x[tr].std(dtype=np.float64),atol=1e-12))
 checks['extraction_exact_all_uid_alignment']=bool(np.array_equal(ex.record_uid,f.record_uid))
 checks['no_manufactured_intervals']=all(all(int(pair[0]) in peaks and int(pair[1]) in peaks and 150<=int(pair[1])-int(pair[0])<=1000 for pair in intervals) for peaks,intervals in zip(ex.rpeaks,ex.valid_intervals))
 checks['retained_fraction_from_observed_intervals']=all(abs(float(np.diff(np.array(list(intervals),dtype=np.int64).reshape(-1,2),axis=1).sum()/5000)-fraction)<1e-12 for intervals,fraction in zip(ex.valid_intervals,ex.represented_fraction))
 checks['original_reference_hashes_unchanged']=all(sha(BASE/name)==digest for name,digest in json.loads((OUT/'protocol.json').read_text())['reference_hashes'].items())
 save(OUT/'independent_ledger_checks.json',{k:bool(v) for k,v in checks.items()});assert all(checks.values()),checks
 print('INDEPENDENT_LEDGER_CHECKS',len(checks),'passed',flush=True)
if __name__=='__main__':main()
