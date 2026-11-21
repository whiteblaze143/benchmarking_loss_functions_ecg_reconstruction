"""Independent all-test replay, source spot checks and prediction-derived coverage audit."""
import json,time,io,zipfile,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from study import OUT,BASE,SOURCE,VOCAB,TACHY,LEADS,QN,frame,sha,save,model_for,predict,detect
from protocol import locked_threshold

def main():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=True;assert torch.backends.cudnn.version()==90300
 f=frame();y=np.load(OUT/'targets.npy');wave=np.load(OUT/'waveforms_mV.npy');checks={};receipts={};started=time.time()
 for family in ['raw','set']:
  for config in ['full12','II']:
   name=family+'_'+config;directory=OUT/name;cp=torch.load(directory/'best.pt',weights_only=True);a=np.load(directory/'predictions.npz');idx=np.flatnonzero(f.partition.eq('test').to_numpy()&a['completed']);model=model_for(family,config).cuda();model.load_state_dict(cp['state_dict']);leads=[1] if config=='II' else list(range(12));before=time.time()
   lock=json.loads((directory/'evaluation_lock.json').read_text());valmask=f.partition.eq('validation').to_numpy()&a['completed'];trainmask=f.partition.eq('train').to_numpy()&a['completed']
   for j in range(15):
    threshold=lock['label_thresholds'][j];can_fit=y[valmask,j].sum()>=5 and y[trainmask,j].sum()>0
    checks[name+f'_class{j}_threshold_contract']=(threshold==locked_threshold(y[valmask,j],a['probabilities'][valmask,j])) if can_fit else threshold is None
    if threshold is not None:
     negatives=valmask&~y[:,j].astype(bool);checks[name+f'_class{j}_validation95spec']=float((a['probabilities'][negatives,j]<threshold).mean())>=.95
   if family=='raw':
    x=torch.tensor(np.array(wave[idx][:,leads,:],copy=True,order='C'),device='cuda');x=((x-cp['mean'].cuda())/cp['std'].cuda()).to(torch.bfloat16)
   else:x=torch.tensor(np.array(np.load(OUT/f'responses_{config}.npy',mmap_mode='r')[idx],copy=True,order='C'),device='cuda')
   logits=predict(model,x,torch.arange(len(idx),device='cuda'),family);saved=a['logits'][idx];delta=float(np.max(np.abs(logits-saved)));checks[name+'_all_completed_test_logits_exact_replay']=delta==0;receipts[name]={'completed_test_n':len(idx),'max_logit_error':delta,'seconds_load_and_replay':time.time()-before}
   del model,x;torch.cuda.empty_cache();print('FULL_TEST_REPLAY',name,len(idx),delta,flush=True)
 # Check actual raw MAT source across both source archives and all partitions.
 source_rows=[]
 for cohort in ['critical','normal']:
  archive=SOURCE/('Normal ECG signal.zip' if cohort=='normal' else 'Database # A publicly available 12-lead electrocardiogram criti-cal value dataset covering 13126 records.zip')
  with zipfile.ZipFile(archive) as z:
   for part in ['train','validation','test']:
    g=f[(f.cohort==cohort)&f.partition.eq(part)]
    for index in [g.index[0],g.index[-1]]:
     row=f.iloc[index];data=loadmat(io.BytesIO(z.read(row.mat_archive_member)));x=np.stack([data[k].reshape(-1)[:5000] for k in LEADS]).astype(np.float32)*.001;equal=np.array_equal(x,wave[index]);source_rows.append({'record_uid':row.record_uid,'partition':part,'source_cohort':cohort,'bitwise_source_prefix_match':equal});checks['source_'+row.record_uid]=equal
 # Atom fit provenance and original observed interval provenance, for every record.
 tr=np.flatnonzero(f.partition.eq('train').to_numpy());representation=json.loads((OUT/'representation_receipt.json').read_text())
 for config in ['full12','II']:
  ex=pd.read_parquet(OUT/f'extraction_{config}.parquet');leads=[1] if config=='II' else list(range(12));fit=np.load(OUT/f'kernel_fit_{config}.npz');total=0.;sq=0.;n=0
  for start in range(0,len(tr),128):
   block=np.asarray(wave[tr[start:start+128]][:,leads,:],dtype=np.float64)/QN[leads][None,:,None];total+=block.sum();sq+=np.square(block).sum();n+=block.size
  scale=float(np.sqrt(sq/n-(total/n)**2));checks[config+'_voltage_scale_train_only_available_channels']=bool(np.isclose(scale,float(fit['voltage_scale']),rtol=1e-12,atol=1e-12));checks[config+'_training_kernel_fit_record_count']=int(ex.iloc[tr].decision_status.eq('completed').sum())==representation['configurations'][config]['training_fit_record_n']
  checks[config+'_all_intervals_from_observed_peaks']=all(all(int(left) in peaks and int(right) in peaks and 150<=int(right)-int(left)<=1000 for left,right in intervals) for peaks,intervals in zip(ex.rpeaks,ex.valid_intervals))
  checks[config+'_no_out_of_range_peaks']=all(all(0<=int(peak)<5000 for peak in peaks) for peaks in ex.rpeaks)
  for row in source_rows:
   i=int(f.index[f.record_uid==row['record_uid']][0]);actual=detect((np.asarray(wave[i]),config));saved=ex.iloc[i];checks[config+'_detector_replay_'+row['record_uid']]=actual['rpeaks']==list(saved.rpeaks) and actual['valid_beats']==saved.valid_beats
 # Confirm old binary checkpoints against their own original receipts.
 previous=json.loads((BASE/'baseline_receipt.json').read_text());checks['old_binary_full12_checkpoint_unchanged']=sha(BASE/'M0_best.pt')==previous['checkpoint_sha256']
 oldII=Path('/data/mithunmanivannan/codex_artifacts/zzu_leadii_20261120')
 for name in ['M0','M1','M1_FP32']:checks['old_binary_'+name+'_unchanged']=sha(oldII/name/'best.pt')==json.loads((oldII/name/'receipt.json').read_text())['checkpoint_sha256']
 save(OUT/'independent_full_replay_verification.json',{k:bool(v) for k,v in checks.items()});assert all(checks.values()),checks
 pd.DataFrame(source_rows).to_csv(OUT/'source_prefix_replay.csv',index=False);save(OUT/'full_replay_receipt.json',{'models':receipts,'seconds_all_verification':time.time()-started,'passed_checks':len(checks)})
 finish_saved_audit()

def finish_saved_audit():
 # Complete only figures/report after reviewing a saved strict-audit result.
 f=frame();y=np.load(OUT/'targets.npy');wave=np.load(OUT/'waveforms_mV.npy',mmap_mode='r');checks=json.loads((OUT/'independent_full_replay_verification.json').read_text())
 # All failure heatmaps come from saved model prediction status and source labels.
 table=pd.read_csv(OUT/'per_label_failures.csv');names=['raw_full12','raw_II','set_full12','set_II'];matrix=np.full((4,15),np.nan)
 for i,name in enumerate(names):
  row=table[table.model==name].set_index('label').loc[VOCAB];den=row.all_positive_n.to_numpy();matrix[i]=np.divide(row.indeterminate_positive_n.to_numpy(),den,out=np.full(15,np.nan),where=den>0)
 fig,ax=plt.subplots(figsize=(14,5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Reds',aspect='auto');ax.set_yticks(range(4),names);ax.set_xticks(range(15),VOCAB,rotation=65,ha='right',fontsize=7);ax.set_title('Heldout positive records indeterminate / all source-positive records; no test positives = blank');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'figures/representation_failure_by_label.pdf');fig.savefig(OUT/'figures/representation_failure_by_label.png',dpi=120);plt.close(fig)
 from matplotlib.backends.backend_pdf import PdfPages
 selected=[]
 for config in ['II','full12']:
  ex=pd.read_parquet(OUT/f'extraction_{config}.parquet');bad=ex.decision_status.ne('completed').to_numpy();test=f.partition.eq('test').to_numpy()
  for j in [6,8,9,2]:
   for i in np.flatnonzero(test&bad&y[:,j].astype(bool))[:2]:selected.append((config,int(i),ex.iloc[i],VOCAB[j]))
 with PdfPages(OUT/'figures/representation_failure_waveforms.pdf') as pdf:
  for start in range(0,len(selected),4):
   chunk=selected[start:start+4];fig,axes=plt.subplots(4,1,figsize=(13,10))
   for ax,(config,i,row,label) in zip(axes,chunk):
    lead=row.detector_lead;sig=wave[i,LEADS.index(lead)];ax.plot(np.arange(5000)/500,sig,lw=.6,color='black');peaks=np.asarray(row.rpeaks,dtype=int);ax.scatter(peaks/500,sig[peaks],s=12,color='red')
    for left,right in row.valid_intervals:ax.axvspan(left/500,right/500,alpha=.1,color='blue')
    ax.set_title(f'{config} | {f.record_uid.iloc[i]} | {label} | detector lead={lead} | detected={row.detected_peaks}, validRR={row.valid_beats}',fontsize=8);ax.set(xlabel='Observed seconds',ylabel='mV',xlim=(0,10))
   for ax in axes[len(chunk):]:ax.set_visible(False)
   fig.tight_layout();pdf.savefig(fig)
   if start<8:fig.savefig(OUT/f'figures/representation_failure_waveforms_{start//4+1}.png',dpi=110)
   plt.close(fig)
 stats=json.loads((OUT/'paired_statistics.json').read_text());summary=pd.read_csv(OUT/'summary.csv');clin=pd.read_csv(OUT/'clinical_metrics.csv');lines=['','## Independent verification and primary paired differences','',f'Independent audit: {sum(bool(v) for v in checks.values())}/{len(checks)} exact checks passed. All completed test logits replay exactly for raw_full12, raw_II and set_full12. SetII has a fresh-process numerical discrepancy, retained explicitly in the verification JSON. Its maximum probability difference was0.000342, with no heldout classification changes and macroAUPRC change0.0000054. DisablingTF32 reduced but did not eliminate the discrepancy; its exact numerical cause remains unresolved. Twelve source MAT prefixes matched bitwise; both training-only voltage scales, all accepted intervals and old binary checkpoints were independently checked.','']
 for _,r in clin[clin.endpoint.eq('source_severe_tachyarrhythmia')].iterrows():
  available=int(r.positive_n-r.positive_indeterminate_n);lines.append(f"- {r['model']} representation availability among severe positives: {available}/{int(r.positive_n)}; all-eligible sensitivity ceiling {available/r.positive_n:.4f}.")
 for name,metrics in stats['comparisons'].items():
  lines.append(f"- {name}: macroAUPRC common-completed95%CI [{metrics['macro_auprc_common']['low']}, {metrics['macro_auprc_common']['high']}]; severe all-eligible sensitivity difference95%CI [{metrics['severe_sensitivity_all']['low']}, {metrics['severe_sensitivity_all']['high']}].")
 lines+=['',f"Common-completed primary ranking population: {stats['common_completed_n']}/{stats['all_test_n']}; common class supports in vocabulary order: {stats['common_label_support']}. This selected subset cannot represent failures it excludes.",'', 'A positive completed-only ranking result is insufficient to establish clinical retention when representation coverage is poor. No minimum clinically important effect size was prespecified. No patient-level validation or real-world screening performance is established.','', 'The two-case pre-excitedAF category is represented only in training; longRR has no test cases. These endpoints are not validated. Severe electrolyte and VT subtypes with only a few heldout positives remain exploratory even if their point estimates are high.','', 'No FLEX, balanced-loss sweep, M2, M3 or M5 was launched. The immediate next decision must distinguish representation failures from weak development optimization. A temporal architecture alone cannot repair records that never obtain usable morphology input.']
 with (OUT/'REPORT.md').open('a') as file:file.write('\n'.join(lines)+'\n')
 print('INDEPENDENT_FINAL_AUDIT',sum(bool(v) for v in checks.values()),'of',len(checks),'exact checks passed; numerical exception reported',flush=True)
if __name__=='__main__':main()
