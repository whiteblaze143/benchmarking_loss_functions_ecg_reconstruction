"""Metrics and paired uncertainty from immutable 15-label prediction archives."""
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import OUT,BASE,SOURCE,REPO,VOCAB,TACHY,LEADS,frame,sha,save,targets
NAMES=['raw_full12','raw_II','set_full12','set_II']

def ranking(y,s):
 if y.sum()==0:return {'auroc':None,'auprc':None,'brier':float(np.mean((s-y)**2)) if len(y) else None}
 return {'auroc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,'auprc':float(average_precision_score(y,s)),'brier':float(np.mean((s-y)**2))}

def clinical(y,score,valid,threshold):
 pred=valid&(score>=threshold);pos=y==1;neg=~pos
 return {'all_n':len(y),'positive_n':int(pos.sum()),'negative_n':int(neg.sum()),'completed_n':int(valid.sum()),'indeterminate_n':int((~valid).sum()),'detected_positive_n':int(pred[pos].sum()),'false_positive_n':int(pred[neg].sum()),'positive_indeterminate_n':int((~valid&pos).sum()),'negative_indeterminate_n':int((~valid&neg).sum()),'sensitivity_all_eligible':float(pred[pos].mean()) if pos.any() else None,'specificity_completed':float((~pred[neg&valid]).mean()) if (neg&valid).any() else None,'correct_negative_fraction_all_eligible':float((valid[neg]&~pred[neg]).mean()) if neg.any() else None,'false_positive_fraction_all_eligible':float(pred[neg].mean()) if neg.any() else None,'threshold':float(threshold)}

def weighted_ap(y,score,weights):
 order=np.argsort(-score,kind='stable');s=score[order];label=y[order];w=weights[:,order];cuts=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
 tp=np.cumsum(w*label[None,:],axis=1)[:,cuts];total=np.cumsum(w,axis=1)[:,cuts];precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0);delta=np.diff(np.c_[np.zeros(len(weights)),tp],axis=1);den=tp[:,-1]
 return np.divide((delta*precision).sum(axis=1),den,out=np.full(len(weights),np.nan),where=den>0)


def weighted_auc(y,score,weights):
 order=np.argsort(-score,kind='stable');s=score[order];label=y[order];w=weights[:,order];cuts=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
 tp=np.cumsum(w*label[None,:],axis=1)[:,cuts];fp=np.cumsum(w*(1-label)[None,:],axis=1)[:,cuts];den=tp[:,-1]*fp[:,-1]
 area=(np.diff(np.c_[np.zeros(len(w)),fp],axis=1)*(tp+np.c_[np.zeros(len(w)),tp[:,:-1]])*.5).sum(axis=1)
 return np.divide(area,den,out=np.full(len(w),np.nan),where=den>0)

def bootstrap(f,y,pred,valid,locks,classes):
 # All contrasts use the same group draws; rankings use an explicitly common-completed subset.
 groups,group_index=np.unique(f.record_group.to_numpy(),return_inverse=True);rng=np.random.default_rng(42);common=np.logical_and.reduce([valid[n] for n in NAMES]);counts={};stats={}
 comparisons={'leadII_set_minus_raw':('set_II','raw_II'),'full12_set_minus_raw':('set_full12','raw_full12'),'raw_II_minus_full12':('raw_II','raw_full12'),'set_II_minus_full12':('set_II','set_full12')}
 group_y=(y[:,TACHY].sum(axis=1)>0).astype(int);decisions={n:valid[n]&(np.nanmax(pred[n][:,TACHY],axis=1,initial=-np.inf)>=locks[n]['severe_group_threshold']) for n in NAMES}
 values={n:{'macro_auprc_common':[],'severe_sensitivity_all':[]} for n in NAMES}
 # Per-class bootstrap intervals use model-specific completed rows and preserve undefined replicates.
 label_values={n:{j:[] for j in range(15)} for n in NAMES};auc_values={n:{j:[] for j in range(15)} for n in NAMES}
 for start in range(0,2000,100):
  gweight=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=100);w=gweight[:,group_index].astype(float);posden=w[:,group_y==1].sum(axis=1)
  for name in NAMES:
   sensitivity=np.divide((w*(decisions[name]&group_y.astype(bool))[None,:]).sum(axis=1),posden,out=np.full(100,np.nan),where=posden>0);values[name]['severe_sensitivity_all'].extend(sensitivity.tolist())
   aps=[]
   for j in range(15):
    vc=valid[name];ap=weighted_ap(y[vc,j],pred[name][vc,j],w[:,vc]);label_values[name][j].extend(ap.tolist());auc_values[name][j].extend(weighted_auc(y[vc,j],pred[name][vc,j],w[:,vc]).tolist())
    if j in classes:aps.append(weighted_ap(y[common,j],pred[name][common,j],w[:,common]))
   macro=np.mean(aps,axis=0);values[name]['macro_auprc_common'].extend(macro.tolist())
  print('BOOTSTRAP',start+100,flush=True)
 def interval(v):
  v=np.asarray(v);v=v[np.isfinite(v)];return {'low':float(np.quantile(v,.025)) if len(v) else None,'high':float(np.quantile(v,.975)) if len(v) else None,'valid_replicates':len(v)}
 for label,(a,b) in comparisons.items():stats[label]={key:interval(np.array(values[a][key])-np.array(values[b][key])) for key in values[a]}
 for name in NAMES:counts[name]={key:interval(v) for key,v in values[name].items()}
 perlabel=[{'model':name,'label_index':j,**interval(v),**{'auroc_'+k:value for k,value in interval(auc_values[name][j]).items()}} for name,table in label_values.items() for j,v in table.items()]
 pd.DataFrame(perlabel).to_csv(OUT/'per_label_auprc_intervals.csv',index=False)
 save(OUT/'paired_statistics.json',{'comparisons':stats,'models':counts,'ranking_population':'same heldout records completed by all four models; selection/coverage bias explicitly remains','common_completed_n':int(common.sum()),'all_test_n':len(f),'common_label_support':y[common].sum(axis=0).astype(int).tolist(),'replicates':2000,'unit':'conservative record group, not verified patient','conditional':'fixed fitted models and validation-selected thresholds; no training/threshold uncertainty or patient-overlap correction','formal_pvalues':None})
 return common

def main():
 f=frame();test=f.partition.eq('test').to_numpy();ft=f[test].reset_index(drop=True);yy=np.load(OUT/'targets.npy');y=yy[test];proto=json.loads((OUT/'protocol.json').read_text());classes=proto['macro_indices'];pred={};valid={};locks={};archives={};rows=[];clinical_rows=[];cal=[];fail=[]
 for name in NAMES:
  a=np.load(OUT/name/'predictions.npz');assert np.array_equal(a['record_uid'],f.record_uid) and np.array_equal(a['target'],yy);archives[name]=a;pred[name]=a['probabilities'][test];valid[name]=a['completed'][test]&np.isfinite(pred[name]).all(axis=1);locks[name]=json.loads((OUT/name/'evaluation_lock.json').read_text());v=valid[name];s=pred[name]
  group_y=(y[:,TACHY].sum(axis=1)>0).astype(int);group_score=np.nanmax(s[:,TACHY],axis=1,initial=-np.inf);clinical_rows.append({'model':name,'endpoint':'source_severe_tachyarrhythmia',**clinical(group_y,group_score,v,locks[name]['severe_group_threshold'])})
  for j,label in enumerate(VOCAB):
   r={'model':name,'label_index':j,'label':label,'all_test_positive_n':int(y[:,j].sum()),'completed_positive_n':int(y[v,j].sum()),'completed_n':int(v.sum()),'all_test_n':len(y),'prevalence_all':float(y[:,j].mean()),'prevalence_completed':float(y[v,j].mean()) if v.any() else None,'primary_macro_included':j in classes,**ranking(y[v,j],s[v,j])};rows.append(r)
   threshold=locks[name]['label_thresholds'][j]
   if threshold is not None:clinical_rows.append({'model':name,'endpoint':label,**clinical(y[:,j].astype(int),s[:,j],v,threshold)})
   fail.append({'model':name,'label':label,'all_positive_n':int(y[:,j].sum()),'indeterminate_positive_n':int((~v&y[:,j].astype(bool)).sum())})
   for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
    mask=v&(s[:,j]>=lo)&(s[:,j]<hi if hi<1 else s[:,j]<=hi);n=int(mask.sum());cal.append({'model':name,'label':label,'bin_low':lo,'bin_high':hi,'n':n,'mean_probability':float(s[mask,j].mean()) if n else None,'positive_fraction':float(y[mask,j].mean()) if n else None})
 strata=[]
 for name in NAMES:
  gy=(y[:,TACHY].sum(axis=1)>0).astype(int);s=np.nanmax(pred[name][:,TACHY],axis=1,initial=-np.inf);v=valid[name]
  for cohort in ['normal','critical']:
   mask=ft.cohort.eq(cohort).to_numpy();strata.append({'model':name,'source_cohort':cohort,**clinical(gy[mask],s[mask],v[mask],locks[name]['severe_group_threshold'])})
  for j,label in enumerate(VOCAB):
   mask=y[:,j].astype(bool)&~gy.astype(bool)
   if mask.any():strata.append({'model':name,'source_cohort':'other_critical_carrying_'+label,**clinical(gy[mask],s[mask],v[mask],locks[name]['severe_group_threshold'])})
 pd.DataFrame(strata).to_csv(OUT/'severe_group_false_alarm_strata.csv',index=False)
 model_support=[]
 for name,a in archives.items():
  for part in ['train','validation','test']:
   mask=f.partition.eq(part).to_numpy();vc=mask&a['completed']
   for j,label in enumerate(VOCAB):model_support.append({'model':name,'partition':part,'label':label,'eligible_records':int(mask.sum()),'eligible_positive_n':int(yy[mask,j].sum()),'completed_records':int(vc.sum()),'completed_positive_n':int(yy[vc,j].sum())})
 pd.DataFrame(model_support).to_csv(OUT/'model_partition_label_support.csv',index=False)
 errors=[]
 for name in NAMES:
  for j,label in enumerate(VOCAB):
   threshold=locks[name]['label_thresholds'][j]
   if threshold is None:continue
   fp=valid[name]&(pred[name][:,j]>=threshold)&~y[:,j].astype(bool)
   for k,truth in enumerate(VOCAB):errors.append({'model':name,'predicted_label':label,'true_source_label':truth,'false_alert_records':int((fp&y[:,k].astype(bool)).sum())})
   errors.append({'model':name,'predicted_label':label,'true_source_label':'Normal','false_alert_records':int((fp&(y.sum(axis=1)==0)).sum())})
 pd.DataFrame(errors).to_csv(OUT/'diagnosis_false_alert_matrix.csv',index=False)
 per=pd.DataFrame(rows);per.to_csv(OUT/'per_label_metrics.csv',index=False);clin=pd.DataFrame(clinical_rows);clin.to_csv(OUT/'clinical_metrics.csv',index=False);pd.DataFrame(cal).to_csv(OUT/'calibration.csv',index=False);pd.DataFrame(fail).to_csv(OUT/'per_label_failures.csv',index=False)
 # Paired archive retains labels, logits, scores and statuses for exactly the same source records.
 pair=ft[['record_uid','record_group','labels_json','duration_seconds','cohort','label_count','einthoven_max_residual_uV']].copy();qc=pd.read_parquet(OUT/'prefix_quality.parquet').set_index('record_uid').loc[pair.record_uid].reset_index();pair=pair.merge(qc,on='record_uid',validate='one_to_one')
 quality_rows=[]
 allqc=pd.read_parquet(OUT/'prefix_quality.parquet')
 for j,label in enumerate(VOCAB):
  mask=yy[:,j].astype(bool);flat=allqc[[lead+'_flat' for lead in LEADS]].to_numpy()
  quality_rows.append({'label':label,'eligible_n':int(mask.sum()),'any_prefix_flat_n':int(flat[mask].any(axis=1).sum()),'LeadII_flat_n':int(flat[mask,1].sum())})
 pd.DataFrame(quality_rows).to_csv(OUT/'prefix_quality_by_label.csv',index=False)
 for name in NAMES:
  pair[name+'_completed']=valid[name]
  for j in range(15):pair[f'{name}_probability_{j}']=pred[name][:,j]
 pair.to_parquet(OUT/'paired_test_predictions.parquet',index=False)
 common=bootstrap(ft,y,pred,valid,locks,classes)
 summary=[]
 for name in NAMES:
  macro_values=per[(per.model==name)&per.primary_macro_included].auprc
  common_values=[ranking(y[common,j],pred[name][common,j])['auprc'] for j in classes]
  clinical_group=clin[(clin.model==name)&clin.endpoint.eq('source_severe_tachyarrhythmia')].iloc[0].to_dict()
  summary.append({'model':name,'completed_n':int(valid[name].sum()),'indeterminate_n':int((~valid[name]).sum()),'macro_AUPRC_completed':float(macro_values.mean()) if macro_values.notna().all() else None,'macro_AUPRC_common_completed':float(np.mean(common_values)) if all(v is not None for v in common_values) else None,'primary_macro_label_n':len(classes),'group_sensitivity_all':clinical_group['sensitivity_all_eligible'],'group_specificity_completed':clinical_group['specificity_completed'],'group_positive_n':clinical_group['positive_n'],'group_detected_n':clinical_group['detected_positive_n'],'group_false_positive_n':clinical_group['false_positive_n']})
 pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
 # Quality/extraction tables include every cohort record, never model-selected cases alone.
 for config in ['full12','II']:
  ext=pd.read_parquet(OUT/f'extraction_{config}.parquet');assert np.array_equal(ext.record_uid,f.record_uid);data=f[['record_uid','partition','labels_json']].merge(ext,on='record_uid',validate='one_to_one');data.to_parquet(OUT/f'extraction_with_partitions_{config}.parquet',index=False)
  stats=[]
  for partition in ['train','validation','test']:
   sel=f.partition.eq(partition).to_numpy()
   for j,label in enumerate(VOCAB):
    g=ext[sel&yy[:,j].astype(bool)];stats.append({'partition':partition,'label':label,'n':len(g),'indeterminate_n':int(g.decision_status.ne('completed').sum()),'median_detected_peaks':float(g.detected_peaks.median()) if len(g) else None,'median_valid_beats':float(g.valid_beats.median()) if len(g) else None,'median_represented_fraction':float(g.represented_fraction.median()) if len(g) else None})
  pd.DataFrame(stats).to_csv(OUT/f'extraction_label_support_{config}.csv',index=False)
 figures(f,ft,y,pred,valid,locks,per,clin)
 verify(f,yy,proto,archives)
 report(f,proto,summary,clin)
 save(OUT/'delivery_manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
 print(pd.DataFrame(summary).to_string(index=False));print('COMPLETE',flush=True)

def figures(f,ft,y,pred,valid,locks,per,clin):
 figdir=OUT/'figures';figdir.mkdir(exist_ok=True);support=pd.read_csv(OUT/'label_support.csv');labels=['MI','Ischemia','AVB','HypoK','R-on-T','PE','VF','T alternans','V flutter','TdP','Poly VT','HyperK','Bidirectional VT','Long RR','Pre-excited AF']
 fig,axes=plt.subplots(1,2,figsize=(16,7));axes[0].barh(np.arange(15),support.eligible_10s_n);axes[0].set_yticks(np.arange(15),labels);axes[0].set_xscale('log');axes[0].invert_yaxis();axes[0].set_title('Eligible source records; multilabel counts')
 overlap=pd.read_csv(OUT/'label_overlap.csv',index_col=0).to_numpy();im=axes[1].imshow(np.log1p(overlap),cmap='viridis');axes[1].set_xticks(range(15),labels,rotation=90,fontsize=7);axes[1].set_yticks(range(15),labels,fontsize=7);axes[1].set_title('log(1 + observed label co-occurrence)');fig.colorbar(im,ax=axes[1]);fig.tight_layout();fig.savefig(figdir/'label_distribution_overlap.pdf');fig.savefig(figdir/'label_distribution_overlap.png',dpi=130);plt.close(fig)
 err=pd.read_csv(OUT/'diagnosis_false_alert_matrix.csv');fig,axes=plt.subplots(2,2,figsize=(16,13))
 for ax,name in zip(axes.ravel(),NAMES):
  matrix=err[err.model==name].pivot(index='true_source_label',columns='predicted_label',values='false_alert_records');im=ax.imshow(np.log1p(matrix.to_numpy()),aspect='auto',cmap='magma');ax.set_yticks(range(len(matrix.index)),matrix.index,fontsize=6);ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=70,ha='right',fontsize=6);ax.set_title(name+' false alerts: log(1+record count)',fontsize=9);fig.colorbar(im,ax=ax)
 fig.tight_layout();fig.savefig(figdir/'diagnosis_false_alert_patterns.pdf');plt.close(fig)
 ci=pd.read_csv(OUT/'per_label_auprc_intervals.csv');fig,axes=plt.subplots(1,2,figsize=(16,8))
 for k,name in enumerate(NAMES):
  table=per[per.model==name].sort_values('label_index');interval=ci[ci.model==name].sort_values('label_index');pos=np.arange(15)+(k-1.5)*.16
  axes[0].plot(table.auprc,pos,'o',ms=4,label=name);axes[0].hlines(pos,interval.low,interval.high,alpha=.5)
  axes[1].plot(table.auroc,pos,'o',ms=4,label=name);axes[1].hlines(pos,interval.auroc_low,interval.auroc_high,alpha=.5)
 for ax,title in zip(axes,['Per-label AUPRC with conditional95% intervals','Per-label AUROC; undefined remains blank']):ax.set_yticks(range(15),[f'{lab} (test n={int(n)})' for lab,n in zip(labels,support.test)]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_title(title);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(figdir/'per_label_performance.pdf');fig.savefig(figdir/'per_label_performance.png',dpi=130);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,5));heat=[];common=np.logical_and.reduce([valid[n] for n in NAMES]);common_rows=[]
 for name in NAMES:
  for j,label in enumerate(VOCAB):common_rows.append({'model':name,'label_index':j,'label':label,'n':int(common.sum()),'positive_n':int(y[common,j].sum()),**ranking(y[common,j],pred[name][common,j])})
 common_table=pd.DataFrame(common_rows);common_table.to_csv(OUT/'per_label_common_completed_metrics.csv',index=False)
 for family in ['raw','set']:
  aa=common_table[common_table.model==family+'_II'].sort_values('label_index').auprc.to_numpy();bb=common_table[common_table.model==family+'_full12'].sort_values('label_index').auprc.to_numpy();heat.append(aa-bb)
 im=ax.imshow(np.array(heat),vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_xticks(range(15),labels,rotation=60,ha='right');ax.set_yticks([0,1],['Raw II minus full12','Set II minus full12']);ax.set_title('Paired AUPRC change on identical common-completed records');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'lead_degradation_heatmap.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(16,7));included=[j for j in range(15) if locks['raw_full12']['label_thresholds'][j] is not None]
 for ax,j in zip(axes.ravel(),included):
  for family in ['raw','set']:
   sens=[]
   for config in ['full12','II']:
    name=family+'_'+config;threshold=locks[name]['label_thresholds'][j]
    sens.append(float((valid[name]&(pred[name][:,j]>=threshold))[y[:,j]==1].mean()) if threshold is not None and y[:,j].sum()>0 else np.nan)
   ax.plot([0,1],sens,'o-',label=family)
  ax.set_xticks([0,1],['full12','LeadII']);ax.set_ylim(0,1);ax.set_title(f'{labels[j]} (test positive n={int(y[:,j].sum())})',fontsize=8)
 axes[0,0].legend(fontsize=7);fig.suptitle('All-eligible sensitivity; failures remain nondetections; validation95% specificity policy',fontsize=10);fig.tight_layout();fig.savefig(figdir/'sensitivity_by_available_leads.pdf');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,6));gy=(y[:,TACHY].sum(axis=1)>0).astype(int)
 for name in NAMES:
  v=valid[name];s=pred[name][v][:,TACHY].max(axis=1)
  if len(np.unique(gy[v]))==2:
   fp,tp,_=roc_curve(gy[v],s);ax.plot(fp,tp,label=f'{name} (completed n={int(v.sum())})')
 ax.set(xlabel='False-positive fraction among completed controls',ylabel='Sensitivity among completed positives',title='Source severe-tachyarrhythmia endpoint');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(figdir/'severe_group_ROC.pdf');plt.close(fig)
 # Prediction-derived false-negative composition for the clinical group (labels may overlap).
 errors=[];examples=[]
 for name in NAMES:
  v=valid[name];s=np.nanmax(pred[name][:,TACHY],axis=1,initial=-np.inf);decision=v&(s>=locks[name]['severe_group_threshold']);fn=gy.astype(bool)&~decision
  for j in TACHY:errors.append({'model':name,'label':VOCAB[j],'group_FN_carrying_label':int((fn&y[:,j].astype(bool)).sum())})
  if name in ['raw_II','set_II']:
   cases={'rare_true_positive':np.flatnonzero(gy.astype(bool)&decision),'rare_false_negative':np.flatnonzero(fn),'hard_negative_false_alarm':np.flatnonzero(~gy.astype(bool)&decision&ft.cohort.eq('critical').to_numpy())}
   for kind,idx in cases.items():
    for i in idx[:3]:examples.append({'model':name,'kind':kind,'test_index':int(i),'record_uid':ft.record_uid.iloc[i],'labels':ft.labels_json.iloc[i],'group_score':float(s[i]) if v[i] else None,'completed':bool(v[i])})
 pd.DataFrame(errors).to_csv(OUT/'false_negative_composition.csv',index=False);pd.DataFrame(examples).to_csv(OUT/'waveform_example_ledger.csv',index=False)
 wave=np.load(OUT/'waveforms_mV.npy',mmap_mode='r');lookup={uid:i for i,uid in enumerate(f.record_uid)}
 from matplotlib.backends.backend_pdf import PdfPages
 with PdfPages(figdir/'actual_waveform_errors.pdf') as pdf:
  for start in range(0,len(examples),4):
   fig,axes=plt.subplots(4,1,figsize=(13,10))
   for ax,row in zip(axes,examples[start:start+4]):
    ax.plot(np.arange(5000)/500,wave[lookup[row['record_uid']],1],lw=.5,color='black');ax.set_title(f"{row['model']} | {row['kind']} | {row['record_uid']} | group score={row['group_score']}\n{row['labels']}",fontsize=8);ax.set(xlabel='Observed seconds',ylabel='Lead II (mV)')
   for ax in axes[len(examples[start:start+4]):]:ax.set_visible(False)
   fig.tight_layout();pdf.savefig(fig)
   if start<8:fig.savefig(figdir/f'waveform_examples_{start//4+1}.png',dpi=110)
   plt.close(fig)
 # Per-class reliability panels: only macro-included labels with adequate heldout positives.
 fig,axes=plt.subplots(2,5,figsize=(17,7));cal=pd.read_csv(OUT/'calibration.csv');include=pd.read_csv(OUT/'label_support.csv').query('primary_macro_included').label.tolist()
 for ax,label in zip(axes.ravel(),include):
  for name in NAMES:
   g=cal[(cal.model==name)&cal.label.eq(label)&cal.n.ge(20)];ax.plot(g.mean_probability,g.positive_fraction,'o-',ms=3,label=name)
  ax.plot([0,1],[0,1],'k:',lw=.5);ax.set_title(label,fontsize=7);ax.set_xlim(0,1);ax.set_ylim(0,1)
 axes[0,0].legend(fontsize=6);fig.tight_layout();fig.savefig(figdir/'calibration.pdf');plt.close(fig)

def verify(f,y,proto,archives):
 checks={'cohort20113':len(f)==20113,'all15targets':y.shape==(20113,15),'normal_allzero':bool((y[f.cohort.eq('normal')]==0).all()),'source_hashes_unchanged':all(sha(p)==h for p,h in proto['source_hashes'].items()),'old_split_unchanged':sha(BASE/'split_manifest.parquet')==proto['old_split_sha256'],'old_predictions_unchanged':sha(BASE/'prediction_manifest.parquet')==proto['old_prediction_sha256'],'group_partitions_disjoint':f.groupby('record_group').partition.nunique().max()==1}
 old=pd.read_parquet(BASE/'split_manifest.parquet').set_index('record_uid').loc[f.record_uid];checks['original_partitions_preserved']=np.array_equal(old.partition.to_numpy(),f.partition.to_numpy());checks['targets_exactly_source_labels']=np.array_equal(targets(f),y)
 for name,a in archives.items():
  lock=json.loads((OUT/name/'evaluation_lock.json').read_text());cp=torch_load(OUT/name/'best.pt');history=pd.read_csv(OUT/name/'history.csv');checks[name+'_best_epoch']=cp['epoch']==int(history.validation_macro_auprc.idxmax());checks[name+'_hash']=sha(OUT/name/'best.pt')==lock['checkpoint_sha256'];validation=f.partition.eq('validation').to_numpy()&a['completed'];s=a['probabilities'];gy=(y[:,TACHY].sum(axis=1)>0).astype(int);threshold=locked_threshold_native(gy[validation],s[validation][:,TACHY].max(axis=1));checks[name+'_group_threshold']=threshold==lock['severe_group_threshold'];checks[name+'_group_validation95spec']=float((s[validation&~gy.astype(bool)][:,TACHY].max(axis=1)<threshold).mean())>=.95;checks[name+'_failed_rows_no_scores']=bool(np.isnan(s[~a['completed']]).all())
  if name.startswith('raw'):
   wave=np.load(OUT/'waveforms_mV.npy',mmap_mode='r');leads=[1] if name.endswith('II') else list(range(12));tr=np.flatnonzero(f.partition.eq('train').to_numpy()&a['completed']);xx=np.asarray(wave[tr][:,leads,:]);checks[name+'_mean_train_only']=bool(np.allclose(xx.mean(axis=(0,2)),cp['mean'].numpy().reshape(-1),atol=2e-6));checks[name+'_std_train_only']=bool(np.allclose(xx.std(axis=(0,2)),cp['std'].numpy().reshape(-1),atol=2e-6))
 save(OUT/'verification.json',{k:bool(v) for k,v in checks.items()});assert all(checks.values()),checks

def torch_load(p):
 import torch
 return torch.load(p,weights_only=True)
def locked_threshold_native(y,s):
 from protocol import locked_threshold
 return locked_threshold(y,np.asarray(s,dtype=np.float32))

def fmt(value):return 'undefined' if value is None or not np.isfinite(value) else f'{value:.4f}'

def report(f,proto,summary,clin):
 support=pd.read_csv(OUT/'label_support.csv');stats=json.loads((OUT/'paired_statistics.json').read_text());receipts={n:json.loads((OUT/n/'receipt.json').read_text()) for n in NAMES};n=f.partition.value_counts().to_dict()
 lines=['# ZZU-ZUA 15-label first comparison','',f"Broad cohort:20113 records, partitions {n}. Source Normal=allzero; all15 independent labels retained. Primary macro labels: {len(proto['macro_indices'])}, fixed using training/validation support. Exact supports are in label_support.csv.",'',f"Previously examined binary test overlap: {proto['prior_test_overlap']} records. Patient independence is unverified. Partitions were preserved for all source records; evaluation is exploratory and conditional on record groups, not independent patient validation.",'','| Model | Completed test | Macro AUPRC completed | Macro AUPRC common | Severe group detected/positive | Specificity completed |','|---|---:|---:|---:|---:|---:|']
 for r in summary:lines.append(f"| {r['model']} | {r['completed_n']} | {fmt(r['macro_AUPRC_completed'])} | {fmt(r['macro_AUPRC_common_completed'])} | {r['group_detected_n']}/{r['group_positive_n']} | {r['group_specificity_completed']:.4f} |")
 lines+=['','## Exact label support','','| Source label | All released | Eligible10s | Train | Validation | Test | Macro included |','|---|---:|---:|---:|---:|---:|---|']
 for _,r in support.iterrows():lines.append(f"| {r['label']} | {r['source_all_n']} | {r['eligible_10s_n']} | {r['train']} | {r['validation']} | {r['test']} | {r['primary_macro_included']} |")
 lines+=['','## Interpretation and limits','', 'Compare primary macroAUPRC on identical common-completed records; model-specific completed metrics use different populations. Indeterminate positive records count as nondetections for clinical sensitivity. Indeterminate controls never count as true negatives. Failure rates by label and lead configuration are reported separately. Do not infer real-world PPV from the enriched source cohort.', '', 'All four models use standard unweighted multilabel BCE. Native full12 and nativeII are separately trained, with training-only input-specific preprocessing. Raw uses lr.001 BF16; authentic SetOperator uses lr.0001 FP32, fixed before new-task training because the previous development fit was unstable and BF16 quantized scores. These are distinct optimization recipes, not an isolated architecture causal effect. A single seed does not quantify training variance.', '', 'The source15label vocabulary and all274multilabel records with33combinations are retained. Legacy partitions were originally stratified around the binary task, not optimized for multilabel coverage; we did not move previously exposed records to improve rare-label support. Native full12-versusII comparisons involve separately trained models and fits; they do not establish robustness of a single flexible model to missing leads.', '', 'No individual validation claim is possible for pre-excited AF (zero validation/test), longRR(zero test), or tiny other subtype counts. Undefined ranking metrics remain null. Extremely rare labels have no unstable threshold fitting. Source labels apply to whole recordings and do not verify presence in the first10seconds.', '', 'Original operator architecture, PhaseCNN, Transformer and learned-query pooling are imported unchanged with15output head; reconstruction decoder is unused. Full12 uses actual measured channels and nominal normalized coefficients; limb-lead linear relations are imperfect in this release. Peak detection sees only observed channels. Original acceptedRR300-2000ms discards extreme rhythms; no fabricated beats or regular-grid fallback is used.', '', '## Published-reference status', '', 'The released script omits infarction from its merge rule, drops multilabel records, merges the remaining diagnoses to two labels, uses10second windows with overlapping tail segments and random window splitting. Its task cannot reproduce the stated full published task. Paper descriptions of wavelet/double-median denoising and5second windows disagree with released code and lack parameters. The released preprocessing was previously executed separately: /data/mithunmanivannan/codex_artifacts/zzu_author_preprocessing_20261120/AUTHOR_PREPROCESSING_REFERENCE.md. No paper-matched performance reproduction is claimed. Corrected15label results are not compared numerically to the paper’s multiclass AUROC.', '', '## Resource use','']
 for name,r in receipts.items():lines.append(f"- {name}: {r['epochs']}epochs, {r['seconds']:.1f}s load/train/evaluate/replay, peak {r['peak_GPU_GiB']:.2f}GiB, {r['gradient_parameters']:,} gradient-bearing parameters.")
 lines+=['', 'Preprocessing costs are in cache_receipt.json and representation_receipt.json; predictions, independent logits, selected/final checkpoints, hashes and history are archived for each model. Each selected checkpoint was reloaded and replayed on the same heldout batch with zero logit difference.', '', '## Next gate', '', 'No variablelead grid, balanced-loss optimization, M2/M3/M5 or manuscript has been launched. Review per-class support, validation histories, coverage and paired uncertainty before selecting a next experiment. A weak development fit cannot establish lost temporal information; clinical mechanisms need direct evidence. The author-linkage request is drafted separately; no external contact or linkage data has been received.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
