"""Paired frozen-prediction analysis; no classifier fitting or threshold changes."""
import json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from leadii import OUT,BASE,sha,save,cohort
from protocol import metrics
from evaluate import paired_group_intervals

def main():
    f=cohort();ex=pd.read_parquet(OUT/'extraction.parquet');assert np.array_equal(f.record_uid,ex.record_uid)
    ledgers={name:pd.read_parquet(OUT/('M1_FP32' if name=='M1' else name)/'prediction_manifest.parquet') for name in ['M0','M1']}
    tests={name:l[l.partition=='test'].sort_values('record_uid').reset_index(drop=True) for name,l in ledgers.items()}
    a,b=tests['M0'],tests['M1'];assert len(a)==len(b)==1865 and np.array_equal(a.record_uid,b.record_uid)
    paired=a[['record_uid','record_group','label','negative_stratum','duration_seconds','cohort','label_count']].copy()
    for name,t in tests.items():
        for key in ['score','decision_status','threshold','decision']:paired[name+'_'+key]=t[key].to_numpy()
    paired=paired.merge(ex,on='record_uid',validate='one_to_one');assert len(paired)==1865
    # Link original source diagnoses and first-prefix quality without changing eligibility.
    source=pd.read_parquet('/data/mithunmanivannan/datasets/figshare_31229326_v3/analysis/record_manifest.parquet')
    diagnosis_cols=[c for c in source.columns if 'diagnos' in c or 'flat' in c or 'finite' in c]
    if 'record_uid' in source:paired=paired.merge(source[['record_uid']+diagnosis_cols],on='record_uid',how='left',validate='one_to_one')
    prefix=pd.read_parquet(BASE/'prefix_availability.parquet')
    # Preserve all source quality fields in a separate aligned table if its key is configuration-specific.
    paired=paired.merge(prefix,on='record_uid',validate='one_to_one')
    prefix[prefix.record_uid.isin(paired.record_uid)].to_parquet(OUT/'heldout_prefix_quality.parquet',index=False)
    paired.to_parquet(OUT/'paired_predictions.parquet',index=False);paired.to_csv(OUT/'paired_predictions.csv',index=False)
    results=[];sub=[];cal=[]
    for name,t in tests.items():
        m=metrics(t.label,t.score,t.decision_status,float(t.threshold.iloc[0]));ci=paired_group_intervals(t)
        results.append({'model':name,'threshold':float(t.threshold.iloc[0]),**m,**{k+'_ci_'+side:v[side] for k,v in ci.items() for side in ['low','high']}})
        for field in ['negative_stratum','cohort']:
            for value,g in t.groupby(field):
                sub.append({'model':name,'field':field,'value':value,'n':len(g),'positive_n':int(g.label.sum()),'positive_calls':int(g.decision.sum()),'indeterminate_n':int(g.decision_status.ne('completed').sum()),**metrics(g.label,g.score,g.decision_status,float(g.threshold.iloc[0]))})
        for lo,hi in zip(np.arange(0,1,.1),np.arange(.1,1.1,.1)):
            g=t[t.score.ge(lo)&(t.score.lt(hi) if hi<1 else t.score.le(hi))];cal.append({'model':name,'low':lo,'high':hi,'n':len(g),'probability_mean':g.score.mean(),'positive_fraction':g.label.mean()})
    pd.DataFrame(results).to_csv(OUT/'results.csv',index=False);pd.DataFrame(sub).to_csv(OUT/'subgroups.csv',index=False);pd.DataFrame(cal).to_csv(OUT/'calibration.csv',index=False)
    full=pd.read_parquet(BASE/'prediction_manifest.parquet');full=full[full.partition=='test'].set_index('record_uid').loc[a.record_uid].reset_index()
    assert np.array_equal(full.label,a.label)
    save(OUT/'full12_to_leadii_reference.json',{'M0_LII_minus_full12':paired_group_intervals(a,full),'interpretation':'paired selected cohort; both models trained natively on their available leads; independent training trajectories'})
    difference=paired_group_intervals(b,a)
    common=b.decision_status.eq('completed').to_numpy();common_diff=paired_group_intervals(b[common].reset_index(drop=True),a[common].reset_index(drop=True))
    # Sensitivity above includes all positive failures. Rankings above for all-eligible model tables are completed-only.
    save(OUT/'paired_statistics.json',{'M1_minus_M0':difference,'common_completed_M1_minus_M0':common_diff,'replicates':2000,'unit':'record group, not verified patient','interpretation':'conditional on fixed fitted models and development thresholds; ranking contrasts require common-completed subset','all_test_n':1865,'common_completed_n':int(common.sum()),'indeterminate_not_true_negative':True})
    ext=f[['record_uid','partition','label','negative_stratum']].merge(ex,on='record_uid',validate='one_to_one')
    rows=[]
    for (partition,label),g in ext.groupby(['partition','label']):
        rows.append({'partition':partition,'label':int(label),'n':len(g),'failures':int(g.decision_status.ne('completed').sum()),'median_detected_peaks':float(g.detected_peaks.median()),'median_valid_beats':float(g.valid_beats.median()),'median_represented_fraction':float(g.represented_fraction.median()),'median_inferred_rate_bpm':float(g.median_rate_bpm.median())})
    pd.DataFrame(rows).to_csv(OUT/'extraction_summary.csv',index=False)
    ext.to_parquet(OUT/'extraction_all_records_with_partitions.parquet',index=False)
    correct0=(paired.M0_decision==paired.label);correct1=paired.M1_decision_status.eq('completed')&(paired.M1_decision==paired.label)
    paired['error_category']=np.select([correct0&correct1,correct0&~correct1,~correct0&correct1],['both_correct','M0_correct_M1_incorrect','M1_correct_M0_incorrect'],default='both_incorrect')
    counts=paired.groupby(['label','error_category']).size().rename('n').reset_index();counts.to_csv(OUT/'disagreements.csv',index=False)
    # All failures, plus class-stratified deterministic examples from each disagreement category.
    selected=[]
    for _,g in paired.groupby(['label','error_category']):selected.extend(g.head(2).record_uid)
    selected.extend(paired[paired.M1_decision_status.ne('completed')].record_uid)
    selected=list(dict.fromkeys(selected));examples=paired.set_index('record_uid').loc[selected].reset_index();examples.to_csv(OUT/'waveform_example_ledger.csv',index=False)
    wave=np.load(OUT/'leadII_mV.npy',mmap_mode='r');index={uid:i for i,uid in enumerate(f.record_uid)};figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(figdir/'physiological_examples.pdf') as pdf:
        for start in range(0,len(examples),4):
            chunk=examples.iloc[start:start+4];fig,axes=plt.subplots(len(chunk),1,figsize=(14,2.7*len(chunk)),squeeze=False)
            for ax,(_,row) in zip(axes[:,0],chunk.iterrows()):
                sig=wave[index[row.record_uid]];ax.plot(np.arange(5000)/500,sig,lw=.6,color='black')
                peaks=np.asarray(row.rpeaks,dtype=int);ax.scatter(peaks/500,sig[peaks],s=14,color='red',label='XQRS detections (unadjudicated)')
                for left,right in row.valid_intervals:ax.axvspan(left/500,right/500,alpha=.08,color='blue')
                ax.set_title(f"{row.record_uid} | source AVB={row.label} | {row.error_category} | raw={row.M0_score:.3f}, set={row.M1_score:.3f} | peaks={row.detected_peaks}, valid RR={row.valid_beats}, represented={row.represented_fraction:.0%}",fontsize=9)
                ax.set(xlabel='Observed seconds',ylabel='Lead II (mV)',xlim=(0,10));ax.legend(loc='upper right',fontsize=7)
            fig.tight_layout();pdf.savefig(fig)
            if start<12:fig.savefig(figdir/f'examples_{start//4+1}.png',dpi=120)
            plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7));testext=ext[ext.partition=='test']
    for label,color in [(0,'steelblue'),(1,'darkorange')]:
        g=testext[testext.label==label]
        for ax,col,title in zip(axes.ravel(),['detected_peaks','valid_beats','represented_fraction','median_rate_bpm'],['Detected peaks','Valid RR beats','Recording fraction represented','Inferred median ventricular rate (unadjudicated)']):
            ax.hist(g[col].dropna(),bins=20,alpha=.5,density=True,label=f'AVB={label}',color=color);ax.set_title(title,fontsize=9);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(figdir/'extraction_distributions.pdf');fig.savefig(figdir/'extraction_distributions.png',dpi=140);plt.close(fig)
    verification={}
    proto=json.loads((OUT/'protocol.json').read_text())
    for name,digest in proto['reference_hashes'].items():verification['reference_unchanged_'+name]=sha(BASE/name)==digest
    verification.update(exact_test_alignment=bool(np.array_equal(a.record_uid,b.record_uid)),no_missing_test_records=len(paired)==1865,all_source_hashes_unchanged=all(sha(p)==h for p,h in proto['source_hashes'].items()),no_test_groups_in_training=not bool(set(f[f.partition=='train'].record_group)&set(f[f.partition=='test'].record_group)),no_validation_groups_in_training=not bool(set(f[f.partition=='train'].record_group)&set(f[f.partition=='validation'].record_group)))
    assert all(verification.values()),verification;save(OUT/'verification.json',verification)
    # Protocol-restricted interpretation, driven by actual saved outputs.
    r={d['model']:d for d in results};sens=r['M1']['sensitivity_all_eligible']-r['M0']['sensitivity_all_eligible']
    text=f'''# Lead-II matched comparison\n\nLocked selected cohort:9426 records (967 AVB); test1865 (190 AVB). Existing full12 artifacts were unchanged. Patients are not verified disjoint; this is a conditional record-group comparison, with the test previously examined for the full12 model. One seed per model.\n\n| Model | All-eligible sensitivity | Completed specificity | AUROC completed | AUPRC completed | Brier completed | Indeterminate |\n|---|---:|---:|---:|---:|---:|---:|\n'''
    for name in ['M0','M1']:
        v=r[name];text+=f"| {name} Lead II | {v['sensitivity_all_eligible']:.4f} | {v['specificity_completed']:.4f} | {v['auroc']:.4f} | {v['auprc']:.4f} | {v['brier']:.4f} | {v['failed_or_indeterminate_n']}/{v['n']} |\n"
    text+=f'''\nM1 minus M0 sensitivity difference: {sens:.6f}; paired95% record-group interval [{difference['sensitivity_all_eligible']['low']:.6f}, {difference['sensitivity_all_eligible']['high']:.6f}]. Thresholds were chosen separately on validation to target95% specificity. Indeterminate positives count as nondetections; indeterminate controls do not count as correct negatives. See paired_statistics.json for common-completed ranking contrasts and conditional uncertainty.\n\n## Interpretation\n\nThe first BF16 M1 run had constant outputs and is retained separately. Training-only diagnostics showed numerical quantization, so M1 here is the prospectively recorded FP32 technical correction with identical architecture and optimizer. Its test has already been examined in the earlier failed run; treat the corrected comparison as exploratory. No architectural or threshold tuning used test errors.\n\nOriginal SetOperator {'has lower all-eligible sensitivity than' if sens<0 else 'does not establish a robust advantage over'} the matched raw baseline in this run. This does not isolate representation loss from optimization failure. One seeded comparison cannot establish general model-family superiority or clinical benefit. Report false-positive strata in subgroups.csv alongside sensitivity. Full12 M0 remains a reference (189/190 sensitivity), not the matched single-lead comparator.\n\n## Physiological audit limits\n\nEvery record has saved actual XQRS detections, accepted RR intervals, valid beat count, retained fraction, RR list and inferred rate. Waveform examples include all extraction failures and class-stratified model disagreements; red markers are detector outputs, not clinician-confirmed QRS labels. Source diagnoses apply to full records, not necessarily the first10seconds. AV dissociation, absent atrial information and loss of order cannot be established causally from classification disagreement or these unadjudicated detections. Original RR filtering rejects intervals outside300–2000ms; that is a representation limitation rather than proof of a specific physiological mechanism. No atrial/P-wave annotations were invented.\n\n## Next experiment\n\nPrepare named-lead FLEX only: I,II,V1,I+II,II+V1,limb6,precordial6,full12. Sample a single uniform named configuration per training record each epoch, with an identical precomputed schedule for SetOperator and the masked12channel raw control. Raw control receives zero-filled standardized unavailable channels plus12explicit masks; SetOperator receives only actual observed lead signals and true canonical coefficients. Restrict detection to supplied channels; declare the deterministic multilead selection rule before fitting. Fit kernel/scaling only on training, with equal record/configuration sampling and the same frozen partitions. Train each model once, then evaluate all eight lead sets at10seconds. Duration extensions20/30seconds require separate declared training distribution and input support. No synthetic operators in the primary comparison.\n\nM3 is not launched. Timing loss remains a plausible hypothesis, not a proven failure mechanism. A temporal proposal would preserve original per-beat morphology encoding, chronological beat index and observed onset timestamp; unreliable intervals remain explicitly masked, and records with no usable beats remain indeterminate. Matched knockout must share morphology encoder, capacity and training budget but remove chronological order/timestamps. Active parameters and cost must be calculated from an actual proposed module before authorization.\n\n## Reproduction\n\nSee protocol.json, source hashes, evaluation_lock.json for each model, selected/final checkpoints, replay receipts, training histories, paired_predictions.parquet, kernel_audit.json and verification.json. Preprocessing matches the previous raw M0 comparison contract; this is not a replication of the authors' unspecified denoising.\n'''
    receipts={name:json.loads((OUT/('M1_FP32' if name=='M1' else name)/'receipt.json').read_text()) for name in ['M0','M1']}
    text+='\nResource observations: '+ '; '.join(f"{name}: {v['seconds']:.1f}s load/train/predict/replay, {v['epochs_run']}epochs, {v['parameters_with_gradient']:,} parameters with gradients, {v['peak_gpu_GiB']:.2f}GiB peak" for name,v in receipts.items())+'.\n'
    text+='\nFLEX planning estimate: two training runs, not a grid. Budget up to12x measured combined Lead-II runtime for larger response sets and masked raw input, plus full multi-lead feature extraction; estimate is provisional until a training-only throughput witness. Kernel response cache at9426x12x16x128float32 is about0.93GB per duration. Physical waveform cache can be reused. Feature extraction and detection require a separate benchmark; do not give a firm completion ETA from model-only time.\n'
    (OUT/'REPORT.md').write_text(text)
    save(OUT/'delivery_manifest.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='delivery_manifest.json'})
    print(pd.DataFrame(results).to_string(index=False));print('REPORT',OUT/'REPORT.md',flush=True)
if __name__=='__main__':main()
