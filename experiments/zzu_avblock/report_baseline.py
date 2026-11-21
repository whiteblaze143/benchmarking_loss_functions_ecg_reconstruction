"""Render first-stage figures and reports from saved labels/predictions/checkpoint receipts."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve

R=Path(sys.argv[1]);f=pd.read_parquet(R/'split_manifest.parquet');pred=pd.read_parquet(R/'prediction_manifest.parquet');te=pred[pred.partition=='test'].copy();c=json.loads((R/'clinical_cohort_contract.json').read_text());receipt=json.loads((R/'baseline_receipt.json').read_text());stats=json.loads((R/'paired_statistics.json').read_text());point=pd.read_csv(R/'primary_results.csv').iloc[0];meta=pd.read_csv(R/'metadata_results.csv').iloc[0]
figdir=R/'figures';figdir.mkdir(exist_ok=True);tables=R/'figure_data';tables.mkdir(exist_ok=True)
# Population counts are computed rather than transcribed into the diagram.
whole=len(f);pos=int(f.label.sum());dur=f[f.duration_seconds>=30];primary=f[f.primary_eligible]
boxes=[f'ZZU-ZUA v3: {whole:,} records\n{pos:,} source-labelled AV-block positives',f'Actual duration >=30 seconds: {len(dur):,}\n{int(dur.label.sum()):,} positives',f'Common nested-prefix cohort: {len(primary):,}\n{int(primary.label.sum()):,} positives / {len(primary)-int(primary.label.sum()):,} controls\nAll 12 leads nonflat in actual first10s']
flow=[]
fig,ax=plt.subplots(figsize=(10,8));ax.axis('off')
for y,text in zip([.92,.72,.50],boxes):ax.text(.5,y,text,ha='center',va='center',bbox={'boxstyle':'round,pad=.7','facecolor':'#e8eff4','edgecolor':'#46718a'},fontsize=11)
for a,b in [(.86,.79),(.66,.58)]:ax.annotate('',(.5,b),(.5,a),arrowprops={'arrowstyle':'->'})
ax.text(.95,.82,f'Too short: {whole-len(dur):,}\nPositives: {pos-int(dur.label.sum()):,}',ha='right',fontsize=9)
ax.text(.95,.61,f'Prefix technical exclusions: {len(dur)-len(primary):,}\nPositives: {int(dur.label.sum()-primary.label.sum()):,}',ha='right',fontsize=9)
for x,(partition,g) in zip([.18,.5,.82],[(name,primary[primary.partition==name]) for name in ['train','validation','test']]):
 text=f'{partition}\n{len(g):,} records\n{int(g.label.sum()):,} positives';ax.text(x,.25,text,ha='center',bbox={'boxstyle':'round,pad=.5','facecolor':'white','edgecolor':'#46718a'})
 ax.annotate('',(x,.33),(.5,.42),arrowprops={'arrowstyle':'->'});flow.append({'partition':partition,'records':len(g),'positives':int(g.label.sum()),'record_groups_NOT_PATIENTS':g.record_group.nunique()})
ax.text(.5,.07,'Conservative record groups; residual patient-overlap risk.\nFirst baseline: full12, first10s, 500Hz; no verified crop-level event times.',ha='center',fontsize=10)
fig.savefig(figdir/'cohort_flowchart.pdf',bbox_inches='tight');plt.close(fig);pd.DataFrame(flow).to_csv(tables/'cohort_flow.csv',index=False)
# Discrimination and calibration are descriptive for the locked selected cohort.
fpr,tpr,threshold=roc_curve(te.label,te.score);precision,recall,_=precision_recall_curve(te.label,te.score)
pd.DataFrame({'false_positive_rate':fpr,'sensitivity':tpr,'threshold':threshold}).to_csv(tables/'roc.csv',index=False);pd.DataFrame({'precision':precision,'recall':recall}).to_csv(tables/'pr.csv',index=False)
cal=pd.read_csv(R/'calibration_results.csv');fig,axs=plt.subplots(1,3,figsize=(14,4))
axs[0].plot(fpr,tpr);axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set(xlabel='False-positive fraction',ylabel='Sensitivity',title=f'M0 AUROC {point.auroc:.4f}')
axs[1].plot(recall,precision);axs[1].axhline(te.label.mean(),linestyle='--',color='gray');axs[1].set(xlabel='Recall',ylabel='Precision',title=f'M0 average precision {point.auprc:.4f}')
axs[2].plot(cal.mean_probability,cal.observed_fraction,'o-');axs[2].plot([0,1],[0,1],'--',color='gray');axs[2].set(xlabel='Mean predicted probability',ylabel='Observed source-label fraction',title=f'Raw-score calibration; Brier {point.brier:.4f}')
fig.suptitle('First baseline only; record-group conditional evidence');fig.tight_layout();fig.savefig(figdir/'baseline_discrimination_calibration.pdf');plt.close(fig)
# ECG examples are unannotated source waveforms; no invented P/QRS timing truth.
false_neg=te[(te.label==1)&~te.decision].sort_values('score').head(3)
false_pos=te[(te.label==0)&te.decision].sort_values('score',ascending=False).head(3)
true_pos=te[(te.label==1)&te.decision].sort_values('score',ascending=False).head(1)
true_neg=te[(te.label==0)&~te.decision].sort_values('score').head(1)
examples=pd.concat([false_neg,false_pos,true_pos,true_neg]).merge(f[['record_uid','cache_index','diagnosis_raw']],on='record_uid',validate='one_to_one')
examples.to_csv(tables/'failure_examples.csv',index=False);x=np.load(R/'primary_30s_mV.npy',mmap_mode='r');fig,axs=plt.subplots(len(examples),1,figsize=(13,2.2*len(examples)),squeeze=False)
for ax,row in zip(axs[:,0],examples.itertuples()):
    trace=x[int(row.cache_index),1,:5000];kind='TP' if row.label and row.decision else 'FN' if row.label else 'FP' if row.decision else 'TN'
    ax.plot(np.arange(5000)/500,trace,lw=.7,color='#a03e35' if kind in ['FN','FP'] else '#376c88');ax.set(xlim=(0,10),ylabel='Lead II (mV)',title=f'{kind}: {row.record_uid} | record label={row.label} | score={row.score:.4f}, threshold={row.threshold:.4f}')
axs[-1,0].set_xlabel('Seconds from original record start');fig.suptitle('Computational failure review; record diagnosis, no independently annotated event timing',y=1);fig.tight_layout();fig.savefig(figdir/'critical_ecg_failure_examples.pdf',bbox_inches='tight');plt.close(fig)
# Inclusive diagnosis-specific controls: retain overlaps, expose denominator.
z=te.merge(f[['record_uid','labels_json']],on='record_uid',validate='one_to_one');hard=[]
for label in sorted({v for ls in z.labels_json.map(json.loads) for v in ls}):
    g=z[z.labels_json.map(lambda s:label in json.loads(s))];ng=g[g.label==0]
    hard.append({'source_diagnosis':label,'records':len(g),'negative_records':len(ng),'positive_calls_on_negative_records':int(ng.decision.sum()),'false_positive_fraction_controls':ng.decision.mean() if len(ng) else np.nan})
pd.DataFrame(hard).to_csv(R/'diagnosis_specific_errors.csv',index=False)
tp=int(((te.label==1)&te.decision).sum());fn=int(((te.label==1)&~te.decision).sum());fp=int(((te.label==0)&te.decision).sum());tn=int(((te.label==0)&~te.decision).sum());stability=max(v['max_endpoint_change_vs_first1000'] for v in stats['M0_intervals'].values())
report=f'''# ZZU-ZUA AV-block: first baseline and framework report

This is the requested first-baseline checkpoint before expansion. It is not a
completed SetOperator comparison or a submission-ready HRS result.

## Locked population

All {whole:,} cohort-qualified source records are preserved; {pos:,} contain the
inclusive high-grade/third-degree AV-block label. The primary paired cohort has
{len(primary):,} records, {int(primary.label.sum()):,} positives, and {len(primary)-int(primary.label.sum()):,}
controls. Duration is >=30 actual seconds; all 12 leads must vary in the first10s
prefix so all eight configurations and nested 10/20/30s views share the same source
records. Configuration-specific secondary support remains separately available.
Labels are source record diagnoses, not verified local crop diagnoses.

Split counts: {json.dumps(flow)}.

Groups preserve normal source-ID strings conservatively. The fixed-grid approximate
near-duplicate search considered {c['grouping_search']['windows']:,} five-second
blocks and {c['grouping_search']['lsh_candidates']:,} candidates, with no additional
cross-record match above the source-locked similarity gates. This does not rule
out arbitrary-offset overlaps, near-duplicates or repeated patients. No reliable
critical-cohort patient mapping or official fold manifest was found. The draft
investigator linkage request is saved and unsent. Patient independence is unverified.

## Actual crop audit amendment

The provisional whole-record-flatness rule missed 15 flat record-leads in the
first10s of ten otherwise nonflat records. The initial development run was stopped
at epoch26, before the post-training history/checkpoint/threshold/test phase.
Epoch26 is a zero-based index (27 completed epochs); an erratum corrects the frozen amendment wording without changing its hashes. Its checkpoint, log and old manifests are preserved under
`aborted_before_test_crop_qc/`. Control-flow evidence is saved separately.
The corrected cohort removed ten records (two positives), kept every partition
assignment and group unchanged, and restarted fresh weights with unchanged
architecture, optimizer, seed and training budget. No held-out result informed
that technical correction. Configuration eligibility now uses raw-confirmed actual
prefix flatness; a fingerprint is used only as a necessary candidate condition,
never as the final raw-signal decision.

## M0 implementation and selection

ECG-only ResNet18-style 1D residual model, {receipt['active_parameters']:,} active
parameters, trained from scratch on actual full12 first10s at native500Hz.
Physical uV are converted to mV; per-lead mean/population SD is fitted only to
training prefixes. There is no patient metadata input, hidden long-record content,
R-peak oracle, temporal padding or label-dependent preprocessing. Unweighted BCE,
AdamW lr0.001/weight_decay0.0001, batch128, maximum40 epochs, validation-AUROC
selection and patience8 were locked before training. Best weights are copied to
CPU on strict improvement, then reloaded for validation threshold selection and
one held-out pass. Seed42; cuDNN autotuning and BF16 are recorded, so bitwise
retraining identity is not asserted.

Threshold {receipt['threshold']:.10f} was selected on validation controls to achieve
at least95% specificity while preserving score ties conservatively. This is a
common target operating point for model comparisons, not a guarantee that held-out
specificity remains95%. Future models require their own development-selected
threshold under the same target; no test tuning is allowed.

Checkpoint SHA256: `{receipt['checkpoint_sha256']}`.

## First baseline result

Held-out n={len(te):,}, positives={int(te.label.sum())}, controls={int((te.label==0).sum())}.
TP={tp}, FN={fn}, FP={fp}, TN={tn}; all inputs completed with no abstentions.

- Sensitivity: {point.sensitivity_all_eligible:.6f} (95% conditional group-bootstrap
  interval {point.sensitivity_all_eligible_ci_low:.6f}–{point.sensitivity_all_eligible_ci_high:.6f}).
- Specificity: {point.specificity_completed:.6f} (interval
  {point.specificity_completed_ci_low:.6f}–{point.specificity_completed_ci_high:.6f}).
- AUROC: {point.auroc:.6f}; AUPRC (average precision): {point.auprc:.6f}.
- Brier score: {point.brier:.6f}; this is calibration in a selected case/control cohort.

Normal controls: 0/271 positive calls. Other rhythm-critical controls: 23/97.
Other critical controls:70/1307. The elevated rhythm-critical false-positive
fraction is a more demanding failure stratum than comparison against normal ECGs.
These counts are generated from the saved prediction ledger, with inclusive
per-diagnosis breakdowns in `diagnosis_specific_errors.csv`.

A fixed metadata-only logistic control (age, sex, original duration and source
cohort; training-only transformation/fit; same validation specificity rule) has
AUROC {meta.auroc:.6f}, sensitivity {meta.sensitivity_all_eligible:.6f}, specificity
{meta.specificity_completed:.6f}. Its weaker result does not prove absence of all
acquisition or source-label shortcuts. Flatness is constant by primary eligibility;
its effects require the explicitly defined secondary populations.

All intervals use 2,000 record-group bootstrap draws, conditional on the trained
checkpoint and locked threshold. They do not incorporate unknown patient overlap,
annotation error or model-training variability. First1000 versus2000 endpoint
changes are saved for every metric; maximum absolute change is {stability:.6f}.
No formal SetOperator paired test or superiority p-value exists yet.

## Resources and verification

A100-PCIE-40GB, shared Python environment reused with task-local cuDNN9.3.
{receipt['epochs_run']} epochs ran, best epoch index {receipt['best_epoch']}; actual training
{receipt['training_seconds']:.2f}s, load/train/predict/evaluate stage
{receipt['total_load_train_predict_seconds']:.2f}s. Peak allocated GPU memory
{receipt['gpu_peak_allocated_GiB']:.2f}GiB. Amortized forward latency
{receipt['inference_latency_ms_per_record_amortized_batch128']:.4f}ms/record at batch128,
with warmed GPU execution, excluding I/O and preprocessing; this is not streaming
single-record latency. Cold NFS reads, grouping and crop QC dominate preparation.

Five protocol tests and two fresh documented GPU preflight passes succeeded.
The tests cover tied-score threshold conservatism, failed-positive denominators,
known scaled-copy retrieval, exact paired-resampling identity/alignment and the
locked lead-set definitions. The kernel witness exercised forward/backward.

A comparable raw model is about one minute of warm-cache execution on this A100;
that estimate is measured for this one configuration, not promised for 30s inputs
or Phase-KME fitting. A full experiment ETA is not established until original
M1 preprocessing and training are profiled. Artifacts currently occupy roughly
10GB, mainly native-waveform and grouping caches. No foundation pretraining,
reconstruction training or broad grid was launched.

## Remaining study work

M1 original Phase-KME/OperatorSetModel remains untrained for this endpoint. Its
existing EchoNext convenience extractor uses synthetic interval fallbacks and
must be adapted to report actual peak failures explicitly, with a training-only
ZZU representation fit and observed-channel-only peak detection. M2/M3, paired
lead/duration comparisons and physiological perturbations are not executed.
No SetOperator superiority, temporal-information advantage or clinical deployment
claim follows from this baseline. Full12 performance is an exploratory reference;
the formal first contrast remains M1 versus M0 on LeadII/10s, locked in the contract.

External search identified PTB-XL3AVB as a narrower complete-block candidate:
16 records from15 source patients, with15 source likelihood100 and one50. It has
not been evaluated/admitted as same-endpoint external confirmation, and absence
of3AVB is not proof of absence of the broader high-grade endpoint. No first-degree
AV-block or future-risk endpoint is substituted. The public sources are recorded
in `external_candidate_summary.json`.

Delivered now: clinical contract; all-record split; crop/config eligibility and
endpoint counts; trained M0 checkpoint; validation/test prediction ledger with
hashes; primary/subgroup/calibration tables; conditional resampling statistics;
metadata control; cohort flowchart; ROC/PR/calibration and unannotated failure
waveforms. The full lead-duration heatmap and paired model-comparison figure are
pending actual model/configuration runs and are not fabricated from empty cells.

Reproduce prediction-only evaluation using `evaluate.py --output ROOT`, and
figures/report using `report_baseline.py ROOT`. Re-training refuses to overwrite
held-out predictions. Original data and the interrupted-run history are retained.
'''
(R/'reproducibility_report.md').write_text(report)
status=json.loads((R/'model_implementation_status.json').read_text());status['M0']['status']='first full12/10s baseline complete, checkpoint and held-out ledger verified';(R/'model_implementation_status.json').write_text(json.dumps(status,indent=2))
print('Rendered baseline figures/report',flush=True)
