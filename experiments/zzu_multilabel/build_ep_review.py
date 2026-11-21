"""Blinded two-pass EP pilot and separate coordinator atlas; no model/label changes."""
import json,hashlib,csv
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from study import OUT,LEADS,TACHY,frame
DEST=OUT/'ep_review_pilot'
GROUPS=[('A','Both agree with source group'),('B','Raw agrees; Set disagrees'),('C','Set agrees; Raw disagrees'),('D','Both disagree with source group'),('E','Set representation unavailable'),('F','False alert among other critical records')]
COLORS=['#27744c','#1768ac','#ac5b17','#9b2745','#7351a8','#69727b']

def limits(sig):
 lo=float(np.min(sig));hi=float(np.max(sig));pad=max(.05,(hi-lo)*.12);return lo-pad,hi+pad

def trace(ax,sig,start=0,stop=5000,peaks=None,intervals=None):
 tt=np.arange(start,stop)/500;ax.plot(tt,sig[start:stop],lw=.65,color='#172b3a',rasterized=False);ax.set_xlim(start/500,stop/500);ax.set_ylim(*limits(sig[start:stop]));ax.set_ylabel('mV',fontsize=9);ax.set_xlabel('Seconds',fontsize=9)
 ax.set_xticks(np.arange(start/500,stop/500+.001,1));ax.set_xticks(np.arange(start/500,stop/500+.001,.2),minor=True);ax.grid(which='major',color='#c7cfd6',lw=.55);ax.grid(which='minor',color='#e6ebef',lw=.35);ax.tick_params(labelsize=8)
 if peaks is not None:
  ids=np.asarray(peaks,dtype=int);ids=ids[(ids>=start)&(ids<stop)];ax.scatter(ids/500,sig[ids],s=18,color='#be3144',zorder=3,label='XQRS detections (not adjudicated)')
 if intervals is not None:
  for left,right in intervals:ax.axvspan(left/500,right/500,color='#3781b8',alpha=.12)

def make_overview(table,wave):
 fig,axes=plt.subplots(3,2,figsize=(16,12));fig.subplots_adjust(top=.84,bottom=.07,left=.065,right=.98,hspace=.85,wspace=.22)
 short={'Ventricular Fibrillation':'VF','Ventricular Flutter':'ventricular flutter','Torsade de Pointes':'torsades','Polymorphic Ventricular Tachycardia':'polymorphic VT','Bidirectional Ventricular Tachycardia':'bidirectional VT','Atrial Fibrillation with Pre-excitation':'pre-excited AF','High-Grade/Third-Degree Atrioventricular Block':'high-grade/third-degree AV block','R-on-T Ventricular Premature Contractions':'R-on-T PVC','Acute Myocardial Infarction':'MI','Acute Myocardial Ischemia':'ischemia'}
 for ax,(letter,title),color in zip(axes.ravel(),GROUPS,COLORS):
  candidates=table[table.category==letter]
  row=candidates[candidates.source_severe_group_positive].iloc[0] if letter!='F' else candidates.iloc[0]
  labels=', '.join(short.get(l,l) for l in json.loads(row.source_labels));trace(ax,wave[int(row.cache_row),1]);truth='positive' if row.source_severe_group_positive else 'negative'
  ax.set_title(f'{letter}. {title}\n{row.case_id} | Source severe group: {truth}\nRaw: {row.raw_decision}; Set: {row.set_decision}\nSource record label: {labels}',loc='left',fontsize=10,color=color,fontweight='bold')
 fig.suptitle('Computational review categories — coordinator overview',fontsize=18,fontweight='bold',y=.98)
 fig.text(.065,.935,'Endpoint: source severe tachyarrhythmia, not every critical finding. Source-negative does not mean clinically normal.',fontsize=11)
 fig.text(.065,.015,'One illustrative case per category. Source record labels are not adjudicated10-second event labels. Do not share this unblinded overview with the reviewer initially.',fontsize=10)
 fig.savefig(DEST/'FAILURE_OVERVIEW_UNBLINDED.pdf');fig.savefig(DEST/'FAILURE_OVERVIEW_UNBLINDED.png',dpi=130);plt.close(fig)

def main():
 if DEST.exists():raise RuntimeError('Pilot already exists; preserve selection and blinding key')
 blind=DEST/'reviewer_blinded';keydir=DEST/'coordinator_only';blind.mkdir(parents=True);keydir.mkdir()
 f=frame();test=f.partition.eq('test').to_numpy();ft=f[test].reset_index(drop=True);y=np.load(OUT/'targets.npy')[test];group=y[:,TACHY].any(axis=1);scores={};valid={};decisions={};thresholds={}
 for name in ['raw_II','set_II']:
  a=np.load(OUT/name/'predictions.npz');valid[name]=a['completed'][test];scores[name]=np.nanmax(a['probabilities'][test][:,TACHY],axis=1,initial=-np.inf);thresholds[name]=json.loads((OUT/name/'evaluation_lock.json').read_text())['severe_group_threshold'];decisions[name]=valid[name]&(scores[name]>=thresholds[name])
 raw=decisions['raw_II'];setd=decisions['set_II'];both=valid['raw_II']&valid['set_II'];c0=raw==group;c1=setd==group
 masks=[both&c0&c1,both&c0&~c1,both&~c0&c1,both&~c0&~c1,~valid['set_II'],both&~group&(raw|setd)&ft.cohort.eq('critical').to_numpy()]
 rng=np.random.default_rng(20261120);used=set();chosen=[];flow=[]
 for (letter,title),mask in zip(GROUPS,masks):
  requested=[(True,2),(False,2)] if letter!='F' else [(False,4)]
  for truth,n in requested:
   available=np.array([i for i in np.flatnonzero(mask&(group==truth)) if int(i) not in used],dtype=int)
   if len(available)<n:raise RuntimeError(f'Insufficient disjoint cases in {letter}, sourcegroup={truth}: {len(available)}<{n}')
   picked=rng.choice(available,n,replace=False)
   for i in picked:chosen.append((int(i),letter,title));used.add(int(i))
   flow.append({'category':letter,'definition':title,'source_group_positive':truth,'available_before_selection':len(available),'selected':n})
 rng.shuffle(chosen);ex=pd.read_parquet(OUT/'extraction_II.parquet').set_index('record_uid');wave=np.load(OUT/'waveforms_mV.npy',mmap_mode='r');lookup={u:i for i,u in enumerate(f.record_uid)};rows=[]
 for j,(i,letter,title) in enumerate(chosen):
  row=ft.iloc[i];uid=row.record_uid;rows.append({'case_id':f'EP-{j+1:03d}','category':letter,'computational_category':title,'record_uid':uid,'record_group':row.record_group,'source_labels':row.labels_json,'source_severe_group_positive':bool(group[i]),'source_duration_seconds':row.duration_seconds,'raw_score':float(scores['raw_II'][i]),'raw_threshold':thresholds['raw_II'],'raw_decision':'positive' if raw[i] else 'negative','set_score':float(scores['set_II'][i]) if valid['set_II'][i] else None,'set_threshold':thresholds['set_II'],'set_decision':('positive' if setd[i] else 'negative') if valid['set_II'][i] else 'indeterminate','detected_peaks':int(ex.loc[uid].detected_peaks),'accepted_RR_intervals':int(ex.loc[uid].valid_beats),'represented_fraction':float(ex.loc[uid].represented_fraction),'cache_row':lookup[uid]})
 table=pd.DataFrame(rows);assert len(table)==24 and table.record_uid.is_unique;table.to_csv(keydir/'case_key_DO_NOT_SHARE.csv',index=False);pd.DataFrame(flow).to_csv(keydir/'selection_flow.csv',index=False)
 # Reviewer pass A: no diagnoses, model outputs, detector markings or computational categories.
 with PdfPages(blind/'PASS_A_LeadII_blinded.pdf') as pdf:
  for _,row in table.iterrows():
   sig=wave[int(row.cache_row),1];fig=plt.figure(figsize=(11.69,8.27));grid=fig.add_gridspec(2,2,left=.07,right=.97,top=.87,bottom=.13,hspace=.42,wspace=.22);ax=fig.add_subplot(grid[0,:]);trace(ax,sig);ax.set_title('Lead II — full 10-second segment',loc='left',fontsize=10)
   for ax,start,stop in [(fig.add_subplot(grid[1,0]),0,2500),(fig.add_subplot(grid[1,1]),2500,5000)]:trace(ax,sig,start,stop);ax.set_title(f'Lead II — {start/500:g}–{stop/500:g} s detail',loc='left',fontsize=10)
   fig.suptitle(f"{row.case_id}   |   PASS A: Lead II",x=.07,ha='left',fontsize=17,fontweight='bold');fig.text(.07,.045,'Public dataset • native 500 Hz • physical mV • no filtering or model normalization\nPanels use independently labeled vertical scales. Record interpretation before opening Pass B.',fontsize=9,color='#465563');pdf.savefig(fig)
   if row.case_id=='EP-001':fig.savefig(DEST/'preview_blinded_pass_A.png',dpi=120)
   plt.close(fig)
 # Reviewer pass B: same input time interval, additional actual named leads, still no labels.
 with PdfPages(blind/'PASS_B_12lead_blinded.pdf') as pdf:
  for _,row in table.iterrows():
   sig=wave[int(row.cache_row)];fig,axes=plt.subplots(12,1,figsize=(16.54,11.69),sharex=True);fig.subplots_adjust(left=.075,right=.97,top=.90,bottom=.075,hspace=.16)
   for k,ax in enumerate(axes):
    trace(ax,sig[k]);ax.set_ylabel(LEADS[k]+'\nmV',fontsize=8,rotation=0,labelpad=22);ax.set_xlabel('');ax.tick_params(labelsize=6);ax.set_yticks(np.linspace(*ax.get_ylim(),3));ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2g'))
   axes[-1].set_xlabel('Observed seconds');fig.suptitle(f"{row.case_id}   |   PASS B: all 12 leads, same 10 seconds",x=.075,ha='left',fontsize=17,fontweight='bold');fig.text(.075,.025,'Native 500 Hz; physical mV; independent labeled vertical scales. No source diagnoses or model outputs are shown.',fontsize=9);pdf.savefig(fig)
   if row.case_id=='EP-001':fig.savefig(DEST/'preview_blinded_pass_B.png',dpi=110)
   plt.close(fig)
 # Coordinator atlas: one case per page, plain waveform and algorithm overlay kept distinct.
 with PdfPages(keydir/'TECHNICAL_ATLAS_UNBLINDED.pdf') as pdf:
  for _,row in table.sort_values(['category','case_id']).iterrows():
   color=COLORS[ord(row.category)-ord('A')];sig=wave[int(row.cache_row),1];info=ex.loc[row.record_uid];fig,axes=plt.subplots(2,1,figsize=(11.69,8.27));fig.subplots_adjust(left=.075,right=.97,top=.68,bottom=.12,hspace=.4)
   trace(axes[0],sig);axes[0].set_title('1. Observed Lead II — no algorithm markings',loc='left',fontsize=10);trace(axes[1],sig,peaks=info.rpeaks,intervals=info.valid_intervals);axes[1].set_title('2. Detector overlay — red detections; blue accepted RR intervals',loc='left',fontsize=10)
   fig.suptitle(f"{row.category}  |  {row.computational_category}\n{row.case_id}  ·  {row.record_uid}",x=.075,y=.96,ha='left',fontsize=15,fontweight='bold',color=color)
   source='positive' if row.source_severe_group_positive else 'negative';setscore='no score' if pd.isna(row.set_score) else f'{row.set_score:.5f}'
   fig.text(.075,.79,f"SOURCE RECORD LABELS: {', '.join(json.loads(row.source_labels))}\nSevere-tachyarrhythmia source group: {source} (not a crop-level adjudication)\nRaw: {row.raw_decision}, score {row.raw_score:.5f}, threshold {row.raw_threshold:.5f}   |   Set: {row.set_decision}, score {setscore}, threshold {row.set_threshold:.5f}\nDetector: {row.detected_peaks} peaks; {row.accepted_RR_intervals} accepted RR intervals; {row.represented_fraction:.0%} of time retained",fontsize=9,linespacing=1.5)
   fig.text(.075,.025,'COORDINATOR ONLY — do not share before blinded ratings are locked. Source agreement is not clinical correctness.\nRed markers are unadjudicated detections. Accepted RR rule: 300–2,000 ms; at least two intervals required.',fontsize=9,color='#8b233a');pdf.savefig(fig);plt.close(fig)
 # Compact category overview, deliberately showing positive-source failures and a hard negative.
 make_overview(table,wave)
 # Empty, neutral response sheet has no labels or category/selection clues.
 columns=['reviewer_id','case_id','pass','interpretable_yes_partial_no','visible_rhythm_free_text','other_visible_ECG_findings','urgent_finding_yes_no_uncertain','interpretive_difficulty_1_easy_5_very_difficult','confidence_1_low_5_high','limiting_factors_free_text','additional_information_needed','changed_after_12lead_yes_no','comments']
 with (blind/'RATINGS_blank.csv').open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
  for stage in ['A_LeadII','B_12lead']:
   for cid in table.case_id:writer.writerow({'case_id':cid,'pass':stage})
 (blind/'START_HERE.md').write_text('''# Blinded EP review pilot\n\nPlease review the24 deidentified public ECG segments in Pass A first. Complete and lock those ratings before opening Pass B, which shows the same time interval with12 leads. Record what is visible, confidence and interpretive difficulty; insufficient information is a valid response. Do not attempt to infer whole-record clinical diagnoses from context that is not shown. There are no source labels, model outputs or detector markings in these reviewer files.\n\nUse RATINGS_blank.csv. Difficulty:1=straightforward,2=slightly difficult,3=moderate,4=difficult,5=very difficult. Confidence:1=very low through5=very high. In Pass B, also record whether extra leads change your interpretation. Clinical judgment, not agreement with any algorithm, is requested.\n\nThese are deliberately selected pilot segments, not a prevalence-representative sample. Reviewers may stop and suggest a different protocol or case count. A second reviewer should rate independently before discussing disagreements. Two-pass review has recall/order effects and is exploratory.\n\nNo new clinical validation claim will be made from this pilot alone.\n''')
 (DEST/'CHRIS_MESSAGE_DRAFT.md').write_text('''Subject: Advice on an EP adjudication pilot for our public-ECG study\n\nHi Chris,\n\nCould I get your advice on where the clinically valuable contribution should be in our public-ECG study? We are comparing raw-waveform and phase-based models under restricted lead availability. The computational disagreements could reflect difficult ECG interpretation, inadequate lead information, detector failures, or a mismatch between whole-record labels and the particular10-second segment. Model errors alone do not distinguish those explanations.\n\nWould you or another EP colleague be willing to review a small blinded pilot—perhaps12–24 challenging segments—to help us determine which explanations matter clinically? I have prepared24 cases: Lead II first, followed by the same segments with12 leads. Source diagnoses, model outputs and computational error categories are withheld from the reviewer. The form asks for visible findings, interpretability, difficulty and confidence, including whether extra leads change the interpretation.\n\nYour advice on the case mix, useful adjudication questions and appropriate clinical scope would be especially valuable before we expand the modeling. If feasible, two independent readers could also show whether apparent difficulty is shared or reader-dependent. This would be exploratory adjudication, not a claim of clinical validation.\n\nWould that be feasible, and what pilot size would you recommend? I can send just the blinded packet for an initial look.\n\nThanks!\n\nStatus: draft only; NOT SENT. No email-sending connector is available in this session.\n''')
 (DEST/'README.md').write_text('''# Organized failure figures and EP pilot\n\n1. Start with FAILURE_OVERVIEW_UNBLINDED.pdf: six clearly named computational categories. This is for the research team.\n2. TECHNICAL_ATLAS_UNBLINDED.pdf in coordinator_only gives every selected case a plain trace, separate detector overlay, source labels, model decisions and explicit thresholds. Do not send this or case_key_DO_NOT_SHARE.csv to a blinded reviewer.\n3. Send only the contents of reviewer_blinded to an EP reviewer after feasibility is agreed. Pass A is Lead II only; Pass B is12lead, same10seconds. Source identifiers are replaced by random-order EP IDs. The rating sheet contains no labels or categories.\n4. CHRIS_MESSAGE_DRAFT.md is the unsent feasibility/advice request.\n\nSelection:24 unique heldout records; four per computational category. Categories A–E contain two source severe-group positives and two negatives each; F contains four other-critical source negatives with a false alert. Cases are sampled without replacement in declared category order, then randomly ordered with seed20261120. Categories may overlap in the available pool, but no selected record appears twice. This is an enriched convenience pilot and cannot estimate accuracy, prevalence or general clinical difficulty. Source-positive/negative status is record-level, not adjudicated segment truth.\n\nClinical contribution to assess: whether independent, blinded ratings of actual visible findings and difficulty correspond to the computational categories; whether extra leads resolve uncertainty; and whether source-label mismatches or acquisition/algorithm failures explain discrepancies. One reader cannot establish inter-reader agreement. Failure alone is not evidence of clinical difficulty. Preserve existing predictions and benchmark results; any adjudication is a new exploratory analysis, not retrospective relabeling or test-set tuning.\n\nNo reviewer has been contacted and no adjudication has occurred.\n''')
 # Verify exact blindness boundaries in the exported form and case mapping.
 ratings=pd.read_csv(blind/'RATINGS_blank.csv');assert len(ratings)==48 and ratings.case_id.nunique()==24;assert not any(c in ratings.columns for c in ['record_uid','source_labels','category','raw_score','set_score']);assert table.groupby('category').size().eq(4).all()
 manifest={str(p.relative_to(DEST)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.rglob('*') if p.is_file()};(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2));print('EP_PACKET',len(table),'unique cases;24pages per blinded pass;48blank ratings; no models or labels changed',flush=True)
if __name__=='__main__':main()
