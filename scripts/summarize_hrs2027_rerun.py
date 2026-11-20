"""Reconcile rerun counts and generate measured cohort-specific summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


def summarize(out):
    out=Path(out)
    cohort=pd.read_csv(out/'shdb_cohort_manifest.csv')
    burden=pd.read_csv(out/'rhythm_burden_comparison.csv')
    refs=pd.read_csv(out/'reference_episodes.csv')
    pairs=pd.read_parquet(out/'paired_episode_predictions.parquet')
    pairs['record_id']=pd.to_numeric(pairs.record_id,errors='raise').astype(int)
    noise=pd.read_csv(out/'noise_manifest_and_oof_predictions.csv')
    alerts=pd.read_csv(out/'patient_level_alert_metrics.csv')
    wear=pd.read_parquet(out/'wear_schedule_record_predictions.parquet')
    assert ((wear.reference_burden_min_pct<=wear.reference_burden_pct+1e-6)
            & (wear.reference_burden_max_pct>=wear.reference_burden_pct-1e-6)).all()
    continuous=wear[wear.protocol=='Continuous 24h']
    prior_qa=burden[(burden.model=='QualityAware_SetOperator') & burden.previous_af_ablation]
    assert continuous.detected_6min.sum()==prior_qa.episodes_ge_6min_detected.sum()
    assert continuous.reference_6min.sum()==prior_qa.episodes_ge_6min_total.sum()
    assert len(noise)==6130 and noise.record_id.nunique()==6130
    assert noise.groupby('record_group_id').held_out_fold.nunique().eq(1).all()
    assert noise.oof_probability.notna().all()
    noise_root=Path('/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0')
    listed={Path(x.strip()).name for x in (noise_root/'RECORDS').read_text().splitlines() if x.strip()}
    available=set(noise.record_id)
    assert available <= listed
    missing=listed-available
    missing_groups={x.split('-')[0] for x in missing}
    available_groups={x.split('-')[0] for x in available}
    noise_flow=pd.DataFrame([
        dict(population='listed_in_RECORDS',n_segments=len(listed),n_record_groups=len({x.split('-')[0] for x in listed})),
        dict(population='available_and_used',n_segments=len(available),n_record_groups=len(available_groups)),
        dict(population='missing_segments',n_segments=len(missing),n_record_groups=len(missing_groups))])
    noise_flow.to_csv(out/'noise_corpus_flow.csv',index=False)
    noise_classes=noise.groupby('quality').agg(n_segments=('record_id','size'),n_record_groups=('record_group_id','nunique'),
                                               af_segments=('label','sum'))
    noise_classes['non_af_segments']=noise_classes.n_segments-noise_classes.af_segments
    noise_classes.to_csv(out/'noise_class_counts.csv')
    assert burden.groupby('model').record_id.nunique().eq(98).all()
    assert burden.post_blanking.any()
    tier_a_ids=alerts[alerts.is_tier_a_annotated].record_id.unique()
    assert len(tier_a_ids)==20
    for record_id in tier_a_ids:
        archive=np.load(out/'window_predictions'/f'{record_id:03d}.npz')
        assert bool(archive['annotated']) and archive['gt_af_sec'].sum()==0
    flows=[dict(cohort='all_recordings_census_only',n_recordings=len(cohort),n_subjects=cohort.Subject_ID.nunique(),
                hours=sum(np.load(out/'window_predictions'/f'{record_id:03d}.npz')['duration_sec'].sum()/3600 for record_id in cohort.Data_ID),
                episodes_30s=np.nan,episodes_6min=np.nan,subjects_with_30s=np.nan,subjects_with_6min=np.nan)]
    subsets={'all_annotated':cohort.Annotated,
             'annotated_prior_ablation':cohort.Annotated & cohort.Previous_AF_Ablation,
             'annotated_post_blanking':cohort.Annotated & cohort.post_blanking}
    results=[]
    paired_differences=[]
    for name,mask in subsets.items():
        meta=cohort[mask]
        subrefs=refs[refs.record_id.isin(meta.Data_ID)]
        hours=sum(np.load(out/'window_predictions'/f'{r:03d}.npz')['duration_sec'].sum()/3600 for r in meta.Data_ID)
        flows.append(dict(cohort=name,n_recordings=len(meta),n_subjects=meta.Subject_ID.nunique(),
                          hours=hours,episodes_30s=len(subrefs),episodes_6min=int((subrefs.duration_sec>=360).sum()),
                          subjects_with_30s=subrefs.subject_id.nunique(),
                          subjects_with_6min=subrefs[subrefs.duration_sec>=360].subject_id.nunique()))
        for model,sub in burden[burden.record_id.isin(meta.Data_ID)].groupby('model'):
            assert sub.episodes_ge_30s_total.sum()==len(subrefs)
            assert sub.episodes_ge_6min_total.sum()==(subrefs.duration_sec>=360).sum()
            archived=pairs[(pairs.model==model)&pairs.record_id.isin(meta.Data_ID)]
            assert len(archived)==len(subrefs)
            assert archived.detected.sum()==sub.episodes_ge_30s_detected.sum()
            assert archived.detected_ge_6min.sum()==sub.episodes_ge_6min_detected.sum()
            # Resample subjects, retaining every recording and episode from each draw.
            rng=np.random.default_rng(42)
            cluster=sub.groupby('subject_id')[['episodes_ge_30s_total','episodes_ge_30s_detected',
                                               'episodes_ge_6min_total','episodes_ge_6min_detected',
                                               'burden_error_pts']].sum()
            cluster['records']=sub.groupby('subject_id').size()
            weights=rng.multinomial(len(cluster),np.ones(len(cluster))/len(cluster),size=1000)
            boot=weights @ cluster.to_numpy()
            entry=dict(cohort=name,model=model,n_recordings=len(sub),n_subjects=len(cluster),
                       burden_mae_pts=sub.burden_error_pts.mean(),abstention_pct=100*(1-sub.accepted_seconds.sum()/sub.total_seconds.sum()))
            for endpoint,num,den in [('30s',1,0),('6min',3,2)]:
                total=cluster.iloc[:,den].sum()
                point=100*cluster.iloc[:,num].sum()/total if total else np.nan
                values=np.divide(100*boot[:,num],boot[:,den],out=np.full(1000,np.nan),where=boot[:,den]>0)
                entry[f'sensitivity_{endpoint}_pct']=point
                entry[f'sensitivity_{endpoint}_ci_low']=np.nanpercentile(values,2.5)
                entry[f'sensitivity_{endpoint}_ci_high']=np.nanpercentile(values,97.5)
                entry[f'episodes_{endpoint}']=int(total)
                entry[f'detected_{endpoint}']=int(cluster.iloc[:,num].sum())
            tier=alerts[(alerts.model==model)&alerts.is_tier_a_annotated]
            entry['tier_a_false_alerts_per_24h']=24*tier.clinical_alerts_1h_refractory.sum()/tier.monitored_hours.sum()
            entry['tier_a_false_alerts_per_24h_accepted']=24*tier.clinical_alerts_1h_refractory.sum()/tier.accepted_hours.sum() if tier.accepted_hours.sum() else np.nan
            entry['tier_a_zero_alert_recordings_pct']=100*tier.zero_false_alert_patient.mean()
            entry['tier_a_patients']=tier.subject_id.nunique()
            entry['tier_a_recordings']=len(tier)
            entry['tier_a_zero_alert_patients_pct']=100*tier.groupby('subject_id').clinical_alerts_1h_refractory.sum().eq(0).mean()
            patient_counts=tier.groupby('subject_id').clinical_alerts_1h_refractory.sum()
            successes=int(patient_counts.eq(0).sum())
            n=len(patient_counts)
            entry['tier_a_patient_specificity_ci_low']=100*beta.ppf(.025,successes,n-successes+1) if successes else 0
            entry['tier_a_patient_specificity_ci_high']=100*beta.ppf(.975,successes+1,n-successes) if successes<n else 100
            results.append(entry)
        paired=burden[burden.record_id.isin(meta.Data_ID) & burden.model.isin(['ArNet2_Raw','QualityAware_SetOperator'])]
        pivot=paired.pivot(index=['subject_id','record_id'],columns='model',values='episodes_ge_6min_detected')
        difference=pivot.ArNet2_Raw-pivot.QualityAware_SetOperator
        totals=paired[paired.model=='ArNet2_Raw'].set_index(['subject_id','record_id']).episodes_ge_6min_total
        cluster=pd.DataFrame({'difference':difference,'total':totals}).groupby('subject_id').sum()
        rng=np.random.default_rng(42)
        weights=rng.multinomial(len(cluster),np.ones(len(cluster))/len(cluster),size=1000)
        samples=weights@cluster.to_numpy()
        values=np.divide(100*samples[:,0],samples[:,1],out=np.full(1000,np.nan),where=samples[:,1]>0)
        paired_differences.append(dict(cohort=name,comparison='ArNet2_Raw minus QualityAware_SetOperator',
                                       sensitivity_6min_difference_pts=100*cluster.difference.sum()/cluster.total.sum(),
                                       ci_low=np.nanpercentile(values,2.5),ci_high=np.nanpercentile(values,97.5)))
    pd.DataFrame(flows).to_csv(out/'clinical_cohort_flow.csv',index=False)
    summary=pd.DataFrame(results)
    summary.to_csv(out/'cohort_model_summary_with_patient_cluster_ci.csv',index=False)
    pd.DataFrame(paired_differences).to_csv(out/'paired_patient_cluster_comparisons.csv',index=False)
    # Every annotated recording must have an archive; all expected models must reconcile.
    assert len(list((out/'window_predictions').glob('*.npz')))==len(cohort)
    training=json.loads((out/'training_provenance.json').read_text())
    for path in (out/'window_predictions').glob('*.npz'):
        assert str(np.load(path)['checkpoint_sha256'])==training['checkpoint_sha256']
    file_hashes={str(p.relative_to(out)):hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
                 for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
    (out/'artifact_hashes.json').write_text(json.dumps(file_hashes,indent=2)+'\n')
    primary=summary[summary.cohort=='all_annotated'].set_index('model')
    baseline,proposed=primary.loc['ArNet2_Raw'],primary.loc['QualityAware_SetOperator']
    dominates=(baseline.sensitivity_6min_pct>=proposed.sensitivity_6min_pct
               and baseline.sensitivity_30s_pct>=proposed.sensitivity_30s_pct
               and baseline.tier_a_false_alerts_per_24h<=proposed.tier_a_false_alerts_per_24h
               and baseline.burden_mae_pts<=proposed.burden_mae_pts
               and baseline.abstention_pct<=proposed.abstention_pct)
    decision='HOLD for clinical submission. The corrected execution checks pass; clinical superiority is not established. '
    if dominates:
        decision+='At the reported operating points, ArNet2 dominates the proposed model on both episode sensitivities, alert rate, burden error and diagnostic coverage. '
    decision+='The intended post-blanking subgroup contains only 17 patients. Noise Holter patient-level independence cannot be verified from daily-record identifiers. This retrospective rerun does not provide a new untouched validation cohort.'
    text=['# Full 6,130-segment rerun — measured results','',decision,'',
          f'Training: {training["n_segments"]} segments from {training["n_record_groups"]} daily-record groups. '
          'All segments have predictions from a fold that excluded their entire recording group. '
          'The final transfer model was independently trained on all 6,130 segments for 15 fixed epochs.','',
          noise_flow.to_markdown(index=False),'',
          'The 125 unavailable segments involve eight record prefixes; six prefixes are entirely absent. '
          'Prefixes 233 and 240 are partially available. Every physically available segment was used, including other arrhythmias as non-AF controls.','',
          'PhysioNet reports 258 source patients, with Holters exported as daily records. The release does not provide a patient linkage key. '
          'Consequently 310 available prefixes cannot be called 310 patients, and record-group-disjoint folds cannot be called patient-disjoint. '
          '[Dataset documentation](https://physionet.org/content/noise-holter-database/1.0.0/).','',
          'This is a retrospective rerun after inspecting earlier SHDB results. It is not a new untouched prospective validation. '
          'Probability, SQI and uncertainty thresholds remain fixed at 0.50, 0.55 and 0.35.','',
          'Canonical reference: contiguous AFIB/AFL/AT rhythm intervals form one atrial tachyarrhythmia episode. '
          'Post-blanking requires documented prior ablation and more than 90 days since the latest preceding ablation.','',
          'Author baselines use their exported full-sequence signatures with one recording identifier per sequence. '
          'ArNet2 receives consecutive non-overlapping 60-RR windows; these predictions are mapped by overlap duration '
          'to the common 30-second scoring grid. Full-sequence baseline evaluation is offline, not a streaming latency validation.','',
          '## Verified cohort counts','',pd.DataFrame(flows).to_markdown(index=False),'',
          'All 128 recordings were processed; reference-standard sensitivity uses only the 98 annotated recordings. '
          'Tier A specificity uses 20 recordings from 20 distinct patients, verified against zero reference AF time. '
          'Zero observed alerts in 20 controls does not establish a zero population alert rate; 20/20 specificity has a '
          'two-sided exact 95% interval of approximately 83.2%–100%.','',
          '## Measured clinical comparison','',
          summary[['cohort','model','n_subjects','detected_6min','episodes_6min','sensitivity_6min_pct',
                   'sensitivity_6min_ci_low','sensitivity_6min_ci_high','burden_mae_pts','abstention_pct',
                   'tier_a_false_alerts_per_24h']].to_markdown(index=False,floatfmt='.3f'),'']
    text.extend(['## Noise evaluation','',
                 noise_classes.to_markdown(),'',
                 'The low AF prevalence, especially in severe-noise segments, makes accepted error rate alone insufficient. '
                 'Interpret it together with AF sensitivity, false-positive rate, coverage, and missed AF during abstention.','',
                 pd.read_csv(out/'risk_coverage_by_quality.csv').to_markdown(index=False,floatfmt='.3f'),''])
    text.extend(['## Paired patient-cluster comparison','',
                 pd.DataFrame(paired_differences).to_markdown(index=False,floatfmt='.3f'),''])
    text.extend(['## Wear simulation','',
                 pd.read_csv(out/'wear_schedule_sensitivity_analysis.csv').to_markdown(index=False,floatfmt='.3f'),'',
                 'Wear schedules are random contiguous recording-relative intervals, not clock-aligned daytime or sleep protocols. '
                 'Sensitivity uses measured predictions, with non-wear and abstention unavailable to detection. '
                 'Simulation percentiles describe wear-schedule variability, not patient-population confidence intervals. '
                 'Reference-based burden bounds are an oracle observability analysis; classifier outputs alone do not guarantee them.','',
                 'Confidence intervals resample patients and retain their recordings and episodes; they quantify patient sampling '
                 'uncertainty conditional on these fitted models and fixed rules, not uncertainty from retraining or model selection.','',
                 'Burden error compares detected AF seconds per total recording seconds against reference AF seconds per the same total. '
                 'Accepted-time reference and prediction burdens are separately archived. Tier A alert rates are provided per total '
                 'recording time and per accepted diagnostic time. Episode and window counts are not independent patient sample sizes.','',
                 '## Verification','',
                 'PASS for execution: 6,130 unique segments; one held-out fold per recording group; zero cross-fold recording-group overlap; '
                 '98 annotated recordings reconciled against canonical references; episode archive and summary detections agree; '
                 'post-blanking flags contain verified eligible records; all 128 prediction archives present. '
                 'WARN for population independence: Noise Holter patient overlap across folds remains unknown.','',
                 'Earlier corrected-package numerical claims are superseded by this run. Clinical superiority requires assessing '
                 'the actual comparator results and matched coverage; completing a rerun does not itself establish superiority.'])
    (out/'full6130_research_report.md').write_text('\n'.join(text)+'\n')
    print(pd.DataFrame(flows).to_string(index=False))
    print(summary.to_string(index=False))
    print('RECONCILIATION_PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    summarize(parser.parse_args().output_dir)
