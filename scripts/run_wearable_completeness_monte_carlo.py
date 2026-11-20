#!/usr/bin/env python3
"""Wear simulation using archived measured predictions and the canonical reference episodes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prototype.metrics_engine import extract_continuous_episodes, match_episodes
from scripts.hrs2027_protocol import quality_mask


def causal_average(probs, wear, k=9):
    finite=np.isfinite(probs)
    sums=np.r_[0,np.cumsum(np.where(finite,probs,0),dtype=np.float64)]
    counts=np.r_[0,np.cumsum(finite)]
    ends=np.arange(1,len(probs)+1)
    starts=np.maximum(0,ends-k)
    n=counts[ends]-counts[starts]
    result=np.divide(sums[ends]-sums[starts],n,out=np.full(len(probs),np.nan),where=n>0)
    result[~wear]=np.nan
    return result


def evaluate_schedule(prediction_dir, output_dir, iterations=1000, seed=42):
    out = Path(output_dir)
    manifest = pd.read_csv(out/'shdb_cohort_manifest.csv')
    references = pd.read_csv(out/'reference_episodes.csv')
    rng = np.random.default_rng(seed)
    records = []
    for row in manifest[manifest.Annotated & manifest.Previous_AF_Ablation].itertuples():
        rec_id = int(row.Data_ID)
        archive = np.load(Path(prediction_dir)/f'{rec_id:03d}.npz')
        refs = references[references.record_id == rec_id].to_dict('records')
        records.append((row, archive, refs))
    rows = []
    schedules = [('Continuous 24h',24),('Random contiguous 16h',16),
                 ('Random contiguous 12h',12),('Random contiguous 8h',8),
                 ('Random contiguous 4h',4),('Two random 30s checks',0)]
    for name, hours in schedules:
        for iteration in range(1 if hours==24 else iterations):
            for row, archive, refs in records:
                duration = archive['duration_sec']
                n = len(duration)
                wear = np.zeros(n,dtype=bool)
                if hours==24:
                    wear[:] = True
                elif hours==0:
                    indices = rng.choice(n,size=2,replace=False)
                    wear[indices] = True
                else:
                    count = min(n,int(hours*120))
                    start = rng.integers(0,n-count+1)
                    wear[start:start+count] = True
                raw, sqi = archive['SetOperator_Raw'], archive['sqi']
                abstain = quality_mask(raw,sqi) | ~wear
                selected = np.where(abstain,np.nan,raw)
                probs = causal_average(selected,wear)
                probs[abstain] = np.nan
                if hours==24:
                    probs=archive['QualityAware_SetOperator']
                detected = (probs>=.5) & np.isfinite(probs)
                accepted = np.isfinite(probs)
                total = duration.sum()
                unknown = duration[~accepted].sum()
                # Valid partial-identification bounds use reference truth in accepted time.
                # Model outputs alone cannot guarantee bounds in the presence of classification error.
                reference_observed = archive['gt_af_sec'][accepted].sum()
                b_min = 100*reference_observed/total
                b_max = 100*(reference_observed+unknown)/total
                row_out = dict(protocol=name,iteration=iteration,record_id=int(row.Data_ID),
                               subject_id=int(row.Subject_ID),post_blanking=bool(row.post_blanking),
                               wear_pct=100*duration[wear].sum()/total,
                               diagnostic_coverage_pct=100*duration[accepted].sum()/total,
                               reference_burden_min_pct=b_min,reference_burden_max_pct=b_max,
                               reference_burden_pct=100*archive['gt_af_sec'].sum()/total,
                               indeterminate_no_detected_af=bool(not detected.any() and unknown>0))
                starts = np.arange(n)*30
                for endpoint,minimum,overlap in [('30s',30,15),('6min',360,60)]:
                    eligible = [r for r in refs if r['duration_sec']>=minimum]
                    pred_eps = extract_continuous_episodes(detected,window_sec=30,min_duration_sec=minimum,record_duration_sec=total)
                    match = match_episodes(eligible,pred_eps,min_overlap_sec=overlap)
                    observable = 0
                    for ref in eligible:
                        overlap_seconds = np.maximum(0, np.minimum(starts+duration,ref['end_sec'])
                                                      - np.maximum(starts,ref['start_sec']))
                        observable += overlap_seconds[wear].sum() >= (15 if endpoint=='30s' else 180)
                    row_out[f'reference_{endpoint}'] = len(eligible)
                    row_out[f'detected_{endpoint}'] = match['detected_ref']
                    row_out[f'observable_{endpoint}'] = observable
                rows.append(row_out)
        print(f'WEAR {name} complete',flush=True)
    detail = pd.DataFrame(rows)
    detail.to_parquet(out/'wear_schedule_record_predictions.parquet',index=False)
    summaries=[]
    for (protocol,iteration),sub in detail.groupby(['protocol','iteration']):
        entry=dict(protocol=protocol,iteration=iteration,n_records=len(sub),n_subjects=sub.subject_id.nunique(),
                   wear_pct=sub.wear_pct.mean(),diagnostic_coverage_pct=sub.diagnostic_coverage_pct.mean(),
                   reference_bound_width_pct=(sub.reference_burden_max_pct-sub.reference_burden_min_pct).mean(),
                   indeterminate_pct=100*sub.indeterminate_no_detected_af.mean())
        for endpoint in ['30s','6min']:
            total=sub[f'reference_{endpoint}'].sum()
            entry[f'sensitivity_{endpoint}_pct']=100*sub[f'detected_{endpoint}'].sum()/total if total else np.nan
            entry[f'observable_{endpoint}_pct']=100*sub[f'observable_{endpoint}'].sum()/total if total else np.nan
        summaries.append(entry)
    per_iteration=pd.DataFrame(summaries)
    per_iteration.to_csv(out/'wear_schedule_iterations.csv',index=False)
    grouped=per_iteration.groupby('protocol')
    mean=grouped.mean(numeric_only=True).drop(columns='iteration')
    for metric in ['sensitivity_30s_pct','sensitivity_6min_pct']:
        mean[f'{metric}_simulation_p025']=grouped[metric].quantile(.025)
        mean[f'{metric}_simulation_p975']=grouped[metric].quantile(.975)
    mean.to_csv(out/'wear_schedule_sensitivity_analysis.csv')
    print(mean.to_string(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--iterations',type=int,default=1000)
    args=parser.parse_args()
    evaluate_schedule(Path(args.output_dir)/'window_predictions',args.output_dir,args.iterations)
