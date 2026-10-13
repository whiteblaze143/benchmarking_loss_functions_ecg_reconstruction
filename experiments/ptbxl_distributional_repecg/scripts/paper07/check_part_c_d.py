#!/usr/bin/env python3
import ast
import pandas as pd
import numpy as np

def get_clean_rhythm(scp):
    if not isinstance(scp, dict):
        return 'EXCLUDE'
    has_afib = 'AFIB' in scp
    has_aflt = 'AFLT' in scp
    has_sr = 'SR' in scp
    if has_afib and not has_sr:
        return 'AF'
    elif has_sr and not has_afib and not has_aflt:
        return 'SR'
    else:
        return 'EXCLUDE'

print("Loading metadata...")
df_meta = pd.read_csv('data/ptbxl/ptbxl_database.csv', index_col='ecg_id')
df_meta['scp_codes'] = df_meta['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_meta['clean_rhythm'] = df_meta['scp_codes'].apply(get_clean_rhythm)
df_clean = df_meta[df_meta['clean_rhythm'].isin(['SR', 'AF'])].copy()
df_clean['recording_date'] = pd.to_datetime(df_clean['recording_date'])
df_clean = df_clean.sort_values(by=['patient_id', 'recording_date'])

train_ecg = np.load('/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts/train_ecg_ids.npy')
val_ecg = np.load('/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_full12_aux_mmd/frozen_full12_contexts/validation_ecg_ids.npy')
all_ecg = set(train_ecg.tolist() + val_ecg.tolist())

# Build pairs exactly as in Stage 4B
pairs = []
for pt_id, group in df_clean.groupby('patient_id', sort=False):
    if len(group) < 2:
        continue
    records = group.reset_index().to_dict('records')
    for t_idx in range(1, len(records)):
        target = records[t_idx]
        target_ecg = int(target['ecg_id'])
        if target_ecg not in all_ecg:
            continue
        baseline = None
        for b_idx in range(t_idx - 1, -1, -1):
            cand = records[b_idx]
            cand_ecg = int(cand['ecg_id'])
            if cand_ecg in all_ecg and cand['recording_date'] < target['recording_date']:
                baseline = cand
                break
        if baseline is not None:
            pairs.append({
                'target': target,
                'baseline': baseline,
                'cell': f"{baseline['clean_rhythm']}->{target['clean_rhythm']}",
                'strat_fold': int(target['strat_fold'])
            })

af_pairs = [p for p in pairs if p['baseline']['clean_rhythm'] == 'AF']
test_af = [p for p in af_pairs if p['strat_fold'] == 8]

print(f"Total test AF pairs: {len(test_af)}")

# Part C: Older same-patient AF baseline audit
print("\n=== Part C: Older Baseline Audit ===")
older_audit = []
for p in test_af:
    pt_id = p['target']['patient_id']
    target_date = p['target']['recording_date']
    baseline_date = p['baseline']['recording_date']
    
    # Get all records for this patient
    pt_records = df_clean[df_clean['patient_id'] == pt_id].reset_index().to_dict('records')
    
    # Find older AF baseline strictly before baseline_date
    older_cands = [
        r for r in pt_records
        if int(r['ecg_id']) in all_ecg
        and r['clean_rhythm'] == 'AF'
        and r['recording_date'] < baseline_date
    ]
    # Immediately preceding eligible AF ECG before the current nearest baseline
    older_b = older_cands[-1] if len(older_cands) > 0 else None
    
    older_audit.append({
        'patient_id': pt_id,
        'target_ecg_id': p['target']['ecg_id'],
        'baseline_ecg_id': p['baseline']['ecg_id'],
        'cell': p['cell'],
        'has_older_af': older_b is not None,
        'older_ecg_id': older_b['ecg_id'] if older_b else None,
    })

df_c = pd.DataFrame(older_audit)
print("Older AF baseline availability by cell:")
for cell in ['AF->SR', 'AF->AF']:
    sub = df_c[df_c['cell'] == cell]
    n_pairs_with_older = sub['has_older_af'].sum()
    n_pats_with_older = sub[sub['has_older_af']]['patient_id'].nunique()
    total_pairs = len(sub)
    total_pats = sub['patient_id'].nunique()
    print(f"  {cell}: pairs with older AF = {n_pairs_with_older}/{total_pairs}, unique patients with older AF = {n_pats_with_older}/{total_pats}")

# Part D: Acquisition consistency audit
print("\n=== Part D: Acquisition Consistency Audit (Site & Device) ===")
acq_audit = []
for p in test_af:
    t = p['target']
    b = p['baseline']
    same_site = (t['site'] == b['site']) if pd.notna(t['site']) and pd.notna(b['site']) else False
    same_device = (t['device'] == b['device']) if pd.notna(t['device']) and pd.notna(b['device']) else False
    acq_audit.append({
        'patient_id': t['patient_id'],
        'target_ecg_id': t['ecg_id'],
        'baseline_ecg_id': b['ecg_id'],
        'cell': p['cell'],
        'same_site': same_site,
        'same_device': same_device,
        'target_site': t['site'],
        'baseline_site': b['site'],
        'target_device': t['device'],
        'baseline_device': b['device'],
    })

df_d = pd.DataFrame(acq_audit)
print("Device consistency by cell:")
for cell in ['AF->SR', 'AF->AF']:
    sub = df_d[df_d['cell'] == cell]
    same_dev_pairs = sub['same_device'].sum()
    diff_dev_pairs = (~sub['same_device']).sum()
    same_dev_pats = sub[sub['same_device']]['patient_id'].nunique()
    diff_dev_pats = sub[~sub['same_device']]['patient_id'].nunique()
    print(f"  {cell} device: same={same_dev_pairs} pairs ({same_dev_pats} pats), diff={diff_dev_pairs} pairs ({diff_dev_pats} pats)")

print("Site consistency by cell:")
for cell in ['AF->SR', 'AF->AF']:
    sub = df_d[df_d['cell'] == cell]
    same_site_pairs = sub['same_site'].sum()
    diff_site_pairs = (~sub['same_site']).sum()
    same_site_pats = sub[sub['same_site']]['patient_id'].nunique()
    diff_site_pats = sub[~sub['same_site']]['patient_id'].nunique()
    print(f"  {cell} site: same={same_site_pairs} pairs ({same_site_pats} pats), diff={diff_site_pairs} pairs ({diff_site_pats} pats)")
