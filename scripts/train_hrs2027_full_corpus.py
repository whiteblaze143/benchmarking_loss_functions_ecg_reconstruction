"""Full-corpus record-group-disjoint OOF evaluation and final external-transfer training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import wfdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototype"))
from prototype.models import SetOperatorPlusRR
from scripts.hrs2027_protocol import extract_features, parse_noise_labels, quality_mask
from scripts.hrs2027_gpu_features import batch_phase_kme

NOISE = Path('/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0/wfdb_files')


def sha256(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def fit(features, labels, indices, epochs, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SetOperatorPlusRR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Fixed epoch count declared before SHDB inspection; no best-fold selection.
    generator = np.random.default_rng(seed)
    tensors = [torch.from_numpy(x).to(device) for x in features]
    target = torch.tensor(labels, dtype=torch.float32, device=device).reshape(-1, 1)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in np.array_split(generator.permutation(indices), max(1, int(np.ceil(len(indices) / 64)))):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(x[batch] for x in tensors))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f'seed={seed} epoch={epoch+1}/{epochs} n={len(indices)} loss={total_loss/len(indices):.6f}', flush=True)
    model.eval()
    return model


def predict(model, features, indices, device):
    values = []
    with torch.no_grad():
        for batch in np.array_split(indices, max(1, int(np.ceil(len(indices) / 128)))):
            values.append(model.predict_proba(*(torch.from_numpy(x[batch]).to(device) for x in features)).cpu().numpy().ravel())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'final_setoperator.pt').exists():
        raise FileExistsError('Choose a new output directory; final checkpoint already exists')
    torch.set_num_threads(4)
    if torch.backends.cudnn.version()!=90300:
        raise RuntimeError('Use the isolated cuDNN 9.3 runtime documented in .aris/compute/local.md')
    torch.backends.cudnn.enabled = True
    device = 'cuda'
    anchors = np.random.default_rng(42).normal(size=(128,16)).astype(np.float32)
    files = sorted(NOISE.glob('*.hea'))
    if len(files) != 6130:
        raise ValueError(f'Expected 6130 segments, found {len(files)}')
    rows, beat_cells, rrs, hrvs = [], [], [], []
    for i, path in enumerate(files):
        base = str(path.with_suffix(''))
        record = wfdb.rdrecord(base)
        tier, label, events = parse_noise_labels(record.comments)
        _, rr, hrv, sqi, n_peaks, cells = extract_features(record.p_signal[:,0], record.fs, anchors,return_cells=True)
        beat_cells.append(cells); rrs.append(rr); hrvs.append(hrv)
        rows.append(dict(record_id=path.stem, record_group_id=path.stem.split('-')[0],
                         quality=tier, label=label, events=events, sqi=sqi, n_peaks=n_peaks,
                         duration_sec=len(record.p_signal)/record.fs,
                         header_sha256=sha256(path), signal_sha256=sha256(path.with_suffix('.dat'))))
        if (i+1) % 500 == 0:
            print(f'FEATURES {i+1}/6130', flush=True)
    manifest = pd.DataFrame(rows)
    features = [batch_phase_kme(beat_cells,anchors),np.asarray(rrs,dtype=np.float32),np.asarray(hrvs,dtype=np.float32)]
    del beat_cells
    if not all(np.isfinite(x).all() for x in features):
        raise ValueError('Non-finite extracted features')
    labels = manifest.label.to_numpy()
    groups = manifest.record_group_id.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    # Stratify jointly on rhythm and quality without breaking recording groups.
    strata = manifest.quality + '_' + manifest.label.astype(str)
    manifest['held_out_fold'] = -1
    manifest['oof_probability'] = np.nan
    metrics = []
    for fold, (train, test) in enumerate(splitter.split(features[0], strata, groups)):
        assert not set(groups[train]) & set(groups[test])
        model = fit(features, labels, train, args.epochs, 42+fold, device)
        probabilities = predict(model, features, test, device)
        manifest.loc[test,'held_out_fold'] = fold
        manifest.loc[test,'oof_probability'] = probabilities
        torch.save(model.state_dict(), out / f'fold_{fold}.pt')
        metrics.append(dict(fold=fold,train_segments=len(train),test_segments=len(test),
                            train_record_groups=len(set(groups[train])),test_record_groups=len(set(groups[test])),
                            record_group_overlap=0,auroc=roc_auc_score(labels[test],probabilities),
                            auprc=average_precision_score(labels[test],probabilities)))
        print('FOLD', metrics[-1], flush=True)
        del model
    assert manifest.oof_probability.notna().all()
    assert manifest.groupby('record_group_id').held_out_fold.nunique().max() == 1
    manifest.to_csv(out/'noise_manifest_and_oof_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'noise_record_group_cv.csv',index=False)
    # Evaluate only probabilities from models that never trained on these recording groups.
    sweep = []
    for tier, sub in manifest.groupby('quality'):
        prob, y, sqi = sub.oof_probability.to_numpy(), sub.label.to_numpy(), sub.sqi.to_numpy()
        for q in [.1,.2,.3,.4,.5,.55,.6,.7,.8,.9]:
            abstain = quality_mask(prob,sqi,q)
            accepted = ~abstain
            neg = accepted & (y==0); pos = accepted & (y==1)
            sweep.append(dict(model='SetOperator_OOF',quality=tier,quality_threshold=q,n_segments=len(sub),
                              n_record_groups=sub.record_group_id.nunique(),accepted=int(accepted.sum()),
                              coverage_pct=100*accepted.mean(),abstention_pct=100*abstain.mean(),
                              accepted_error_pct=100*np.mean((prob[accepted]>=.5)!=y[accepted]) if accepted.any() else np.nan,
                              accepted_sensitivity_pct=100*np.mean(prob[pos]>=.5) if pos.any() else np.nan,
                              accepted_fpr_pct=100*np.mean(prob[neg]>=.5) if neg.any() else np.nan,
                              missed_af_during_abstention=int(((y==1)&abstain).sum())))
    pd.DataFrame(sweep).to_csv(out/'noise_risk_coverage_sweep.csv',index=False)
    model = fit(features,labels,np.arange(6130),args.epochs,1042,device)
    torch.save(model.state_dict(),out/'final_setoperator.pt')
    np.save(out/'anchors.npy',anchors)
    spec = dict(n_segments=len(manifest),n_record_groups=manifest.record_group_id.nunique(),epochs=args.epochs,
                seed=1042,fold_seeds=list(range(42,47)),device=device,torch_version=torch.__version__,
                cudnn_version=torch.backends.cudnn.version(),cudnn_enabled=torch.backends.cudnn.enabled,
                final_training='all 6130 available segments; fixed 15 epochs; no fold winner selection',
                noise_evaluation='record-group-disjoint 5-fold out-of-fold; patient disjointness cannot be verified',
                patient_disjointness_verified=False,source_patients_reported=258,grouping_unit='daily-record filename prefix',
                fixed_policy=dict(probability_threshold=.5,sqi_threshold=.55,uncertainty_threshold=.35),
                zero_peak_segments=int((manifest.n_peaks==0).sum()),
                checkpoint_sha256=sha256(out/'final_setoperator.pt'),
                manifest_sha256=sha256(out/'noise_manifest_and_oof_predictions.csv'),
                code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'scripts/hrs2027_protocol.py',ROOT/'scripts/hrs2027_gpu_features.py',ROOT/'prototype/models.py']})
    (out/'training_provenance.json').write_text(json.dumps(spec,indent=2)+'\n')
    print('TRAINING_COMPLETE',json.dumps(spec),flush=True)


if __name__ == '__main__':
    main()
