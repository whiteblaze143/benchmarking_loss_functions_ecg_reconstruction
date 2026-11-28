"""Priority 5 & 7 Addendum: Untrained Random-Init FM Controls, GPU Runtime Benchmarks, SHA-256 & Canonical Registry."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
import foundation_adapters as fa
from waveform_loader import WaveformLoader

OUT_DIR = Path("experiments/af_density_reconciliation_v7r/results")
V7P_DIR = Path("experiments/af_density_prediction_v7p")

CHECKPOINTS = {
    "CSFM": ("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/Cardiac-Sensing-FM/pretrained/CSFM_tiny.pth", "CSFM"),
    "CLEF": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/clef/clef_medium.ckpt", "CLEF"),
    "ECGFOUNDER": (str(V7P_DIR / "checkpoints" / "1_lead_ECGFounder.pth"), "ECGFOUNDER"),
    "ECGFM_KED": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_weights/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt", "KED"),
    "ECHONEXT": ("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/external/IntroECG/7-EchoNext Minimodel/models/echonext_multilabel_minimodel/weights.pt", "ECHONEXT"),
    "DINOSR": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/checkpoints/dinosr_e10.ckpt", "DINOSR"),
    "JEPA": ("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/checkpoints/jepa_e10.ckpt", "JEPA"),
    "NEFCORE": ("/data/mithunmanivannan/nef-net-aim/02_nef_core/pretrained/nefcore_full.pt", "NEFCORE"),
    "F2": ("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/F2/pilot_final.pt", "F2"),
    "F0": ("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/F0/pilot_final.pt", "F0"),
    "G1": ("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/g1_seed42/best_model.pt", "G1"),
    "CONTROL": ("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/runs/control_capacity_seed42/best_model.pt", "CAPACITY"),
}


def sha256_file(path_str: str) -> str:
    h = hashlib.sha256()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def randomize_module_weights(module: torch.nn.Module, seed: int = 42):
    torch.manual_seed(seed)
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return module


def eval_embeddings(X_emb, y, folds, pca_dim=32):
    out = {}
    for mk in ["logistic", "lgbm"]:
        oof = np.zeros(len(y))
        for f in sorted(np.unique(folds)):
            tr, va = folds != f, folds == f
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_emb[tr])
            Xva = sc.transform(X_emb[va])
            pca = PCA(n_components=min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1), svd_solver="randomized", random_state=42)
            Xtr_p = pca.fit_transform(Xtr)
            Xva_p = pca.transform(Xva)
            clf = LogisticRegression(C=1.0, max_iter=500) if mk == "logistic" else LGBMClassifier(
                n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1
            )
            oof[va] = clf.fit(Xtr_p, y[tr]).predict_proba(Xva_p)[:, 1]
        out[mk] = round(float(roc_auc_score(y, oof)), 4)
    return out


def run_random_init_and_runtime():
    print("=== Priority 7: Untrained Random-Init Control & A100 GPU Runtime Benchmark ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    p4 = pd.read_parquet(V7P_DIR / "P04_noise_classification_manifest.parquet")

    wl = WaveformLoader()
    # Extract P04 (6130 strips, 10s @ 200Hz) once for fast A100 runtime & random-init check
    raw_p4 = []
    for row in p4.itertuples():
        sig, fs = wl.get_causal_segment("Noise-Holter", row.strip_id, 10.0, duration_s=10.0)
        raw_p4.append((sig, fs))
    wl.close()

    rows = []
    y4 = p4["af_present"].astype(int).values
    f4 = p4["fold"].values

    for mkey in ["ECGFOUNDER", "ECHONEXT", "F2", "CSFM"]:
        ckpt_path, adapter_key = CHECKPOINTS[mkey]
        ad = fa.get_model_adapter(adapter_key, ckpt_path, device)
        target_fs, target_len = float(ad.fs), ad.target_len

        # Resample all 6130 segments cleanly once per target_fs
        arr = np.stack([fa.resample_signal(s, fs, target_fs, target_len) for s, fs in raw_p4], axis=0)[:, np.newaxis, :]
        n = len(arr)

        # Pretrained A100 forward pass
        emb_pre = np.zeros((n, ad.dim), dtype=np.float32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for b0 in range(0, n, 256):
            xb = torch.from_numpy(arr[b0:b0 + 256]).float().to(device)
            with torch.no_grad():
                emb_pre[b0:b0 + 256] = ad(xb).cpu().numpy()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_pre = time.time() - t0
        m_pre = eval_embeddings(emb_pre, y4, f4)

        # Untrained random-init forward pass (same architecture, random weights, seed=42)
        randomize_module_weights(ad, seed=42)
        ad.eval().to(device)
        emb_rnd = np.zeros((n, ad.dim), dtype=np.float32)
        for b0 in range(0, n, 256):
            xb = torch.from_numpy(arr[b0:b0 + 256]).float().to(device)
            with torch.no_grad():
                emb_rnd[b0:b0 + 256] = ad(xb).cpu().numpy()
        m_rnd = eval_embeddings(emb_rnd, y4, f4)

        rows.append({
            "model": mkey,
            "task": "P04_NoiseHolter",
            "gpu_infer_s_6130": round(dt_pre, 3),
            "throughput_strips_per_s": round(n / dt_pre, 1),
            "pretrained_logistic_auroc": m_pre["logistic"],
            "pretrained_lgbm_auroc": m_pre["lgbm"],
            "random_init_logistic_auroc": m_rnd["logistic"],
            "random_init_lgbm_auroc": m_rnd["lgbm"],
            "pretrain_gain_logistic": round(m_pre["logistic"] - m_rnd["logistic"], 4),
            "pretrain_gain_lgbm": round(m_pre["lgbm"] - m_rnd["lgbm"], 4),
        })
        del ad
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "V7R_fm_random_init_and_runtime.csv", index=False)
    print(df.to_string(index=False))


def build_canonical_registry():
    print("\n=== Priority 5: Building V7R_canonical_model_registry.json ===")
    ckpt_hashes = {}
    for k, (p, ak) in CHECKPOINTS.items():
        if Path(p).exists():
            ckpt_hashes[k] = {"path": p, "adapter_key": ak, "sha256": sha256_file(p)}

    manifest_paths = {
        "P01_raw": str(V7P_DIR / "P01_landmark_manifest.parquet"),
        "P02": str(V7P_DIR / "P02_episode_persistence_manifest.parquet"),
        "P03": str(V7P_DIR / "P03_survival_manifest.parquet"),
        "P04": str(V7P_DIR / "P04_noise_classification_manifest.parquet"),
        "V7R_recording_inventory": str(OUT_DIR / "V7R_recording_inventory.parquet"),
    }
    manifest_hashes = {k: sha256_file(p) for k, p in manifest_paths.items() if Path(p).exists()}

    reg = {
        "audit_version": "V7-R",
        "cv_protocol": "5-fold patient-disjoint cross-validation; zero inner hyperparam search for fixed canonical models; logistic C=1.0 max_iter=500; lgbm n_estimators=100 lr=0.05 num_leaves=31 seed=42",
        "calibration_policy": "Unmodified CV out-of-fold probabilities evaluated directly for AUROC, AUPRC, Brier, and clinical threshold sweep",
        "bootstrap_protocol": "Patient-clustered paired bootstrap B=2,000 replicates; exact two-sided Monte Carlo p-value with +1/(B+1) resolution floor (min_attainable_p = 2/2001 = 0.0009995)",
        "tasks": {
            "P01_CleanAnnotated": {
                "horizon_s": 1800.0,
                "target": "target_sustained_30m",
                "ascertainment": "Initiation of >=30m AF episode in [t0, t0+1800s] strictly inside annotated known-rhythm non-AF landmarks (removed 8,126 unannotated SHDB + 221 UNKNOWN LTAFDB rows)",
            },
            "P02_Persistence60s": {
                "horizon_s": 1800.0,
                "target": "target_persists_30m",
                "ascertainment": "Landmark at t=60s after AF onset for episodes with duration >=60s; outcome = eventual total_duration_s >= 1800s; future duration strictly excluded from predictors",
            },
            "P03_Survival15m": {
                "horizon_s": 900.0,
                "target": "target_terminate_in_15m",
                "ascertainment": "Landmark at t in {60s, 300s, 900s} during ongoing AF; outcome = episode terminates within (t, t+900s]",
            },
            "P04_NoiseHolter": {
                "horizon_s": 10.0,
                "target": "af_present",
                "ascertainment": "10s annotated excerpt classification across Clean, N1, N2 quality strata (n=6,130)",
            },
        },
        "canonical_feature_sets": {
            "S0_Baseline": ["time_since_rec_start_s", "elapsed_af_duration_s"],
            "S1_Burden": ["S0 + last_af_duration_s, time_since_last_af_s, has_prior_af, af_burden_{15m,30m,1h,3h,6h}"],
            "S2_Architecture": ["S1 + {episode_count, longest_af_run, brief_run_count}_{15m,30m,1h,3h,6h}"],
            "S3a_Arch_Plus_DensityOnly": ["S2 + charitos_density_{15m,30m,1h,3h,6h} + is_density_defined_3h"],
            "S3b_Arch_Plus_AggrOnly": ["S2 + af_aggregation_{15m,30m,1h,3h,6h} + is_density_defined_3h"],
            "S3c_Arch_Plus_Dens_Aggr": ["S2 + charitos_density + af_aggregation + is_density_defined_3h"],
            "S4_CleanRR_NoCohortShortcut": ["S3c + {mean_hr_bpm, std_hr_bpm, rmssd_ms, cvrr, pnn50} evaluated strictly within LTAFDB+SHDB valid-RR subset without is_rr_valid"],
        },
        "manifest_sha256": manifest_hashes,
        "checkpoint_sha256": ckpt_hashes,
    }
    (OUT_DIR / "V7R_canonical_model_registry.json").write_text(json.dumps(reg, indent=2))
    print("Saved V7R_canonical_model_registry.json")


if __name__ == "__main__":
    run_random_init_and_runtime()
    build_canonical_registry()
