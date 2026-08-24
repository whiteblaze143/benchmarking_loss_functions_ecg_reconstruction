"""Bounded, provenance-aware waveform UMAPs for every local real ECG cohort."""
from __future__ import annotations

import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
import wfdb
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SEEDS = (42, 7, 101)


def pseudonym(dataset: str, value: object) -> str:
    digest = hashlib.sha256(f"book-umap-v1|{dataset}|{value}".encode()).hexdigest()[:12]
    return f"{dataset.lower()}-{digest}"


def deterministic_subset(items, maximum: int, namespace: str):
    items = list(items)
    return sorted(items, key=lambda x: hashlib.sha256(f"{namespace}|{x}".encode()).digest())[:maximum]


def waveform_features(signal: np.ndarray) -> np.ndarray:
    """Return 12-lead robust time/frequency summaries; never retain waveform rows."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected two-dimensional waveform, got {x.shape}")
    if x.shape[0] != 12 and x.shape[1] == 12:
        x = x.T
    if x.shape[0] != 12 or x.shape[1] < 100:
        raise ValueError(f"Expected 12 leads and >=100 samples, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Non-finite waveform")
    centered = x - np.median(x, axis=1, keepdims=True)
    diff = np.diff(centered, axis=1)
    spectrum = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    spectrum = spectrum[:, 1:]  # remove DC
    edges = np.linspace(0, spectrum.shape[1], 5, dtype=int)
    total = np.maximum(spectrum.sum(axis=1), 1e-12)
    bands = np.stack([
        spectrum[:, edges[i]:edges[i + 1]].sum(axis=1) / total for i in range(4)
    ], axis=1)
    per_lead = np.column_stack([
        np.sqrt(np.mean(centered ** 2, axis=1)),
        np.quantile(centered, .75, axis=1) - np.quantile(centered, .25, axis=1),
        np.quantile(centered, .99, axis=1) - np.quantile(centered, .01, axis=1),
        np.sqrt(np.mean(diff ** 2, axis=1)),
        bands,
    ])
    return per_lead.ravel().astype(np.float32)


def _frame(dataset, rows, provenance):
    metadata, features = zip(*rows)
    frame = pd.DataFrame(metadata)
    matrix = np.stack(features)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{dataset}: non-finite feature matrix")
    return {"dataset": dataset, "metadata": frame, "features": matrix,
            "provenance": provenance}


def load_ptbxl(maximum=240):
    root = DATA / "ptb_xl"
    meta = pd.read_csv(root / "ptbxl_database.csv")
    ids = deterministic_subset(meta.ecg_id.tolist(), maximum, "ptbxl")
    chosen = meta.set_index("ecg_id").loc[ids].reset_index()
    rows = []
    for row in chosen.itertuples(index=False):
        x, fields = wfdb.rdsamp(str(root / row.filename_hr))
        split = "train" if row.strat_fold <= 8 else ("validation" if row.strat_fold == 9 else "test")
        rows.append(({"record": pseudonym("PTBXL", row.ecg_id), "group": split}, waveform_features(x)))
    return _frame("PTB-XL", rows, {"source": "ptb_xl/ptbxl_database.csv + filename_hr", "sample": len(rows), "selection": "deterministic hash sample", "group": "official fold-derived split"})


def load_echonext(maximum=300):
    root = DATA / "echonext"
    meta = pd.read_csv(root / "echonext_metadata_100k.csv")
    test = meta.loc[meta.split == "test"].reset_index(drop=True)
    wave = np.load(root / "EchoNext_test_waveforms.npy", mmap_mode="r")
    if len(test) != len(wave):
        raise ValueError("EchoNext test metadata/waveform count mismatch")
    idx = deterministic_subset(range(len(test)), maximum, "echonext-test")
    endpoint = "shd_moderate_or_greater_flag"
    rows = []
    for i in idx:
        row = test.iloc[i]
        group = "SHD positive" if endpoint in test and row[endpoint] == 1 else "SHD negative"
        rows.append(({"record": pseudonym("EchoNext", row.ecg_key), "group": group},
                     waveform_features(np.asarray(wave[i, 0]).T)))
    return _frame("EchoNext", rows, {"source": "echonext/EchoNext_test_waveforms.npy", "sample": len(rows), "selection": "deterministic test-index hash sample", "group": "released composite SHD endpoint"})


def _ludb_group(header: Path):
    text = header.read_text(errors="replace").lower()
    return "atrial fibrillation" if "atrial fibrillation" in text else "other diagnosis/rhythm"


def load_ludb(maximum=200):
    root = DATA / "ludb"
    ids = deterministic_subset([p.stem for p in root.glob("*.hea")], maximum, "ludb")
    rows = []
    for rid in ids:
        record = wfdb.rdrecord(str(root / rid), physical=True)
        rows.append(({"record": pseudonym("LUDB", rid), "group": _ludb_group(root / f"{rid}.hea")},
                     waveform_features(record.p_signal)))
    return _frame("LUDB", rows, {"source": "ludb/*.hea + *.dat", "sample": len(rows), "selection": "all records" if len(rows) < maximum else "deterministic record hash sample", "group": "header statement contains atrial fibrillation"})


def load_isp(maximum=240):
    root = DATA / "isp_delineation_dataset"
    candidates = [(split, p.stem) for split in ("train", "test") for p in (root / f"{split}_data").glob("*.hea")]
    candidates = deterministic_subset(candidates, maximum, "isp")
    rows = []
    for split, rid in candidates:
        x, _ = wfdb.rdsamp(str(root / f"{split}_data" / rid))
        rows.append(({"record": pseudonym("ISP", f"{split}/{rid}"), "group": split}, waveform_features(x)))
    return _frame("ISP", rows, {"source": "isp_delineation_dataset/{train,test}_data", "sample": len(rows), "selection": "deterministic split/path hash sample", "group": "provided local split; patient independence unverified"})


def load_sunnybrook():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.evaluate_sunnybrook_registry import load_record
    paths = sorted((DATA / "sunnybrook_12_lead_ecg_samples").glob("*.xml"))
    rows = []
    for path in paths:
        record = load_record(path)
        peak = float(np.max(np.abs(record["signal"])))
        rows.append(({"record": pseudonym("Sunnybrook", record["record_id"]),
                      "group": "peak >5 mV QC outlier" if peak > 5 else "peak <=5 mV"},
                     waveform_features(record["signal"])))
    return _frame("Sunnybrook", rows, {"source": "sunnybrook_12_lead_ecg_samples/*.xml", "sample": len(rows), "selection": "all records", "group": "derived amplitude-QC stratum; not diagnosis"})


def load_zhejiang(maximum=220):
    root = DATA / "zhejiang"
    ids = sorted({p.stem.rsplit("_", 1)[0] for p in (root / "ecg").glob("*.pkl")})
    ids = deterministic_subset(ids, maximum, "zhejiang")
    rows = []
    for rid in ids:
        signals = []
        for lead in LEADS:
            with (root / "ecg" / f"{rid}_{lead}.pkl").open("rb") as handle:
                signals.append(np.asarray(pickle.load(handle), dtype=np.float32))
        with (root / "label" / f"{rid}.pkl").open("rb") as handle:
            mask = np.asarray(pickle.load(handle))
        qrs_fraction = float(np.mean(mask == 2))
        rows.append(({"record": pseudonym("Zhejiang", rid), "qrs_fraction": qrs_fraction},
                     waveform_features(np.stack(signals))))
    fractions = np.asarray([r[0]["qrs_fraction"] for r in rows])
    bins = pd.qcut(fractions, q=min(4, len(np.unique(fractions))), duplicates="drop")
    for (metadata, _), interval in zip(rows, bins.astype(str)):
        metadata["group"] = f"QRS-mask fraction {interval}"
    return _frame("Zhejiang", rows, {"source": "zhejiang/ecg + paired label masks", "sample": len(rows), "selection": "deterministic record hash sample", "group": "derived QRS-mask-coverage quartile; clinical provenance unresolved"})


def load_rdb(maximum=300):
    root = DATA / "rdb_wavelet_delineation_cache"
    paths = [(split, p) for split in ("train", "val", "test") for p in (root / split).glob("*.pt")]
    paths = deterministic_subset(paths, maximum, "rdb-cache")
    rows = []
    for split, path in paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
        rows.append(({"record": pseudonym("RDB", item["record_id"]),
                      "group": str(item["canonical_rhythm"]), "split": split},
                     waveform_features(item["waveform"].numpy())))
    return _frame("RDB", rows, {"source": "rdb_wavelet_delineation_cache/{train,val,test}/*.pt", "sample": len(rows), "selection": "deterministic split/path hash sample", "group": "released rhythm mapped to canonical rhythm"})


LOADERS = (load_ptbxl, load_echonext, load_ludb, load_isp, load_sunnybrook, load_zhejiang, load_rdb)


def embed_dataset(bundle):
    matrix = RobustScaler(quantile_range=(10, 90)).fit_transform(bundle["features"])
    n = len(matrix)
    neighbors = min(15, max(2, n - 1))
    projections = []
    for seed in SEEDS:
        reducer = umap.UMAP(n_neighbors=neighbors, min_dist=0.15, metric="cosine", random_state=seed, n_jobs=1)
        projections.append(reducer.fit_transform(matrix))
    primary = projections[0]
    frame = bundle["metadata"].copy()
    frame["umap_1"], frame["umap_2"] = primary[:, 0], primary[:, 1]
    upper = np.triu_indices(n, 1)
    stability = [spearmanr(pdist(primary), pdist(other)).statistic for other in projections[1:]]
    diagnostics = {
        **bundle["provenance"], "dataset": bundle["dataset"], "records_embedded": n,
        "features": matrix.shape[1], "n_neighbors": neighbors, "min_dist": 0.15,
        "metric": "cosine", "primary_seed": SEEDS[0], "sensitivity_seeds": str(SEEDS[1:]),
        "trustworthiness_k5": trustworthiness(matrix, primary, n_neighbors=min(5, n // 2 - 1)),
        "mean_pairwise_distance_rank_stability": float(np.mean(stability)),
    }
    return frame, diagnostics


def build_all():
    outputs, diagnostics = {}, []
    for loader in LOADERS:
        bundle = loader()
        frame, diag = embed_dataset(bundle)
        outputs[bundle["dataset"]] = frame
        diagnostics.append(diag)
    return outputs, pd.DataFrame(diagnostics)
