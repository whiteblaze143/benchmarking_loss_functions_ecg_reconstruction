"""Bounded, provenance-aware waveform UMAPs for every local real ECG cohort."""
from __future__ import annotations

import ast
import hashlib
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import umap
import wfdb
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SEEDS = (42, 7, 101)
PURPOSES = {
    "PTB-XL": ("Do official splits occupy comparable waveform-feature support?", "global RMS (mV)"),
    "EchoNext": ("Does the released SHD endpoint align with broad ECG phenotype?", "LVEF (%)"),
    "LUDB": ("Is atrial fibrillation distinguishable without erasing other morphology?", "age (years)"),
    "ISP": ("Does the provided test split shift relative to training?", "annotated intervals / record"),
    "Sunnybrook": ("Are extreme physical amplitudes isolated QC cases?", "maximum |amplitude| (mV)"),
    "Zhejiang": ("Does dense QRS-mask burden track waveform phenotype?", "QRS mask coverage (%)"),
    "RDB": ("Do rhythm-diverse records span distinct waveform regimes?", "P-mask coverage (%)"),
}
GROUP_ORDERS = {
    "PTB-XL": ["train", "validation", "test"],
    "EchoNext": ["SHD negative", "SHD positive"],
    "LUDB": ["other", "AF header"],
    "ISP": ["train", "test"],
    "Sunnybrook": ["peak <=5 mV", "peak >5 mV"],
    "Zhejiang": ["Q1 lowest", "Q2", "Q3", "Q4 highest"],
    "RDB": ["SR", "SB", "ST", "SA", "AF", "AFIB", "AT", "SVT"],
}


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
        rows.append(({"record": pseudonym("PTBXL", row.ecg_id), "group": split,
                      "metric_value": float(np.sqrt(np.mean(x ** 2)))}, waveform_features(x)))
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
        rows.append(({"record": pseudonym("EchoNext", row.ecg_key), "group": group,
                      "metric_value": float(row.get("lvef_value", np.nan))},
                     waveform_features(np.asarray(wave[i, 0]).T)))
    return _frame("EchoNext", rows, {"source": "echonext/EchoNext_test_waveforms.npy", "sample": len(rows), "selection": "deterministic test-index hash sample", "group": "released composite SHD endpoint"})


def _ludb_header(header: Path):
    text = header.read_text(errors="replace").lower()
    age = re.search(r"#<age>:\s*(\d+)", text)
    return ("AF header" if "atrial fibrillation" in text else "other",
            float(age.group(1)) if age else np.nan)


def load_ludb(maximum=200):
    root = DATA / "ludb"
    ids = deterministic_subset([p.stem for p in root.glob("*.hea")], maximum, "ludb")
    rows = []
    for rid in ids:
        record = wfdb.rdrecord(str(root / rid), physical=True)
        group, age = _ludb_header(root / f"{rid}.hea")
        rows.append(({"record": pseudonym("LUDB", rid), "group": group, "metric_value": age},
                     waveform_features(record.p_signal)))
    return _frame("LUDB", rows, {"source": "ludb/*.hea + *.dat", "sample": len(rows), "selection": "all records" if len(rows) < maximum else "deterministic record hash sample", "group": "header statement contains atrial fibrillation"})


def load_isp(maximum=240):
    root = DATA / "isp_delineation_dataset"
    candidates = [(split, p.stem) for split in ("train", "test") for p in (root / f"{split}_data").glob("*.hea")]
    candidates = deterministic_subset(candidates, maximum, "isp")
    interval_counts = {}
    for split in ("train", "test"):
        table = pd.read_csv(root / f"{split}_isp_delineation_data.csv")
        interval_counts.update({(split, str(row.file_name)): len(ast.literal_eval(row.target))
                                for row in table.itertuples(index=False)})
    rows = []
    for split, rid in candidates:
        x, _ = wfdb.rdsamp(str(root / f"{split}_data" / rid))
        rows.append(({"record": pseudonym("ISP", f"{split}/{rid}"), "group": split,
                      "metric_value": float(interval_counts[(split, rid)])}, waveform_features(x)))
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
                      "group": "peak >5 mV" if peak > 5 else "peak <=5 mV",
                      "metric_value": peak},
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
        rows.append(({"record": pseudonym("Zhejiang", rid), "qrs_fraction": qrs_fraction,
                      "metric_value": 100 * qrs_fraction},
                     waveform_features(np.stack(signals))))
    fractions = np.asarray([r[0]["qrs_fraction"] for r in rows])
    bins = pd.qcut(fractions, q=4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
    for (metadata, _), interval in zip(rows, bins.astype(str)):
        metadata["group"] = interval
    return _frame("Zhejiang", rows, {"source": "zhejiang/ecg + paired label masks", "sample": len(rows), "selection": "deterministic record hash sample", "group": "derived QRS-mask-coverage quartile; clinical provenance unresolved"})


def load_rdb(maximum=300):
    root = DATA / "rdb_wavelet_delineation_cache"
    paths = [(split, p) for split in ("train", "val", "test") for p in (root / split).glob("*.pt")]
    paths = deterministic_subset(paths, maximum, "rdb-cache")
    rows = []
    for split, path in paths:
        item = torch.load(path, map_location="cpu", weights_only=False)
        segmentation = item["segmentation"].numpy()
        valid = item["seg_valid"].numpy()
        p_fraction = 100 * float(np.mean(segmentation[valid] == 1)) if valid.any() else np.nan
        rows.append(({"record": pseudonym("RDB", item["record_id"]),
                      "group": str(item["canonical_rhythm"]), "split": split,
                      "metric_value": p_fraction},
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
    labels = frame["group"].astype(str).to_numpy()
    k = min(10, n - 1)
    original_neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(matrix).kneighbors(return_distance=False)[:, 1:]
    projected_neighbors = NearestNeighbors(n_neighbors=k + 1).fit(primary).kneighbors(return_distance=False)[:, 1:]
    original_agreement = float(np.mean(labels[original_neighbors] == labels[:, None]))
    projected_agreement = float(np.mean(labels[projected_neighbors] == labels[:, None]))
    counts = pd.Series(labels).value_counts().to_numpy()
    chance = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
    enough_groups = len(counts) > 1 and np.all(counts > 1)
    diagnostics = {
        **bundle["provenance"], "dataset": bundle["dataset"], "records_embedded": n,
        "features": matrix.shape[1], "n_neighbors": neighbors, "min_dist": 0.15,
        "metric": "cosine", "primary_seed": SEEDS[0], "sensitivity_seeds": str(SEEDS[1:]),
        "trustworthiness_k5": trustworthiness(matrix, primary, n_neighbors=min(5, n // 2 - 1)),
        "mean_pairwise_distance_rank_stability": float(np.mean(stability)),
        "original_knn_same_group": original_agreement,
        "umap_knn_same_group": projected_agreement,
        "imbalance_chance_same_group": chance,
        "original_silhouette": float(silhouette_score(matrix, labels, metric="cosine")) if enough_groups else np.nan,
        "umap_silhouette": float(silhouette_score(primary, labels)) if enough_groups else np.nan,
    }
    return frame, diagnostics


def curated_figure(dataset, frame, diagnostics):
    """Three-panel figure: projection, anti-overinterpretation check, raw endpoint."""
    purpose, metric_label = PURPOSES[dataset]
    present = set(frame["group"].astype(str))
    groups = [x for x in GROUP_ORDERS.get(dataset, []) if x in present]
    groups += sorted(present - set(groups))
    cmap = plt.get_cmap("tab10")
    colors = {group: cmap(i % 10) for i, group in enumerate(groups)}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8),
                             gridspec_kw={"width_ratios": [1.35, .9, 1.2]})
    fig.subplots_adjust(top=.76, bottom=.24, left=.055, right=.985, wspace=.31)
    ax = axes[0]
    for group in groups:
        subset = frame[frame.group.astype(str) == group]
        ax.scatter(subset.umap_1, subset.umap_2, s=26, alpha=.78, color=colors[group],
                   edgecolor="white", linewidth=.25, label=f"{group} (n={len(subset)})")
    ax.set(title="A  Waveform-feature UMAP", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(fontsize=7, frameon=False, loc="best")

    ax = axes[1]
    names = ["chance\nfrom imbalance", "original\n96-D kNN", "UMAP\n2-D kNN"]
    values = [diagnostics["imbalance_chance_same_group"], diagnostics["original_knn_same_group"], diagnostics["umap_knn_same_group"]]
    bars = ax.bar(names, values, color=["#B9B9B9", "#345995", "#E45756"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .025, f"{value:.2f}", ha="center", fontsize=9)
    ax.set(ylim=(0, 1.08), ylabel="same-group fraction",
           title="B  Does UMAP exaggerate grouping?")
    ax.axhline(values[0], color="#666", linestyle="--", linewidth=.8)

    ax = axes[2]
    positions = np.arange(len(groups))
    arrays = [frame.loc[frame.group.astype(str) == group, "metric_value"].dropna().to_numpy() for group in groups]
    boxed = [(pos, values) for pos, values in zip(positions, arrays) if len(values) >= 5]
    if boxed:
        ax.boxplot([x[1] for x in boxed], positions=[x[0] for x in boxed], widths=.6,
                   patch_artist=True, showfliers=False,
                   boxprops={"facecolor": "#D9E6F2", "edgecolor": "#345995"},
                   medianprops={"color": "#C23B22", "linewidth": 1.5})
    rng = np.random.default_rng(20260824)
    for pos, values, group in zip(positions, arrays, groups):
        jitter = rng.uniform(-.18, .18, len(values))
        ax.scatter(pos + jitter, values, s=10, alpha=.35, color=colors[group], edgecolor="none")
        if len(values) < 5 and len(values):
            ax.hlines(np.median(values), pos - .25, pos + .25, color="#C23B22", linewidth=1.5)
    ax.set_xticks(positions, groups, rotation=28, ha="right", fontsize=8)
    ax.set(ylabel=metric_label, title="C  Purpose-linked raw measurement")
    if dataset == "PTB-XL":
        ax.set_yscale("log")
        ax.set_ylabel(metric_label + ", log scale")
    fig.suptitle(f"{dataset}: {purpose}\n"
                 f"projection diagnostics — T(5)={diagnostics['trustworthiness_k5']:.3f}; "
                 f"seed stability={diagnostics['mean_pairwise_distance_rank_stability']:.3f}; "
                 f"96-D silhouette={diagnostics['original_silhouette']:.3f}",
                 y=.97, fontsize=13, fontweight="bold")
    return fig


def build_all():
    outputs, diagnostics = {}, []
    for loader in LOADERS:
        bundle = loader()
        frame, diag = embed_dataset(bundle)
        outputs[bundle["dataset"]] = frame
        diagnostics.append(diag)
    return outputs, pd.DataFrame(diagnostics)
