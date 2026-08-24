"""Small, read-only building blocks for the live experiment book.

Nothing in this module copies a result table or writes to an experiment DB.  A
Quarto render therefore acts as a fresh snapshot of whatever is currently on
disk.  Functions return empty frames when an optional source has not landed yet.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for candidate in (p, *p.parents):
        if (candidate / "results").exists() and (candidate / "book").exists():
            return candidate
    raise RuntimeError("Could not locate the project root")


ROOT = project_root()
SNAPSHOT_ROOT = Path(os.environ["BOOK_SNAPSHOT_ROOT"]).resolve() if os.environ.get("BOOK_SNAPSHOT_ROOT") else None

def source_path(relative: str | Path) -> Path:
    """Resolve a live source through the immutable render snapshot when set."""
    relative=Path(relative)
    candidate=(SNAPSHOT_ROOT/relative) if SNAPSHOT_ROOT else (ROOT/relative)
    return candidate


def read_sql(path: str | Path, query: str, params=()) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=30) as con:
        return pd.read_sql_query(query, con, params=params)


def tables(path: str | Path) -> dict[str, int]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=30) as con:
        names = [r[0] for r in con.execute(
            "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
        )]
        return {name: con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] for name in names}


def source_inventory(paths: dict[str, str | Path]) -> pd.DataFrame:
    rows = []
    for label, raw in paths.items():
        p = Path(raw)
        stat = p.stat() if p.exists() else None
        counts = tables(p) if stat else {}
        rows.append({
            "source": label,
            "path": str(p.relative_to(ROOT)) if p.is_absolute() and ROOT in p.parents else str(p),
            "exists": bool(stat),
            "size_mb": round(stat.st_size / 2**20, 2) if stat else np.nan,
            "updated": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds") if stat else None,
            "rows": sum(counts.values()),
            "tables": ", ".join(f"{k}:{v:,}" for k, v in counts.items()),
        })
    return pd.DataFrame(rows)


def json_normalize(series: pd.Series, prefix="metric.") -> pd.DataFrame:
    def decode(v):
        try:
            return json.loads(v) if v else {}
        except (TypeError, json.JSONDecodeError):
            return {}
    return pd.json_normalize([decode(v) for v in series]).add_prefix(prefix)


def metric_direction(name: str) -> str | None:
    n = name.lower()
    if any(x in n for x in ("loss", "mse", "mae", "rmse", "error", "p95_abs", "duration")):
        return "min"
    if any(x in n for x in ("pearson", "iou", "f1", "auroc", "auprc", "r2", "retention", "sens", "spec", "ppv", "npv")):
        return "max"
    return None


def numeric_metrics(df: pd.DataFrame, prefix: str | None = None) -> list[str]:
    cols = [c for c in df if pd.api.types.is_numeric_dtype(df[c]) and metric_direction(c)]
    return [c for c in cols if prefix is None or c.startswith(prefix)]


def leaderboard(df: pd.DataFrame, metrics: list[str], id_col="model_id", group_cols=(), k=5) -> pd.DataFrame:
    rows = []
    groups = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]
    for key, group in groups:
        key = key if isinstance(key, tuple) else (key,)
        for metric in metrics:
            if metric not in group:
                continue
            valid = group[group[metric].notna()]
            if valid.empty:
                continue
            direction = metric_direction(metric)
            chosen = valid.nsmallest(k, metric) if direction == "min" else valid.nlargest(k, metric)
            for rank, (_, row) in enumerate(chosen.iterrows(), 1):
                item = {"metric": metric, "direction": direction, "rank": rank,
                        "model_id": row.get(id_col), "value": row[metric]}
                item.update(dict(zip(group_cols, key)))
                rows.append(item)
    return pd.DataFrame(rows)


def model_metric_matrix(df: pd.DataFrame, id_col="model_id", metrics: list[str] | None = None) -> pd.DataFrame:
    metrics = metrics or numeric_metrics(df)
    metrics = [m for m in metrics if m in df and df[m].notna().sum() >= 3]
    if not metrics or id_col not in df:
        return pd.DataFrame()
    x = df.groupby(id_col, dropna=False)[metrics].mean(numeric_only=True)
    return x.dropna(axis=1, how="all")


def performance_embedding(matrix: pd.DataFrame, seed=42) -> tuple[pd.DataFrame, str]:
    """Embed standardized model metrics; this is not an encoder-latent embedding."""
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return pd.DataFrame(), "insufficient data"
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(matrix))
    method = "PCA"
    if len(matrix) >= 5:
        try:
            import umap
            n_neighbors = min(15, max(2, len(matrix) - 1))
            xy = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=.15,
                           metric="euclidean", random_state=seed).fit_transform(x)
            method = f"UMAP (n_neighbors={n_neighbors})"
        except Exception:
            xy = None
    else:
        xy = None
    if xy is None:
        from sklearn.decomposition import PCA
        xy = PCA(n_components=2, random_state=seed).fit_transform(x)
    out = pd.DataFrame(xy, index=matrix.index, columns=["axis_1", "axis_2"]).reset_index()
    return out, method


def load_blinded(path: str | Path, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = read_sql(path, "select * from evaluations")
    bd = read_sql(path, "select * from boundary_summaries")
    if not ev.empty: ev["cohort"] = cohort
    if not bd.empty: bd["cohort"] = cohort
    return ev, bd


def load_oracle(path: str | Path, cohort: str) -> pd.DataFrame:
    ev = read_sql(path, "select * from evaluations")
    if ev.empty:
        return ev
    if "primary_summary_json" in ev:
        ev = pd.concat([ev.drop(columns="primary_summary_json"), json_normalize(ev.primary_summary_json)], axis=1)
    ev["cohort"] = cohort
    return ev


def load_onelead_queue(path: str | Path) -> pd.DataFrame:
    q = read_sql(path, "select * from jobs order by ordinal")
    if q.empty:
        return q
    parts = [q.drop(columns=[c for c in ("cell_json", "summary_json") if c in q])]
    if "cell_json" in q: parts.append(json_normalize(q.cell_json, "cell."))
    if "summary_json" in q: parts.append(json_normalize(q.summary_json, "metric."))
    q = pd.concat(parts, axis=1)
    q["cell_name"] = q.get("cell.name", q["id"])
    q["seed"] = pd.to_numeric(q.id.str.extract(r"_s(\d+)_")[0], errors="coerce")
    q["observed_lead"] = q.id.str.extract(r"_l([01])$")[0].map({"0": "I", "1": "II"})
    return q


def matched_deltas(df: pd.DataFrame, baseline="A0_raw") -> pd.DataFrame:
    if df.empty: return df.copy()
    metrics = numeric_metrics(df, "metric.")
    keys = [c for c in ("seed", "observed_lead") if c in df]
    base = df[df.cell_name.eq(baseline)][keys + metrics].rename(columns={m: f"base.{m}" for m in metrics})
    out = df.merge(base, on=keys, how="left")
    for m in metrics:
        sign = -1 if metric_direction(m) == "min" else 1
        out[f"improvement.{m}"] = sign * (out[m] - out[f"base.{m}"])
    return out


def short_digest(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()[:12]
