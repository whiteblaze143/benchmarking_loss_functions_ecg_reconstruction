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
import re
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
    """Resolve a live source through the immutable render snapshot when set, falling back to ROOT."""
    relative = Path(relative)
    if SNAPSHOT_ROOT:
        candidate = SNAPSHOT_ROOT / relative
        if candidate.exists():
            return candidate
    return ROOT / relative


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
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=30) as con:
            names = [r[0] for r in con.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
            )]
            return {name: con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0] for name in names}
    except sqlite3.DatabaseError:
        # JSON queue states and other provenance files still belong in the
        # source inventory, but they do not have relational table counts.
        return {}


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


def load_json_queue(path: str | Path) -> pd.DataFrame:
    """Load a legacy JSON queue without treating a lock file as proof of life."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    payload = json.loads(path.read_text())
    jobs = payload.get("jobs", []) if isinstance(payload, dict) else payload
    frame = pd.json_normalize(jobs)
    if frame.empty:
        return frame
    frame["source_updated"] = datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    return frame


def load_summary_tree(relative: str | Path) -> pd.DataFrame:
    """Read one summary.json per run and expose completion evidence explicitly."""
    root = source_path(relative)
    rows = []
    if not root.exists():
        return pd.DataFrame()
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = summary_path.parent
        run_name = summary.get("run_name", run_dir.name)
        expected = re.search(r"conv(\d+)e_", run_name)
        expected_epochs = int(expected.group(1)) if expected else None
        completed_epochs = summary.get("epochs_completed", summary.get("epoch"))
        success_file = run_dir / "_SUCCESS.json"
        row = {
            "run_name": run_name,
            "summary_path": str(summary_path.relative_to(ROOT)),
            "summary_updated": datetime.fromtimestamp(summary_path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "expected_epochs": expected_epochs,
            "success_marker": success_file.exists(),
            "run_state": "completed" if success_file.exists() or (
                expected_epochs is not None and completed_epochs is not None and completed_epochs >= expected_epochs
            ) else "partial",
            "observed_lead": {"0": "I", "1": "II"}.get(
                (re.search(r"_l([01])$", run_name) or [None, None])[1]
            ),
        }
        row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def load_onelead_rdb(path: str | Path) -> dict[str, pd.DataFrame]:
    """Load every normalized table from the compact one-lead RDB evaluator."""
    return {
        name: read_sql(path, f'SELECT * FROM "{name}"')
        for name in (
            "evaluations", "boundary_summaries", "region_summaries",
            "signal_summaries", "thresholds", "screening_decisions",
        )
    }


def load_spatial_training_logs(relative: str | Path) -> pd.DataFrame:
    """Recover the completed legacy spatial study's validation metrics."""
    root = source_path(relative)
    rows = []
    if not root.exists():
        return pd.DataFrame()
    pattern = re.compile(
        r"spatial_1lead_(?P<variant>.+)_(?P<mask>\d{7})_s(?P<seed>\d+)_l(?P<lead>[01])\.log$"
    )
    epoch_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|\s+Val Loss:\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+Val Missing Pearson:\s+(?P<pearson>[-+0-9.eE]+)"
    )
    best_pattern = re.compile(r"Best Val Missing Pearson:\s*([-+0-9.eE]+)")
    for log_path in sorted(root.glob("*.log")):
        match = pattern.match(log_path.name)
        if not match:
            continue
        text = log_path.read_text(errors="replace")
        epochs = list(epoch_pattern.finditer(text))
        completed = "Training Complete for" in text
        last = epochs[-1] if epochs else None
        best = best_pattern.findall(text)
        rows.append({
            "run_name": log_path.stem,
            "variant": match.group("variant"),
            "factorial_mask": match.group("mask"),
            "seed": int(match.group("seed")),
            "observed_lead": {"0": "I", "1": "II"}[match.group("lead")],
            "status": "completed" if completed else "incomplete",
            "epochs_logged": len(epochs),
            "final_epoch": int(last.group("epoch")) if last else np.nan,
            "final_val_loss": float(last.group("loss")) if last else np.nan,
            "final_val_missing_pearson": float(last.group("pearson")) if last else np.nan,
            "best_val_missing_pearson": float(best[-1]) if best else np.nan,
            "log_path": str(log_path.relative_to(ROOT)),
            "log_updated": datetime.fromtimestamp(log_path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        })
    return pd.DataFrame(rows)


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


def load_spatial_epoch_curves(relative: str | Path = "refine-logs/queue_spatial_1lead/jobs") -> pd.DataFrame:
    """Extract epoch-by-epoch validation loss and missing Pearson curves for all spatial runs."""
    root = source_path(relative)
    rows = []
    if not root.exists():
        return pd.DataFrame()
    pattern = re.compile(
        r"spatial_1lead_(?P<variant>.+)_(?P<mask>\d{7})_s(?P<seed>\d+)_l(?P<lead>[01])\.log$"
    )
    epoch_pattern = re.compile(
        r"Epoch\s+(?P<epoch>\d+)\s+\|\s+Val Loss:\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+Val Missing Pearson:\s+(?P<pearson>[-+0-9.eE]+)"
    )
    for log_path in sorted(root.glob("*.log")):
        match = pattern.match(log_path.name)
        if not match:
            continue
        text = log_path.read_text(errors="replace")
        for ep in epoch_pattern.finditer(text):
            rows.append({
                "run_name": log_path.stem,
                "variant": match.group("variant"),
                "factorial_mask": match.group("mask"),
                "seed": int(match.group("seed")),
                "observed_lead": {"0": "I", "1": "II"}[match.group("lead")],
                "epoch": int(ep.group("epoch")),
                "val_loss": float(ep.group("loss")),
                "val_missing_pearson": float(ep.group("pearson")),
            })
    return pd.DataFrame(rows)


def paired_hypothesis_tests(df: pd.DataFrame, candidate_variant: str, reference_variant: str,
                            metric: str = "best_val_missing_pearson",
                            match_keys: list[str] | None = None) -> dict:
    """Compute paired contrast statistics between two variants across matched cells."""
    import scipy.stats as stats
    match_keys = match_keys or ["factorial_mask", "observed_lead"]
    cand = df[df.variant == candidate_variant].set_index(match_keys)[metric]
    ref = df[df.variant == reference_variant].set_index(match_keys)[metric]
    shared = pd.concat([cand.rename("candidate"), ref.rename("reference")], axis=1).dropna()
    if len(shared) == 0:
        return {}
    diffs = shared["candidate"] - shared["reference"]
    n = len(diffs)
    mean_diff = float(diffs.mean())
    std_diff = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = std_diff / np.sqrt(n) if n > 0 else 0.0
    # Cohen's d for paired differences
    cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan
    # Hedges' g bias correction
    hedges_g = cohens_d * (1 - (3 / (4 * (n - 1) - 1))) if n > 1 and not np.isnan(cohens_d) else cohens_d
    # Paired t-test
    if n >= 2 and std_diff > 0:
        t_stat, p_val = stats.ttest_rel(shared["candidate"], shared["reference"])
    else:
        t_stat, p_val = np.nan, np.nan
    # Wilcoxon signed-rank test
    if n >= 5 and (diffs != 0).any():
        try:
            w_stat, w_pval = stats.wilcoxon(diffs, zero_method="pratt")
        except Exception:
            w_stat, w_pval = np.nan, np.nan
    else:
        w_stat, w_pval = np.nan, np.nan
    # Bootstrap 95% CI
    np.random.seed(42)
    boot_means = [np.random.choice(diffs, size=n, replace=True).mean() for _ in range(2000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "candidate": candidate_variant,
        "reference": reference_variant,
        "n_pairs": n,
        "mean_difference": mean_diff,
        "median_difference": float(diffs.median()),
        "std_difference": std_diff,
        "se": se,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "t_statistic": float(t_stat),
        "p_value_t": float(p_val),
        "wilcoxon_p": float(w_pval),
        "win_count": int((diffs > 0).sum()),
        "loss_count": int((diffs < 0).sum()),
        "tie_count": int((diffs == 0).sum()),
    }


def factorial_anova_3way(df: pd.DataFrame, value_col: str,
                         factors: list[str] = ("variant", "factorial_mask", "observed_lead")) -> pd.DataFrame:
    """Compute 3-way ANOVA decomposition with Sum of Squares, F-statistics, and partial eta-squared."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    formula = f"{value_col} ~ C({factors[0]}) + C({factors[1]}) + C({factors[2]}) + C({factors[0]}):C({factors[1]}) + C({factors[0]}):C({factors[2]}) + C({factors[1]}):C({factors[2]})"
    try:
        model = ols(formula, data=df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        ss_total = table["sum_sq"].sum()
        table["eta_sq"] = table["sum_sq"] / ss_total
        table["partial_eta_sq"] = table["sum_sq"] / (table["sum_sq"] + table.loc["Residual", "sum_sq"])
        return table.reset_index().rename(columns={"index": "source"})
    except Exception as exc:
        return pd.DataFrame([{"error": str(exc)}])


def pareto_frontier_2d(df: pd.DataFrame, x_col: str, y_col: str, x_max: bool = True, y_max: bool = True) -> pd.DataFrame:
    """Identify Pareto optimal points in 2D space."""
    data = df[[x_col, y_col]].copy().dropna()
    pts = data.values
    pareto_mask = np.ones(len(pts), dtype=bool)
    for i, p1 in enumerate(pts):
        if not pareto_mask[i]:
            continue
        for j, p2 in enumerate(pts):
            if i == j or not pareto_mask[j]:
                continue
            x_dom = (p2[0] >= p1[0] if x_max else p2[0] <= p1[0])
            y_dom = (p2[1] >= p1[1] if y_max else p2[1] <= p1[1])
            x_strict = (p2[0] > p1[0] if x_max else p2[0] < p1[0])
            y_strict = (p2[1] > p1[1] if y_max else p2[1] < p1[1])
            if (x_dom and y_dom) and (x_strict or y_strict):
                pareto_mask[i] = False
                break
    out = df.loc[data.index[pareto_mask]].copy()
    return out

