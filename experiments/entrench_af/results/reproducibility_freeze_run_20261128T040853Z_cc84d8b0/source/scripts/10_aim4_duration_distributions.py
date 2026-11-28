#!/usr/bin/env python3
"""
Aim 4: Duration Distributions (Power-Law vs Lognormal vs Weibull + Vuong + Clauset GoF)
Script: experiments/entrench_af/scripts/10_aim4_duration_distributions.py

Outputs (in <LATEST_RUN>/aim4/):
- table4_duration_distributions.csv
- ccdf_data.parquet
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

try:
    import yaml
except ImportError:
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SEED = 20260101


def get_run_context() -> Dict[str, Any]:
    try:
        import common  # type: ignore

        if hasattr(common, "get_run_context"):
            return common.get_run_context()
    except Exception:
        pass

    latest_ptr = BASE_DIR / "results" / "LATEST_RUN.txt"
    if not latest_ptr.exists():
        raise FileNotFoundError(f"LATEST_RUN.txt not found at {latest_ptr}")
    run_dir_str = latest_ptr.read_text().strip()
    run_dir = Path(run_dir_str)
    if not run_dir.is_absolute():
        run_dir = BASE_DIR / run_dir_str
        if not run_dir.exists():
            run_dir = BASE_DIR / "results" / run_dir_str

    def_path = BASE_DIR / "config" / "definitions.yaml"
    cfg: Dict[str, Any] = {}
    definitions_hash = "unknown"
    if def_path.exists():
        raw = def_path.read_bytes()
        definitions_hash = hashlib.sha256(raw).hexdigest()
        if yaml is not None:
            cfg = yaml.safe_load(raw.decode("utf-8")) or {}

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(BASE_DIR),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = "nogit"

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "run_dir": run_dir,
        "config": cfg,
        "git_hash": git_hash,
        "definitions_hash": definitions_hash,
        "run_timestamp": run_timestamp,
    }


def add_provenance(df: pd.DataFrame, ctx: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    df["git_hash"] = ctx["git_hash"]
    df["definitions_hash"] = ctx["definitions_hash"]
    df["run_timestamp"] = ctx["run_timestamp"]
    return df


# ---------------------------------------------------------------------------
# Clauset-Shalizi-Newman (2009) Continuous Power-Law MLE + KS + Bootstrap GoF
# ---------------------------------------------------------------------------
def powerlaw_mle(x: np.ndarray, xmin: float) -> Tuple[float, float]:
    """Continuous power-law MLE alpha and KS statistic for x >= xmin."""
    tail = x[x >= xmin]
    n = len(tail)
    if n < 10:
        return np.nan, np.inf
    alpha = 1.0 + n / float(np.sum(np.log(tail / xmin)))
    tail_sorted = np.sort(tail)
    emp_cdf = np.arange(1, n + 1, dtype=float) / n
    mod_cdf = 1.0 - (tail_sorted / xmin) ** (1.0 - alpha)
    ks = float(np.max(np.abs(emp_cdf - mod_cdf)))
    return float(alpha), ks


def estimate_clauset_xmin(x: np.ndarray, max_candidates: int = 120) -> Tuple[float, float, float, int]:
    """Find xmin minimizing KS distance (Clauset et al. 2009)."""
    xs = np.sort(x[x > 0])
    if len(xs) < 20:
        return float(xs[0]), np.nan, np.nan, len(xs)
    # Candidate xmins from unique values up to 80th percentile (leaving at least 20 tail points)
    uniq = np.unique(xs[:-20])
    if len(uniq) > max_candidates:
        idx = np.linspace(0, len(uniq) - 1, max_candidates, dtype=int)
        uniq = uniq[idx]

    best_ks = np.inf
    best_xmin = float(uniq[0])
    best_alpha = np.nan
    for cand in uniq:
        alpha, ks = powerlaw_mle(xs, float(cand))
        if ks < best_ks:
            best_ks = ks
            best_xmin = float(cand)
            best_alpha = alpha

    n_tail = int(np.sum(xs >= best_xmin))
    return best_xmin, best_alpha, best_ks, n_tail


def clauset_bootstrap_gof(
    x: np.ndarray,
    xmin: float,
    alpha: float,
    ks_obs: float,
    n_boot: int = 200,
    seed: int = SEED,
) -> float:
    """Semi-parametric Clauset bootstrap GoF p-value."""
    rng = np.random.default_rng(seed)
    xs = x[x > 0]
    n = len(xs)
    below = xs[xs < xmin]
    p_tail = float(np.mean(xs >= xmin))
    if n < 20 or not np.isfinite(alpha) or alpha <= 1.0:
        return float("nan")

    exceed = 0
    for _ in range(n_boot):
        n_t = int(rng.binomial(n, p_tail))
        n_b = n - n_t
        u = rng.uniform(1e-12, 1.0, size=max(1, n_t))
        sim_tail = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        if n_b > 0 and len(below) > 0:
            sim_below = rng.choice(below, size=n_b, replace=True)
            sim = np.concatenate([sim_below, sim_tail])
        else:
            sim = sim_tail
        _, _, ks_sim, _ = estimate_clauset_xmin(sim, max_candidates=40)
        if ks_sim >= ks_obs:
            exceed += 1
    return float(exceed / n_boot)


# ---------------------------------------------------------------------------
# Truncated (x >= xmin) Log-Likelihoods & Vuong Test
# ---------------------------------------------------------------------------
def loglik_powerlaw_vec(x: np.ndarray, xmin: float, alpha: float) -> np.ndarray:
    return np.log(alpha - 1.0) - np.log(xmin) - alpha * np.log(x / xmin)


def fit_trunc_lognormal(x: np.ndarray, xmin: float) -> Tuple[float, float, np.ndarray]:
    """MLE for left-truncated lognormal (x >= xmin)."""
    log_x = np.log(x)
    log_xmin = np.log(xmin)

    def nll(params: np.ndarray) -> float:
        mu, log_sig = params
        sig = np.exp(log_sig)
        z_min = (log_xmin - mu) / sig
        sf_min = stats.norm.sf(z_min)
        if sf_min <= 1e-300:
            return 1e15
        ll = -log_x - log_sig - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((log_x - mu) / sig) ** 2 - np.log(sf_min)
        return -float(np.sum(ll))

    init = np.array([float(np.mean(log_x)), np.log(max(float(np.std(log_x)), 0.1))])
    res = optimize.minimize(nll, init, method="BFGS")
    mu, sig = float(res.x[0]), float(np.exp(res.x[1]))
    sf_min = max(float(stats.norm.sf((log_xmin - mu) / sig)), 1e-300)
    ll_vec = -log_x - np.log(sig) - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((log_x - mu) / sig) ** 2 - np.log(sf_min)
    return mu, sig, ll_vec


def fit_trunc_weibull(x: np.ndarray, xmin: float) -> Tuple[float, float, np.ndarray]:
    """MLE for left-truncated Weibull (x >= xmin)."""
    log_x = np.log(x)
    log_xmin = np.log(xmin)

    def nll(params: np.ndarray) -> float:
        log_k, log_lam = params
        k = np.exp(log_k)
        z = np.clip(k * (log_x - log_lam), -40.0, 40.0)
        z_min = np.clip(k * (log_xmin - log_lam), -40.0, 40.0)
        ll = log_k - log_lam + (k - 1.0) * (log_x - log_lam) - np.exp(z) + np.exp(z_min)
        return -float(np.sum(ll))

    init = np.array([np.log(0.7), float(np.log(np.median(x)))])
    res = optimize.minimize(nll, init, method="BFGS")
    k, lam = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
    z = np.clip(k * (log_x - np.log(lam)), -40.0, 40.0)
    z_min = np.clip(k * (log_xmin - np.log(lam)), -40.0, 40.0)
    ll_vec = np.log(k) - np.log(lam) + (k - 1.0) * (log_x - np.log(lam)) - np.exp(z) + np.exp(z_min)
    return k, lam, ll_vec


def vuong_test(ll1: np.ndarray, ll2: np.ndarray) -> Tuple[float, float]:
    """Vuong likelihood-ratio test statistic R and two-sided p-value (positive R favors model 1)."""
    diff = ll1 - ll2
    n = len(diff)
    if n < 5:
        return float("nan"), float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return 0.0, 1.0
    R = float(np.sum(diff) / (sd * np.sqrt(n)))
    p = float(2.0 * stats.norm.sf(abs(R)))
    return R, p


def analyze_distribution(
    x: np.ndarray,
    dataset_label: str,
    cohort: str,
    filter_label: str,
    fmin: float = 0.0,
    n_boot: int = 200,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    xs = np.sort(x[np.isfinite(x) & (x > 0)])
    n_all = len(xs)
    if n_all < 25:
        return [], pd.DataFrame()

    rows_out: List[Dict[str, Any]] = []
    # Two explicit modes: (a) fixed_filter (exact filter-defined tail), (b) clauset_ks (KS-minimizing xmin >= fmin)
    for xmin_mode in ("fixed_filter", "clauset_ks"):
        if xmin_mode == "fixed_filter":
            xmin = float(fmin) if fmin > 0.0 else float(xs[0])
            alpha, ks = powerlaw_mle(xs, xmin)
            n_tail = int(np.sum(xs >= xmin))
        else:
            xmin, alpha, ks, n_tail = estimate_clauset_xmin(xs)

        tail = xs[xs >= xmin]
        if len(tail) < 10 or not np.isfinite(alpha) or alpha <= 1.0:
            continue

        ll_pl_vec = loglik_powerlaw_vec(tail, xmin, alpha)
        ln_mu, ln_sig, ll_ln_vec = fit_trunc_lognormal(tail, xmin)
        wb_k, wb_lam, ll_wb_vec = fit_trunc_weibull(tail, xmin)

        ll_pl = float(np.sum(ll_pl_vec))
        ll_ln = float(np.sum(ll_ln_vec))
        ll_wb = float(np.sum(ll_wb_vec))

        aic_pl = 2 * 1 - 2 * ll_pl
        aic_ln = 2 * 2 - 2 * ll_ln
        aic_wb = 2 * 2 - 2 * ll_wb
        bic_pl = 1 * np.log(n_tail) - 2 * ll_pl
        bic_ln = 2 * np.log(n_tail) - 2 * ll_ln
        bic_wb = 2 * np.log(n_tail) - 2 * ll_wb

        R_pl_ln, p_pl_ln = vuong_test(ll_pl_vec, ll_ln_vec)
        R_ln_wb, p_ln_wb = vuong_test(ll_ln_vec, ll_wb_vec)
        gof_p = clauset_bootstrap_gof(xs, xmin, alpha, ks, n_boot=n_boot, seed=SEED)

        model_selection_valid = bool(n_tail >= 50)
        if model_selection_valid:
            best_aic = min([("powerlaw", aic_pl), ("lognormal", aic_ln), ("weibull", aic_wb)], key=lambda z: z[1])[0]
        else:
            best_aic = "insufficient_tail_n<50"

        rows_out.append(
            {
                "dataset": dataset_label,
                "cohort": cohort,
                "filter": filter_label,
                "xmin_mode": xmin_mode,
                "n_all": n_all,
                "xmin_sec": xmin,
                "n_tail": n_tail,
                "model_selection_valid": model_selection_valid,
                "pl_alpha": alpha,
                "pl_ks": ks,
                "pl_gof_p": gof_p,
                "ln_mu": ln_mu,
                "ln_sigma": ln_sig,
                "wb_shape": wb_k,
                "wb_scale": wb_lam,
                "ll_powerlaw": ll_pl,
                "ll_lognormal": ll_ln,
                "ll_weibull": ll_wb,
                "aic_powerlaw": aic_pl,
                "aic_lognormal": aic_ln,
                "aic_weibull": aic_wb,
                "bic_powerlaw": bic_pl,
                "bic_lognormal": bic_ln,
                "bic_weibull": bic_wb,
                "best_aic_model": best_aic,
                "vuong_R_pl_vs_ln": R_pl_ln,
                "vuong_p_pl_vs_ln": p_pl_ln,
                "vuong_R_ln_vs_wb": R_ln_wb,
                "vuong_p_ln_vs_wb": p_ln_wb,
            }
        )

    # Subsample empirical CCDF (up to 300 log-spaced points) for plotting
    ccdf_y = 1.0 - np.arange(n_all, dtype=float) / n_all
    if n_all > 300:
        idx = np.unique(np.linspace(0, n_all - 1, 300, dtype=int))
        x_plot, y_plot = xs[idx], ccdf_y[idx]
    else:
        x_plot, y_plot = xs, ccdf_y

    ccdf_df = pd.DataFrame(
        {
            "dataset": dataset_label,
            "cohort": cohort,
            "filter": filter_label,
            "dur_sec": x_plot,
            "ccdf": y_plot,
        }
    )
    return rows_out, ccdf_df


def main() -> None:
    ctx = get_run_context()
    run_dir: Path = ctx["run_dir"]
    cfg: Dict[str, Any] = ctx["config"]

    stage2_dir = run_dir / "stage2"
    aim2_dir = run_dir / "aim2"
    out_dir = run_dir / "aim4"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = cfg.get("cohorts", ["IRIDIA", "SHDB", "LTAFDB"])
    runs_g0 = pd.read_parquet(stage2_dir / "runs_g0.parquet")
    uncens_g0 = runs_g0[~(runs_g0["boundary_censored_start"] | runs_g0["boundary_censored_end"])]

    results_rows: List[Dict[str, Any]] = []
    ccdf_frames: List[pd.DataFrame] = []

    filters = [("all", 0.0), ("ge_30s", 30.0), ("ge_360s", 360.0)]
    for coh in list(cohorts) + ["POOLED"]:
        sub_c = uncens_g0 if coh == "POOLED" else uncens_g0[uncens_g0["cohort"] == coh]
        for flabel, fmin in filters:
            vals = sub_c.loc[sub_c["dur_sec"] >= fmin, "dur_sec"].to_numpy(dtype=float)
            rows, ccdf = analyze_distribution(vals, "runs_g0", coh, flabel, fmin=fmin, n_boot=200)
            if rows:
                results_rows.extend(rows)
                ccdf_frames.append(ccdf)

    # Aim 2 Spitaler-reconstructed episodes
    recon_path = aim2_dir / "reconstructed_episodes.parquet"
    if recon_path.exists():
        recon_df = pd.read_parquet(recon_path)
        for coh in list(cohorts) + ["POOLED"]:
            sub_r = recon_df if coh == "POOLED" else recon_df[recon_df["cohort"] == coh]
            vals = sub_r["dur_sec"].to_numpy(dtype=float)
            rows, ccdf = analyze_distribution(vals, "spitaler_reconstructed", coh, "all", fmin=0.0, n_boot=200)
            if rows:
                results_rows.extend(rows)
                ccdf_frames.append(ccdf)

    res_df = add_provenance(pd.DataFrame(results_rows), ctx)
    res_df.to_csv(out_dir / "table4_duration_distributions.csv", index=False)

    ccdf_all = add_provenance(pd.concat(ccdf_frames, ignore_index=True) if ccdf_frames else pd.DataFrame(), ctx)
    ccdf_all.to_parquet(out_dir / "ccdf_data.parquet", index=False)

    print(f"[Aim 4] Saved table4 ({len(res_df)} rows) and ccdf_data ({len(ccdf_all)} rows) in {out_dir}")


if __name__ == "__main__":
    main()
