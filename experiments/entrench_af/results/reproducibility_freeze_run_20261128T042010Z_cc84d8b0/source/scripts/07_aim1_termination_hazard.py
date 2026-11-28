#!/usr/bin/env python3
"""
Aim 1: AF Termination Hazard & Frailty / Change-Point Analysis
Script: experiments/entrench_af/scripts/07_aim1_termination_hazard.py

Outputs (in <LATEST_RUN>/aim1/):
- table1_conditional_probs.csv
- tableS1_weibull_frailty.csv
- tableS2_changepoint_lrt.csv
- km_curves.parquet
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

try:
    import yaml
except ImportError:
    yaml = None

try:
    from lifelines import KaplanMeierFitter
except ImportError:
    KaplanMeierFitter = None

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
# Wilson CI & Patient-Cluster Bootstrap for Conditional Termination Prob
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _km_eval_fast(
    durs: np.ndarray,
    events: np.ndarray,
    pat_inv: np.ndarray,
    n_pats: int,
    boot_pat_counts: np.ndarray,
    query_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact Kaplan-Meier survival at `query_times` for point estimate and
    patient-cluster bootstrap resamples (`boot_pat_counts`: shape (n_boot, n_pats)).
    Returns (s_point: (Q,), s_boot: (n_boot, Q)).
    """
    ev_mask = events > 0.5
    ev_times = np.unique(durs[ev_mask])
    Q = len(query_times)
    n_boot = boot_pat_counts.shape[0]
    if len(ev_times) == 0:
        return np.ones(Q, dtype=float), np.ones((n_boot, Q), dtype=float)

    M = len(ev_times)
    # Build per-patient event counts and at-risk counts at each unique event time
    # Map each run's dur to the index of the last event time <= dur
    ev_idx = np.searchsorted(ev_times, durs, side="right") - 1  # in [-1, M-1]
    pat_d = np.zeros((n_pats, M), dtype=np.float64)
    # Events occur only at exact ev_times
    ev_run_indices = np.where(ev_mask)[0]
    ev_time_pos = np.searchsorted(ev_times, durs[ev_run_indices])
    np.add.at(pat_d, (pat_inv[ev_run_indices], ev_time_pos), 1.0)

    # At-risk at ev_times[j]: all runs with dur >= ev_times[j]
    # Equivalently, runs whose last ev_time <= dur is >= j (reverse cumsum of counts at ev_idx)
    pat_end_counts = np.zeros((n_pats, M), dtype=np.float64)
    valid_end = ev_idx >= 0
    np.add.at(pat_end_counts, (pat_inv[valid_end], ev_idx[valid_end]), 1.0)
    pat_y = np.cumsum(pat_end_counts[:, ::-1], axis=1)[:, ::-1]

    # Point estimate
    D_pt = pat_d.sum(axis=0)
    Y_pt = pat_y.sum(axis=0)
    haz_pt = np.divide(D_pt, Y_pt, out=np.zeros_like(D_pt), where=Y_pt > 0)
    S_pt_grid = np.cumprod(1.0 - haz_pt)

    # Query indices
    q_idx = np.searchsorted(ev_times, query_times, side="right") - 1
    s_point = np.where(q_idx >= 0, S_pt_grid[np.maximum(q_idx, 0)], 1.0)

    # Bootstrap in chunks to keep memory small
    s_boot = np.ones((n_boot, Q), dtype=float)
    chunk = 200
    for start in range(0, n_boot, chunk):
        end = min(n_boot, start + chunk)
        W = boot_pat_counts[start:end].astype(np.float64)
        D_b = W @ pat_d
        Y_b = W @ pat_y
        haz_b = np.divide(D_b, Y_b, out=np.zeros_like(D_b), where=Y_b > 0)
        S_b_grid = np.cumprod(1.0 - haz_b, axis=1)
        for qi in range(Q):
            if q_idx[qi] >= 0:
                s_boot[start:end, qi] = S_b_grid[:, q_idx[qi]]

    return s_point, s_boot


def filter_known_onset(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Exclude runs with unknown onset time (boundary_censored_start == True) from elapsed-duration analysis."""
    if df is None or len(df) == 0:
        return df.copy(), 0
    bc_start = df["boundary_censored_start"].astype(bool) if "boundary_censored_start" in df.columns else pd.Series(False, index=df.index)
    n_excl = int(bc_start.sum())
    return df.loc[~bc_start].copy(), n_excl


def compute_conditional_probs(
    df: pd.DataFrame,
    thresholds: List[float],
    windows: List[float],
    n_boot: int = 1000,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """
    Compute on known-onset runs (excluding boundary_censored_start == True):
    1. Raw uncensored P(t, w) = #{dur >= t and dur < t+w} / #{dur >= t}.
    2. KM-based P_KM(t, w) = 1 - S_KM(t + w) / S_KM(t) with right-censoring at end
       and 1000-resample patient-cluster bootstrap 95% CIs.
    3. Patient-weighted termination proportion at risk (equal weight per patient with >=1
       known-status episode at risk in [t, t+w)) plus total person-time exposure in [t, t+w).
    """
    df_known, _ = filter_known_onset(df)
    if len(df_known) == 0:
        return []

    all_durs = df_known["dur_sec"].to_numpy(dtype=float)
    all_events = (~df_known["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
    all_pats = df_known["patient"].to_numpy()
    all_cohorts = df_known["cohort"].to_numpy()

    unique_pats, all_pat_inv = np.unique(all_pats, return_inverse=True)
    n_pats = len(unique_pats)

    rng = np.random.default_rng(seed)
    boot_pat_counts = np.zeros((n_boot, n_pats), dtype=np.int32)
    for b in range(n_boot):
        idx = rng.integers(0, n_pats, size=n_pats)
        boot_pat_counts[b] = np.bincount(idx, minlength=n_pats)

    # Precompute KM survival at all needed query times
    q_set = sorted(set(float(t) for t in thresholds) | set(float(t + w) for t in thresholds for w in windows))
    q_arr = np.array(q_set, dtype=float)
    q_map = {val: i for i, val in enumerate(q_set)}
    s_pt, s_boot = _km_eval_fast(all_durs, all_events, all_pat_inv, n_pats, boot_pat_counts, q_arr)

    uncens_mask = all_events > 0.5
    durs = all_durs[uncens_mask]
    pat_inv = all_pat_inv[uncens_mask]

    def _patient_weighted_km_cond(t_lm: float, w_sec: float) -> Tuple[float, int]:
        """Exact patient-weighted KM conditional probability over risk set D >= t_lm (equal weight per patient at risk)."""
        rm = all_durs >= t_lm
        if not np.any(rm):
            return float("nan"), 0
        rd = all_durs[rm]
        re = all_events[rm]
        rp = all_pat_inv[rm]
        u_p, u_inv = np.unique(rp, return_inverse=True)
        n_pt_lm = int(len(u_p))
        ep_counts = np.bincount(u_inv)
        weights = 1.0 / ep_counts[u_inv]

        ev_in_win = (re > 0.5) & (rd > t_lm) & (rd <= t_lm + w_sec)
        # Include boundary equality rd == t_lm if termination can occur strictly on [t, t+w)
        ev_in_win = (re > 0.5) & (rd >= t_lm) & (rd < t_lm + w_sec)
        ev_times = np.unique(rd[ev_in_win])
        if len(ev_times) == 0:
            return 0.0, n_pt_lm
        surv = 1.0
        for et in ev_times:
            d_w = float(np.sum(weights[(re > 0.5) & (rd == et)]))
            y_w = float(np.sum(weights[rd >= et]))
            if y_w > 0:
                surv *= max(0.0, 1.0 - d_w / y_w)
        return float(np.clip(1.0 - surv, 0.0, 1.0)), n_pt_lm

    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        at_risk_all_mask = all_durs >= t
        n_risk_all = int(np.sum(at_risk_all_mask))
        if n_risk_all == 0:
            continue
        pat_risk_all = np.bincount(all_pat_inv[at_risk_all_mask], minlength=n_pats)
        n_patients_all_at_risk = int(np.sum(pat_risk_all > 0))

        at_risk_mask = durs >= t
        n_risk = int(np.sum(at_risk_mask))
        pat_risk = np.bincount(pat_inv[at_risk_mask], minlength=n_pats) if n_risk > 0 else np.zeros(n_pats, dtype=int)
        n_uncens_patients_at_risk = int(np.sum(pat_risk > 0))
        boot_risk = boot_pat_counts @ pat_risk

        iridia_only = bool(t >= 86400 and set(all_cohorts[at_risk_all_mask]).issubset({"IRIDIA"}))
        it = q_map[float(t)]
        st_pt = s_pt[it]
        st_boot = s_boot[:, it]

        for w in windows:
            if n_risk > 0:
                term_mask = at_risk_mask & (durs < t + w)
                n_term = int(np.sum(term_mask))
                p_hat = n_term / n_risk
                w_lo, w_hi = wilson_ci(n_term, n_risk)

                pat_term = np.bincount(pat_inv[term_mask], minlength=n_pats)
                boot_term = boot_pat_counts @ pat_term
                valid_b = boot_risk > 0
                if np.sum(valid_b) >= 10:
                    boot_p = boot_term[valid_b] / boot_risk[valid_b]
                    b_lo = float(np.percentile(boot_p, 2.5))
                    b_hi = float(np.percentile(boot_p, 97.5))
                else:
                    b_lo, b_hi = float("nan"), float("nan")
            else:
                n_term = 0
                p_hat, w_lo, w_hi, b_lo, b_hi = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

            itw = q_map[float(t + w)]
            if st_pt > 1e-12:
                p_km = float(np.clip(1.0 - s_pt[itw] / st_pt, 0.0, 1.0))
            else:
                p_km = float("nan")
            valid_km = st_boot > 1e-12
            if np.sum(valid_km) >= 10:
                km_bp = np.clip(1.0 - s_boot[valid_km, itw] / st_boot[valid_km], 0.0, 1.0)
                km_lo = float(np.percentile(km_bp, 2.5))
                km_hi = float(np.percentile(km_bp, 97.5))
            else:
                km_lo, km_hi = float("nan"), float("nan")

            # Exposure inside [t, t+w) in person-hours + exact landmark-consistent patient-weighted KM
            exp_sec = float(np.sum(np.clip(all_durs - t, 0.0, w)))
            p_pat_weighted, n_pat_pw = _patient_weighted_km_cond(t, w)
            assert n_pat_pw <= n_patients_all_at_risk, (
                f"Landmark denominator mismatch at t={t}, w={w}: n_pat_pw={n_pat_pw} > n_patients_at_risk={n_patients_all_at_risk}"
            )

            is_spitaler_t3 = bool(w == 86400 and t in {3600, 7200, 10800, 14400, 18000, 21600, 43200, 86400, 172800})
            rows.append(
                {
                    "threshold_sec": float(t),
                    "window_sec": float(w),
                    "n_at_risk": n_risk,
                    "n_at_risk_all": n_risk_all,
                    "n_terminated": n_term,
                    "exposure_hours": float(exp_sec / 3600.0),
                    "p_term_completed_only": p_hat,
                    "p_term": p_hat,
                    "wilson_ci_low": w_lo,
                    "wilson_ci_high": w_hi,
                    "boot_ci_low": b_lo,
                    "boot_ci_high": b_hi,
                    "p_term_km": p_km,
                    "km_boot_ci_low": km_lo,
                    "km_boot_ci_high": km_hi,
                    "p_term_patient_weighted": p_pat_weighted,
                    "n_patients_pw": n_pat_pw,
                    "n_patients_at_risk": n_patients_all_at_risk,
                    "n_uncens_patients_at_risk": n_uncens_patients_at_risk,
                    "iridia_only_ge24h": iridia_only,
                    "is_spitaler_table3": is_spitaler_t3,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Exact Shared-Gamma-Frailty Weibull MLE (patient-level frailty)
# ---------------------------------------------------------------------------
def fit_shared_gamma_frailty_weibull(
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    t_min: float = 0.0,
) -> Dict[str, Any]:
    """
    Exact marginal log-likelihood for left-truncated Weibull with patient-level
    shared gamma frailty:
      h_ij(t | u_i) = u_i * (k / lam) * (t / lam)^(k-1),   u_i ~ Gamma(1/theta, rate=1/theta)
    Conditional on T_ij > t_min, the integrated hazard increment is:
      dH_ij = (t_ij / lam)^k - (t_min / lam)^k  (for t_min > 0, else (t_ij / lam)^k).
    Marginal log-likelihood over clusters i = 1..M:
      L_i = sum_{j in i} d_ij * [log k - k log lam + (k-1) log t_ij]
            + log Gamma(1/theta + D_i) - log Gamma(1/theta)
            + D_i * log(theta)
            - (1/theta + D_i) * log(1 + theta * H_i)
    where D_i = sum_{j in i} d_ij and H_i = sum_{j in i} dH_ij.
    """
    t = np.asarray(durations, dtype=float)
    d = np.asarray(events, dtype=float)
    mask = t > max(0.0, float(t_min))
    t, d, pids = t[mask], d[mask], np.asarray(patient_ids)[mask]
    if len(t) < 10 or np.sum(d) < 5:
        return {
            "converged": False,
            "shape_k": np.nan,
            "shape_ci_low": np.nan,
            "shape_ci_high": np.nan,
            "scale_lam": np.nan,
            "frailty_theta": np.nan,
            "loglik": np.nan,
            "n_runs": int(len(t)),
            "n_events": int(np.sum(d)),
            "n_patients": int(len(np.unique(pids))),
        }

    # Rescale t by median for numerical stability
    t_scale = float(np.median(t))
    t_norm = t / t_scale
    log_t = np.log(t_norm)
    use_trunc = float(t_min) > 0.0
    log_tmin = float(np.log(float(t_min) / t_scale)) if use_trunc else 0.0

    _, pat_inv = np.unique(pids, return_inverse=True)
    n_pats = int(pat_inv.max() + 1)
    D_i = np.bincount(pat_inv, weights=d, minlength=n_pats)
    sum_d = float(np.sum(d))
    sum_d_logt = float(np.sum(d * log_t))

    def nll(params: np.ndarray) -> float:
        log_k, log_lam, log_theta = params
        if log_k < -5.0 or log_k > 3.0 or log_theta < -12.0 or log_theta > 6.0:
            return 1e15
        k = np.exp(log_k)
        theta = np.exp(log_theta)
        inv_theta = 1.0 / theta

        z = np.clip(k * (log_t - log_lam), -40.0, 40.0)
        if use_trunc:
            z_min = np.clip(k * (log_tmin - log_lam), -40.0, 40.0)
            h_ij = np.maximum(0.0, np.exp(z) - np.exp(z_min))
        else:
            h_ij = np.exp(z)
        H_i = np.bincount(pat_inv, weights=h_ij, minlength=n_pats)

        ll_base = sum_d * (log_k - k * log_lam) + (k - 1.0) * sum_d_logt
        ll_frailty = np.sum(
            special.gammaln(inv_theta + D_i)
            - special.gammaln(inv_theta)
            + D_i * log_theta
            - (inv_theta + D_i) * np.log1p(theta * H_i)
        )
        val = -(ll_base + ll_frailty)
        return float(val) if np.isfinite(val) else 1e15

    # Multi-start Nelder-Mead + BFGS polish for global MLE convergence
    best_res = None
    for k0 in (0.5, 0.8):
        for l0 in (-0.5, 0.5):
            init = np.array([np.log(k0), l0, np.log(0.5)])
            r_nm = optimize.minimize(nll, init, method="Nelder-Mead", options={"maxiter": 1500})
            r_bf = optimize.minimize(nll, r_nm.x, method="BFGS", options={"maxiter": 500})
            cand = r_bf if r_bf.fun <= r_nm.fun else r_nm
            if best_res is None or cand.fun < best_res.fun:
                best_res = cand
    res = best_res

    log_k_hat, log_lam_hat, log_theta_hat = res.x
    k_hat = float(np.exp(log_k_hat))
    lam_hat = float(np.exp(log_lam_hat) * t_scale)
    theta_hat = float(np.exp(log_theta_hat))
    ll_hat = float(-res.fun)

    # Numerical Hessian for Wald 95% CI on log_k
    eps = 1e-4
    H = np.zeros((3, 3), dtype=float)
    f0 = nll(res.x)
    for i in range(3):
        for j in range(i, 3):
            ei = np.zeros(3)
            ej = np.zeros(3)
            ei[i] = eps
            ej[j] = eps
            if i == j:
                H[i, i] = (nll(res.x + ei) - 2.0 * f0 + nll(res.x - ei)) / (eps * eps)
            else:
                H[i, j] = (
                    nll(res.x + ei + ej)
                    - nll(res.x + ei - ej)
                    - nll(res.x - ei + ej)
                    + nll(res.x - ei - ej)
                ) / (4.0 * eps * eps)
                H[j, i] = H[i, j]

    try:
        cov = np.linalg.inv(H)
        se_log_k = float(np.sqrt(max(1e-12, cov[0, 0])))
        k_lo = float(np.exp(log_k_hat - 1.959964 * se_log_k))
        k_hi = float(np.exp(log_k_hat + 1.959964 * se_log_k))
    except Exception:
        k_lo, k_hi = np.nan, np.nan

    return {
        "converged": bool(res.success or np.isfinite(ll_hat)),
        "shape_k": k_hat,
        "shape_ci_low": k_lo,
        "shape_ci_high": k_hi,
        "scale_lam": lam_hat,
        "log_lam_hat": float(log_lam_hat),
        "frailty_theta": theta_hat,
        "loglik": ll_hat,
        "at_boundary_k": bool(log_k_hat <= -4.95 or log_k_hat >= 2.95),
        "at_boundary_theta": bool(log_theta_hat <= -11.95 or log_theta_hat >= 5.95),
        "n_runs": int(len(t)),
        "n_events": int(np.sum(d)),
        "n_patients": n_pats,
    }


def verify_delayed_entry_frailty_likelihood(seed: int = SEED) -> pd.DataFrame:
    """
    Independently verify delayed-entry (left-truncated) shared-gamma-frailty Weibull
    likelihood under three synthetic data-generating mechanisms:
      1. Episode-conditional delayed entry (T_ij | T_ij > t_min, u_i) vs naive t_min=0 fit.
      2. Single-population survival selection (episodes from t=0 filtered to T > t_min + 24h admin).
      3. Two-subcohort mixture selection (fast LTAFDB-like + slow IRIDIA-like post-hoc 30s-24h).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    # Fixture A: exact conditional delayed-entry draws at t_min=300s
    k_a, lam_a, th_a, tmin_a = 0.72, 1200.0, 1.40, 300.0
    u_a = rng.gamma(shape=1.0 / th_a, scale=th_a, size=240)
    durs_a, evs_a, pids_a = [], [], []
    H_min_a = (tmin_a / lam_a) ** k_a
    for pi, u in enumerate(u_a):
        for _ in range(12):
            v = rng.exponential(1.0 / u)
            t_ev = lam_a * (H_min_a + v) ** (1.0 / k_a)
            c_t = tmin_a + rng.uniform(600.0, 25000.0)
            durs_a.append(min(t_ev, c_t))
            evs_a.append(1.0 if t_ev <= c_t else 0.0)
            pids_a.append(f"fA_{pi:03d}")
    da_arr, ea_arr, pa_arr = np.array(durs_a), np.array(evs_a), np.array(pids_a)
    fit_lt = fit_shared_gamma_frailty_weibull(da_arr, ea_arr, pa_arr, t_min=tmin_a)
    fit_naive = fit_shared_gamma_frailty_weibull(da_arr, ea_arr, pa_arr, t_min=0.0)
    rows.append({
        "fixture": "A_conditional_delayed_entry_lt300s",
        "true_k": k_a,
        "true_theta": th_a,
        "t_min_sec": tmin_a,
        "admin_censor_sec": np.nan,
        "n_episodes_retained": int(len(da_arr)),
        "retention_frac": 1.0,
        "n_patients_retained": 240,
        "recovered_k_lt": float(fit_lt["shape_k"]),
        "recovered_theta_lt": float(fit_lt["frailty_theta"]),
        "naive_untrunc_k": float(fit_naive["shape_k"]),
        "bias_k_lt": float(fit_lt["shape_k"] - k_a),
        "bias_k_naive": float(fit_naive["shape_k"] - k_a),
        "converged": bool(fit_lt["converged"]),
    })

    # Fixture B: Unconditional t=0 Gamma-frailty population filtered post-hoc to t > 30s & 24h admin
    k_b, lam_b, th_b, tmin_b, adm_b = 0.72, 900.0, 1.40, 30.0, 86400.0
    u_b = rng.gamma(shape=1.0 / th_b, scale=th_b, size=250)
    durs_b, evs_b, pids_b = [], [], []
    tot_b = 250 * 15
    for pi, u in enumerate(u_b):
        for _ in range(15):
            v = rng.exponential(1.0 / u)
            t_ev = lam_b * (v ** (1.0 / k_b))
            if t_ev > tmin_b:
                ev_flag = 1.0 if t_ev <= adm_b else 0.0
                durs_b.append(min(t_ev, adm_b))
                evs_b.append(ev_flag)
                pids_b.append(f"fB_{pi:03d}")
    db_arr, eb_arr, pb_arr = np.array(durs_b), np.array(evs_b), np.array(pids_b)
    fit_b = fit_shared_gamma_frailty_weibull(db_arr, eb_arr, pb_arr, t_min=tmin_b)
    rows.append({
        "fixture": "B_single_pop_survival_selection_30s_24h",
        "true_k": k_b,
        "true_theta": th_b,
        "t_min_sec": tmin_b,
        "admin_censor_sec": adm_b,
        "n_episodes_retained": int(len(db_arr)),
        "retention_frac": float(len(db_arr) / tot_b),
        "n_patients_retained": int(len(np.unique(pb_arr))),
        "recovered_k_lt": float(fit_b["shape_k"]),
        "recovered_theta_lt": float(fit_b["frailty_theta"]),
        "naive_untrunc_k": np.nan,
        "bias_k_lt": float(fit_b["shape_k"] - k_b),
        "bias_k_naive": np.nan,
        "converged": bool(fit_b["converged"]),
    })

    # Fixture C: Heterogeneous fast+slow cohort mixture under 30s left truncation + 24h admin censoring
    durs_c, evs_c, pids_c = [], [], []
    tot_c = 150 * 25 + 150 * 4
    # Fast LTAFDB-like subcohort
    u_c1 = rng.gamma(shape=1.0 / 1.2, scale=1.2, size=150)
    for pi, u in enumerate(u_c1):
        for _ in range(25):
            t_ev = 25.0 * (rng.exponential(1.0 / u) ** (1.0 / 0.60))
            if t_ev > 30.0:
                durs_c.append(min(t_ev, 86400.0))
                evs_c.append(1.0 if t_ev <= 86400.0 else 0.0)
                pids_c.append(f"fC1_{pi:03d}")
    # Slow IRIDIA-like subcohort
    u_c2 = rng.gamma(shape=1.0 / 1.1, scale=1.1, size=150)
    for pi, u in enumerate(u_c2):
        for _ in range(4):
            t_ev = 3200.0 * (rng.exponential(1.0 / u) ** (1.0 / 0.66))
            if t_ev > 30.0:
                durs_c.append(min(t_ev, 86400.0))
                evs_c.append(1.0 if t_ev <= 86400.0 else 0.0)
                pids_c.append(f"fC2_{pi:03d}")
    dc_arr, ec_arr, pc_arr = np.array(durs_c), np.array(evs_c), np.array(pids_c)
    fit_c = fit_nested_hazard_models(dc_arr, ec_arr, pc_arr, t_min=30.0)
    rows.append({
        "fixture": "C_heterogeneous_mixture_selection_30s_24h",
        "true_k": 0.63,
        "true_theta": np.nan,
        "t_min_sec": 30.0,
        "admin_censor_sec": 86400.0,
        "n_episodes_retained": int(len(dc_arr)),
        "retention_frac": float(len(dc_arr) / tot_c),
        "n_patients_retained": int(len(np.unique(pc_arr))),
        "recovered_k_lt": float(fit_c["frailty_wb_k"]),
        "recovered_theta_lt": float(fit_c["frailty_wb_theta"]),
        "naive_untrunc_k": float(fit_c["unfrailty_wb_k"]),
        "bias_k_lt": float(fit_c["frailty_wb_k"] - 0.63),
        "bias_k_naive": float(fit_c["unfrailty_wb_k"] - 0.63),
        "converged": bool(fit_c["converged"]),
    })
    return pd.DataFrame(rows)


def fit_nested_hazard_models(
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    t_min: float = 0.0,
) -> Dict[str, Any]:
    """
    Compare constant patient-specific hazard (Gamma frailty Exponential, k=1) vs
    duration-dependent hazard (Gamma frailty Weibull, k free), plus unfrailty baselines.
    """
    t = np.asarray(durations, dtype=float)
    d = np.asarray(events, dtype=float)
    mask = t > max(0.0, float(t_min))
    t, d, pids = t[mask], d[mask], np.asarray(patient_ids)[mask]
    if len(t) < 10 or np.sum(d) < 5:
        return {"converged": False, "n_runs": int(len(t)), "n_events": int(np.sum(d)), "n_patients": int(len(np.unique(pids)))}

    t_scale = float(np.median(t))
    t_norm = t / t_scale
    log_t = np.log(t_norm)
    use_trunc = float(t_min) > 0.0
    tmin_norm = float(t_min) / t_scale if use_trunc else 0.0
    dt_norm = np.maximum(1e-12, t_norm - tmin_norm)

    _, pat_inv = np.unique(pids, return_inverse=True)
    n_pats = int(pat_inv.max() + 1)
    D_i = np.bincount(pat_inv, weights=d, minlength=n_pats)
    sum_d = float(np.sum(d))
    sum_d_logt = float(np.sum(d * log_t))

    # 1. Unfrailty Exponential (k=1, theta=0) exact closed-form MLE
    total_exp = float(np.sum(dt_norm))
    lam_exp0 = total_exp / sum_d if sum_d > 0 else np.nan
    ll_unfrailty_exp = float(sum_d * np.log(1.0 / lam_exp0) - total_exp / lam_exp0) if lam_exp0 > 0 else np.nan

    # 2. Unfrailty Weibull (k free, theta=0)
    def nll_uw(p: np.ndarray) -> float:
        lk, llam = p
        if lk < -5.0 or lk > 3.0 or llam < -15.0 or llam > 15.0:
            return 1e15
        k = np.exp(lk)
        z = np.clip(k * (log_t - llam), -40.0, 40.0)
        if use_trunc:
            zm = np.clip(k * (np.log(tmin_norm) - llam), -40.0, 40.0)
            hij = np.maximum(0.0, np.exp(z) - np.exp(zm))
        else:
            hij = np.exp(z)
        ll = sum_d * (lk - k * llam) + (k - 1.0) * sum_d_logt - float(np.sum(hij))
        return -ll if np.isfinite(ll) else 1e15

    r_uw = optimize.minimize(
        nll_uw,
        np.array([np.log(0.6), 0.0]),
        method="L-BFGS-B",
        bounds=[(-5.0, 3.0), (-15.0, 15.0)],
        options={"maxiter": 1000},
    )
    k_uw = float(np.exp(r_uw.x[0]))
    ll_unfrailty_wb = float(-r_uw.fun)
    uw_at_bound = bool(abs(r_uw.x[1]) >= 14.95 or r_uw.x[0] <= -4.95 or r_uw.x[0] >= 2.95)

    # 3. Gamma-Frailty Exponential (constant hazard within patient, k=1, theta free)
    H_const_i = np.bincount(pat_inv, weights=dt_norm, minlength=n_pats)

    def nll_fe(p: np.ndarray) -> float:
        llam, ltheta = p
        if ltheta < -12.0 or ltheta > 6.0 or llam < -15.0 or llam > 15.0:
            return 1e15
        lam = np.exp(llam)
        theta = np.exp(ltheta)
        inv_theta = 1.0 / theta
        H_i = H_const_i / lam
        ll = -sum_d * llam + float(
            np.sum(
                special.gammaln(inv_theta + D_i)
                - special.gammaln(inv_theta)
                + D_i * ltheta
                - (inv_theta + D_i) * np.log1p(theta * H_i)
            )
        )
        return -ll if np.isfinite(ll) else 1e15

    best_fe = None
    for l0 in (-1.0, 0.0, 1.0):
        for th0 in (np.log(0.2), np.log(1.5)):
            r_nm = optimize.minimize(nll_fe, np.array([l0, th0]), method="Nelder-Mead", options={"maxiter": 1000})
            r_bf = optimize.minimize(nll_fe, r_nm.x, method="BFGS", options={"maxiter": 500})
            cand = r_bf if r_bf.fun <= r_nm.fun else r_nm
            if best_fe is None or cand.fun < best_fe.fun:
                best_fe = cand
    ll_frailty_exp = float(-best_fe.fun)
    theta_fe = float(np.exp(best_fe.x[1]))

    # 4. Gamma-Frailty Weibull
    fw = fit_shared_gamma_frailty_weibull(t, d, pids, t_min=t_min)
    ll_frailty_wb = float(fw["loglik"])

    lrt_frailty_exp = max(0.0, 2.0 * (ll_frailty_exp - ll_unfrailty_exp))
    lrt_dur_given_frailty = max(0.0, 2.0 * (ll_frailty_wb - ll_frailty_exp))
    p_lrt_dur_given_frailty = float(stats.chi2.sf(lrt_dur_given_frailty, df=1))
    aic_fe = 2.0 * 2.0 - 2.0 * ll_frailty_exp
    aic_fw = 2.0 * 3.0 - 2.0 * ll_frailty_wb

    return {
        "converged": bool(fw["converged"] and np.isfinite(ll_frailty_exp)),
        "n_runs": int(len(t)),
        "n_events": int(sum_d),
        "n_patients": n_pats,
        "unfrailty_exp_ll": ll_unfrailty_exp,
        "unfrailty_wb_k": k_uw,
        "unfrailty_wb_log_lam": float(r_uw.x[1]),
        "unfrailty_wb_at_boundary": uw_at_bound,
        "unfrailty_wb_ll": ll_unfrailty_wb,
        "frailty_exp_theta": theta_fe,
        "frailty_exp_ll": ll_frailty_exp,
        "frailty_wb_k": float(fw["shape_k"]),
        "frailty_wb_k_ci_lo": float(fw["shape_ci_low"]),
        "frailty_wb_k_ci_hi": float(fw["shape_ci_high"]),
        "frailty_wb_theta": float(fw["frailty_theta"]),
        "frailty_wb_at_boundary_k": bool(fw.get("at_boundary_k", False)),
        "frailty_wb_at_boundary_theta": bool(fw.get("at_boundary_theta", False)),
        "frailty_wb_ll": ll_frailty_wb,
        "lrt_frailty_given_const_haz": lrt_frailty_exp,
        "lrt_dur_dep_given_frailty": lrt_dur_given_frailty,
        "p_lrt_dur_dep_given_frailty": p_lrt_dur_given_frailty,
        "delta_aic_const_minus_weibull_frailty": float(aic_fe - aic_fw),
    }


# ---------------------------------------------------------------------------
# Output #2: Multi-Knot Piecewise-Exponential Frailty + Kernel Nelson-Aalen
# ---------------------------------------------------------------------------
HAZARD_PROFILE_KNOTS = [
    0.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0,
    3600.0, 7200.0, 21600.0, 43200.0, 86400.0, 172800.0, float("inf"),
]


def fit_multiknot_pe_frailty(
    df: pd.DataFrame,
    cohort_label: str,
    knots: List[float] = HAZARD_PROFILE_KNOTS,
) -> List[Dict[str, Any]]:
    """
    Fit multi-knot piecewise-exponential hazard WITH patient-level shared gamma frailty
    on known-onset runs (excluding boundary_censored_start == True).
    """
    df_known, _ = filter_known_onset(df)
    t = df_known["dur_sec"].to_numpy(dtype=float)
    d = (~df_known["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
    pids = df_known["patient"].to_numpy()
    mask = t > 0
    t, d, pids = t[mask], d[mask], pids[mask]
    if len(t) < 10:
        return []

    _, pat_inv = np.unique(pids, return_inverse=True)
    n_pats = int(pat_inv.max() + 1)
    D_i = np.bincount(pat_inv, weights=d, minlength=n_pats)

    K = len(knots) - 1
    E_mat = np.zeros((n_pats, K), dtype=np.float64)
    d_k = np.zeros(K, dtype=np.float64)
    n_at_risk_start = np.zeros(K, dtype=int)
    exposure_k = np.zeros(K, dtype=np.float64)

    for m in range(K):
        lo, hi = knots[m], knots[m + 1]
        n_at_risk_start[m] = int(np.sum(t >= lo))
        ev_m = (d > 0.5) & (t >= lo) & (t < hi)
        d_k[m] = float(np.sum(ev_m))
        exp_vec = np.maximum(0.0, np.minimum(t, hi) - lo)
        exposure_k[m] = float(np.sum(exp_vec))
        if exposure_k[m] > 0:
            E_mat[:, m] = np.bincount(pat_inv, weights=exp_vec, minlength=n_pats)

    # Optimize over intervals with exposure > 0
    active_idx = np.where(exposure_k > 0)[0]
    n_act = len(active_idx)
    E_act = E_mat[:, active_idx]
    d_act = d_k[active_idx]

    def nll(params: np.ndarray) -> float:
        log_h = params[:n_act]
        log_theta = params[n_act]
        h = np.exp(log_h)
        theta = np.exp(log_theta)
        inv_theta = 1.0 / theta
        H_i = E_act @ h
        ll = float(np.sum(d_act * log_h)) + float(
            np.sum(
                special.gammaln(inv_theta + D_i)
                - special.gammaln(inv_theta)
                + D_i * log_theta
                - (inv_theta + D_i) * np.log1p(theta * H_i)
            )
        )
        return -ll if np.isfinite(ll) else 1e15

    init_h = np.clip(
        np.log(np.maximum(d_act, 0.25) / np.maximum(exposure_k[active_idx], 1.0)),
        -15.0,
        0.0,
    )
    init = np.concatenate([init_h, [np.log(1.0)]])
    bounds = [(-16.0, 1.0)] * n_act + [(-8.0, 5.0)]
    res = optimize.minimize(nll, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-12})
    converged = bool(res.success)

    log_h_hat = res.x[:n_act]
    theta_hat = float(np.exp(res.x[n_act]))

    # Compute numerical Hessian SEs on intervals with n_events >= 1
    ev_pos = np.where(d_act >= 1.0)[0]
    se_log_h = np.full(n_act, np.nan, dtype=float)
    if len(ev_pos) >= 1:
        sub_idx = np.concatenate([ev_pos, [n_act]])
        p_sub = len(sub_idx)
        H_sub = np.zeros((p_sub, p_sub), dtype=float)
        eps = 1e-4
        f0 = nll(res.x)
        for a in range(p_sub):
            ia = sub_idx[a]
            ea = np.zeros_like(res.x)
            ea[ia] = eps
            H_sub[a, a] = (nll(res.x + ea) - 2.0 * f0 + nll(res.x - ea)) / (eps * eps)
            for b in range(a + 1, p_sub):
                ib = sub_idx[b]
                eb = np.zeros_like(res.x)
                eb[ib] = eps
                val = (
                    nll(res.x + ea + eb)
                    - nll(res.x + ea - eb)
                    - nll(res.x - ea + eb)
                    + nll(res.x - ea - eb)
                ) / (4.0 * eps * eps)
                H_sub[a, b] = val
                H_sub[b, a] = val
        try:
            cov_sub = np.linalg.inv(H_sub)
            diag_var = np.diag(cov_sub)[:-1]
            for pos_i, orig_i in enumerate(ev_pos):
                if np.isfinite(diag_var[pos_i]) and diag_var[pos_i] > 0:
                    se_log_h[orig_i] = float(np.sqrt(diag_var[pos_i]))
        except Exception:
            pass

    rows: List[Dict[str, Any]] = []
    act_map = {orig: pos for pos, orig in enumerate(active_idx)}
    for m in range(K):
        lo, hi = knots[m], knots[m + 1]
        nr = int(n_at_risk_start[m])
        ne = int(d_k[m])
        exp_s = float(exposure_k[m])
        if nr == 0 or exp_s <= 0.0:
            continue
        if np.isfinite(hi):
            mid = float(np.sqrt(max(lo, 1.0) * hi)) if lo > 0 else float(0.5 * hi)
        else:
            mid = float(lo * 1.5)
        if m in act_map and ne >= 1:
            pos = act_map[m]
            lh = float(log_h_hat[pos])
            h_s = float(np.exp(lh))
            se = se_log_h[pos]
            if np.isfinite(se):
                ci_lo = float(np.exp(lh - 1.959964 * se))
                ci_hi = float(np.exp(lh + 1.959964 * se))
            else:
                ci_lo, ci_hi = 0.0, float(3.6889 / exp_s)
        else:
            # Zero observed events in an active interval: exact one-sided 95% Poisson upper bound (chi2_2(0.95)/2 = 2.9957)
            h_s = 0.0
            ci_lo = 0.0
            ci_hi = float(2.995732 / exp_s)

        rows.append(
            {
                "cohort": cohort_label,
                "estimator": "piecewise_exp_frailty",
                "converged": converged,
                "interval_start_sec": float(lo),
                "interval_end_sec": float(hi),
                "mid_sec": mid,
                "n_at_risk_start": nr,
                "n_events": ne,
                "exposure_sec": exp_s,
                "hazard_per_sec": h_s,
                "hazard_ci_lo": ci_lo,
                "hazard_ci_hi": ci_hi,
                "hazard_per_hour": float(h_s * 3600.0),
                "frailty_theta": theta_hat,
                "suppressed_lt20": bool(nr < 20),
            }
        )
    return rows


def compute_nelson_aalen_kernel_hazard(
    df: pd.DataFrame,
    cohort_label: str,
    sigma: float = 0.45,
    n_grid: int = 60,
    n_boot: int = 500,
    seed: int = SEED,
) -> List[Dict[str, Any]]:
    """
    Log-time Gaussian kernel-smoothed Nelson-Aalen hazard on known-onset runs
    [15 s, 172800 s] with patient-cluster bootstrap 95% CI.
    """
    df_known, _ = filter_known_onset(df)
    t = df_known["dur_sec"].to_numpy(dtype=float)
    d = (~df_known["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
    pids = df_known["patient"].to_numpy()
    mask = t > 0
    t, d, pids = t[mask], d[mask], pids[mask]
    if len(t) < 20:
        return []

    ev_mask = d > 0.5
    ev_times = np.unique(t[ev_mask])
    if len(ev_times) < 5:
        return []

    unique_pats, pat_inv = np.unique(pids, return_inverse=True)
    n_pats = len(unique_pats)
    M = len(ev_times)

    pat_d = np.zeros((n_pats, M), dtype=np.float64)
    ev_run_idx = np.where(ev_mask)[0]
    ev_pos = np.searchsorted(ev_times, t[ev_run_idx])
    np.add.at(pat_d, (pat_inv[ev_run_idx], ev_pos), 1.0)

    ev_idx = np.searchsorted(ev_times, t, side="right") - 1
    pat_end = np.zeros((n_pats, M), dtype=np.float64)
    valid_end = ev_idx >= 0
    np.add.at(pat_end, (pat_inv[valid_end], ev_idx[valid_end]), 1.0)
    pat_y = np.cumsum(pat_end[:, ::-1], axis=1)[:, ::-1]

    D_pt = pat_d.sum(axis=0)
    Y_pt = pat_y.sum(axis=0)
    dH_pt = np.divide(D_pt, Y_pt, out=np.zeros_like(D_pt), where=Y_pt > 0)

    grid = np.geomspace(15.0, 172800.0, n_grid)
    log_grid = np.log(grid)
    log_ev = np.log(ev_times)
    # Kernel weights: (n_grid, M)
    diff = (log_grid[:, None] - log_ev[None, :]) / sigma
    K_mat = np.exp(-0.5 * diff * diff) / (np.sqrt(2.0 * np.pi) * sigma)

    h_pt = (K_mat @ dH_pt) / grid

    rng = np.random.default_rng(seed)
    boot_pat_counts = np.zeros((n_boot, n_pats), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_pats, size=n_pats)
        boot_pat_counts[b] = np.bincount(idx, minlength=n_pats)

    h_boot = np.zeros((n_boot, n_grid), dtype=np.float64)
    chunk = 100
    for s_i in range(0, n_boot, chunk):
        e_i = min(n_boot, s_i + chunk)
        W = boot_pat_counts[s_i:e_i]
        D_b = W @ pat_d
        Y_b = W @ pat_y
        dH_b = np.divide(D_b, Y_b, out=np.zeros_like(D_b), where=Y_b > 0)
        h_boot[s_i:e_i] = (dH_b @ K_mat.T) / grid[None, :]

    ci_lo_arr = np.percentile(h_boot, 2.5, axis=0)
    ci_hi_arr = np.percentile(h_boot, 97.5, axis=0)

    rows: List[Dict[str, Any]] = []
    for g_i, t0 in enumerate(grid):
        nr = int(np.sum(t >= t0))
        if nr < 20:
            continue
        h_val = float(h_pt[g_i])
        rows.append(
            {
                "cohort": cohort_label,
                "estimator": "nelson_aalen_kernel",
                "converged": True,
                "interval_start_sec": float(t0),
                "interval_end_sec": float(t0),
                "mid_sec": float(t0),
                "n_at_risk_start": nr,
                "n_events": int(np.sum((d > 0.5) & (t >= t0))),
                "exposure_sec": np.nan,
                "hazard_per_sec": h_val,
                "hazard_ci_lo": float(ci_lo_arr[g_i]),
                "hazard_ci_hi": float(ci_hi_arr[g_i]),
                "hazard_per_hour": h_val * 3600.0,
                "frailty_theta": np.nan,
                "suppressed_lt20": False,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Output #3: Table 1 at Spitaler's Grid — Full Published Table 3 (Heart Rhythm 2026)
# ---------------------------------------------------------------------------
SPITALER_GRID_HOURS = [1, 2, 3, 4, 5, 6, 12, 24, 48]
SPITALER_TABLE3: Dict[int, Dict[str, Any]] = {
    1:   {"label": "1 h",  "n_risk": 5388, "n_term": 4910, "p": 0.911, "ci_lo": 0.903, "ci_hi": 0.919},
    2:   {"label": "2 h",  "n_risk": 4574, "n_term": 4100, "p": 0.896, "ci_lo": 0.887, "ci_hi": 0.905},
    3:   {"label": "3 h",  "n_risk": 3980, "n_term": 3511, "p": 0.882, "ci_lo": 0.872, "ci_hi": 0.892},
    4:   {"label": "4 h",  "n_risk": 3481, "n_term": 3019, "p": 0.867, "ci_lo": 0.855, "ci_hi": 0.878},
    5:   {"label": "5 h",  "n_risk": 3062, "n_term": 2604, "p": 0.850, "ci_lo": 0.837, "ci_hi": 0.863},
    6:   {"label": "6 h",  "n_risk": 2726, "n_term": 2273, "p": 0.834, "ci_lo": 0.819, "ci_hi": 0.848},
    12:  {"label": "12 h", "n_risk": 1408, "n_term": 1015, "p": 0.721, "ci_lo": 0.697, "ci_hi": 0.744},
    24:  {"label": "1 d",  "n_risk": 480,  "n_term": 160,  "p": 0.333, "ci_lo": 0.292, "ci_hi": 0.378},
    48:  {"label": "2 d",  "n_risk": 320,  "n_term": 69,   "p": 0.216, "ci_lo": 0.173, "ci_hi": 0.266},
    72:  {"label": "3 d",  "n_risk": 251,  "n_term": 24,   "p": 0.096, "ci_lo": 0.063, "ci_hi": 0.141},
    96:  {"label": "4 d",  "n_risk": 227,  "n_term": 19,   "p": 0.084, "ci_lo": 0.052, "ci_hi": 0.130},
    120: {"label": "5 d",  "n_risk": 208,  "n_term": 14,   "p": 0.067, "ci_lo": 0.039, "ci_hi": 0.113},
    144: {"label": "6 d",  "n_risk": 194,  "n_term": 9,    "p": 0.046, "ci_lo": 0.023, "ci_hi": 0.089},
    168: {"label": "7 d",  "n_risk": 185,  "n_term": 8,    "p": 0.043, "ci_lo": 0.020, "ci_hi": 0.086},
}


def build_spitaler_grid_table(cond_df: pd.DataFrame, cohorts: List[str]) -> pd.DataFrame:
    sub = cond_df[
        (cond_df["sensitivity"] == "primary_g0")
        & (cond_df["window_sec"] == 86400.0)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for coh in ["POOLED"] + list(cohorts):
        c_sub = sub[sub["cohort"] == coh].set_index("threshold_sec")
        for th in SPITALER_GRID_HOURS:
            ts = float(th * 3600)
            if ts not in c_sub.index:
                continue
            r = c_sub.loc[ts]
            n_risk = int(r["n_at_risk_all"])
            n_unc = int(r["n_at_risk"])
            n_pat = int(r["n_patients_at_risk"])
            p_km = float(r["p_term_km"])
            ci_lo = float(r["km_boot_ci_low"])
            ci_hi = float(r["km_boot_ci_high"])
            if n_risk == 0 or not np.isfinite(p_km):
                continue

            supp = bool(n_risk < 20 or n_pat < 10)
            disp = "<20 runs or <10 pts (suppressed)" if supp else f"{p_km:.3f} [{ci_lo:.3f}-{ci_hi:.3f}]"

            sp = SPITALER_TABLE3[th]
            sp_h_hr = float(-np.log(1.0 - sp["p"]) / 24.0)
            rows.append(
                {
                    "cohort": coh,
                    "threshold_h": int(th),
                    "threshold_sec": ts,
                    "n_at_risk": n_risk,
                    "n_at_risk_uncensored": n_unc,
                    "n_patients_at_risk": n_pat,
                    "p_term_km": p_km,
                    "km_boot_ci_low": ci_lo,
                    "km_boot_ci_high": ci_hi,
                    "spitaler_n_at_risk": int(sp["n_risk"]),
                    "spitaler_n_term": int(sp["n_term"]),
                    "spitaler_published_p": float(sp["p"]),
                    "spitaler_ci_low": float(sp["ci_lo"]),
                    "spitaler_ci_high": float(sp["ci_hi"]),
                    "spitaler_published_display": f"{sp['p']:.3f} [{sp['ci_lo']:.3f}-{sp['ci_hi']:.3f}]",
                    "spitaler_equiv_hazard_per_hour": sp_h_hr,
                    "spitaler_source_status": "spitaler_heart_rhythm_2026_table3",
                    "suppressed_lt20": supp,
                    "p_term_km_display": disp,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Super-contributor identification
# ---------------------------------------------------------------------------
def exclude_super_contributors(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude top 10 patients by event count in LTAFDB and top 12 overall."""
    counts_all = df.groupby("patient").size().sort_values(ascending=False)
    top12_all = set(counts_all.head(12).index)

    lta = df[df["cohort"] == "LTAFDB"]
    top10_lta = set(lta.groupby("patient").size().sort_values(ascending=False).head(10).index)

    drop_pats = top12_all | top10_lta
    return df[~df["patient"].isin(drop_pats)].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ctx = get_run_context()
    run_dir: Path = ctx["run_dir"]
    cfg: Dict[str, Any] = ctx["config"]

    stage2_dir = run_dir / "stage2"
    out_dir = run_dir / "aim1"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohorts = cfg.get("cohorts", ["IRIDIA", "SHDB", "LTAFDB"])
    thresholds = cfg.get(
        "hazard_thresholds_sec",
        [30, 60, 120, 300, 600, 1800, 3600, 7200, 10800, 14400, 18000, 21600, 43200, 86400, 172800],
    )
    windows = cfg.get("hazard_windows_sec", [60, 300, 3600, 86400])
    gap_family = cfg.get("gap_merge_family_sec", [0, 30, 120, 360, 2400])
    min_run_sens = cfg.get("min_run_sec_sensitivity", [120, 300])
    knots = [300.0, 1800.0, 3600.0, 21600.0, 43200.0, 86400.0]

    runs_g0_all = pd.read_parquet(stage2_dir / "runs_g0.parquet")
    runs_g0, _ = filter_known_onset(runs_g0_all)

    # Audit left-censored (unknown-onset) exclusion across cohorts
    audit_rows = []
    for coh_lbl, sub_raw in [("POOLED", runs_g0_all)] + [(c, runs_g0_all[runs_g0_all["cohort"] == c]) for c in cohorts]:
        known_sub, n_lc = filter_known_onset(sub_raw)
        n_rc = int(known_sub["boundary_censored_end"].astype(bool).sum())
        audit_rows.append({
            "cohort": coh_lbl,
            "n_runs_g0_total": int(len(sub_raw)),
            "n_excluded_unknown_onset": int(n_lc),
            "n_analyzed_known_onset": int(len(known_sub)),
            "n_complete_terminations": int(len(known_sub) - n_rc),
            "n_right_censored_end": int(n_rc),
            "n_patients_analyzed": int(known_sub["patient"].nunique()),
        })
    add_provenance(pd.DataFrame(audit_rows), ctx).to_csv(out_dir / "onset_censoring_audit.csv", index=False)

    # Load SHDB clinical metadata & alternative rhythm-definition tiers
    shdb_meta_path = Path("/data/mithunmanivannan/datasets/shdb_af/AdditionalData.csv")
    shdb_cov = pd.DataFrame()
    if shdb_meta_path.exists():
        sm = pd.read_csv(shdb_meta_path)
        sm["record"] = sm["Data_ID"].astype(str).str.zfill(3)
        sm["cohort"] = "SHDB"
        shdb_cov = sm[["cohort", "record", "AF_Type", "Previous_AF_Ablation", "Antiarrhythmic_Drug_nonBB", "Antiarrhythmic_Drug_BB"]].drop_duplicates(["cohort", "record"])

    runs_cs = pd.read_parquet(stage2_dir / "runs_g0_continuity_nonsinus_gaps.parquet") if (stage2_dir / "runs_g0_continuity_nonsinus_gaps.parquet").exists() else pd.DataFrame()
    runs_sn = pd.read_parquet(stage2_dir / "runs_g0_continuity_except_normal.parquet") if (stage2_dir / "runs_g0_continuity_except_normal.parquet").exists() else pd.DataFrame()

    # Expand thresholds & windows to include well-supported short horizons (15 min, 30 min, 1 h, 6 h)
    short_thresholds = [300.0, 900.0, 1800.0, 3600.0, 21600.0]
    short_windows = [900.0, 1800.0, 3600.0, 21600.0]
    all_thresholds = sorted(set(float(x) for x in thresholds) | set(short_thresholds))
    all_windows = sorted(set(float(x) for x in windows) | set(short_windows))

    # 1. KM curves per cohort and pooled (g=0, known onset)
    km_rows: List[Dict[str, Any]] = []
    if KaplanMeierFitter is not None:
        for coh in list(cohorts) + ["POOLED"]:
            sub = runs_g0 if coh == "POOLED" else runs_g0[runs_g0["cohort"] == coh]
            if len(sub) == 0:
                continue
            kmf = KaplanMeierFitter()
            ev = (~sub["boundary_censored_end"].astype(bool)).astype(int)
            kmf.fit(sub["dur_sec"], event_observed=ev)
            sf = kmf.survival_function_.reset_index()
            ci = kmf.confidence_interval_survival_function_.reset_index(drop=True)
            for idx_r in range(len(sf)):
                km_rows.append(
                    {
                        "cohort": coh,
                        "timeline_sec": float(sf.iloc[idx_r, 0]),
                        "survival_prob": float(sf.iloc[idx_r, 1]),
                        "ci_low": float(ci.iloc[idx_r, 0]),
                        "ci_high": float(ci.iloc[idx_r, 1]),
                    }
                )
    km_df = add_provenance(pd.DataFrame(km_rows), ctx)
    km_df.to_parquet(out_dir / "km_curves.parquet", index=False)

    # 2. Conditional termination probabilities (primary g0 known-onset, cohort strata, + sensitivities)
    cond_rows: List[Dict[str, Any]] = []
    for coh in list(cohorts) + ["POOLED"]:
        sub = runs_g0 if coh == "POOLED" else runs_g0[runs_g0["cohort"] == coh]
        for r in compute_conditional_probs(sub, all_thresholds, all_windows, n_boot=1000, seed=SEED):
            r["cohort"] = coh
            r["sensitivity"] = "primary_g0"
            cond_rows.append(r)

    if not shdb_cov.empty:
        sh_m = runs_g0[runs_g0["cohort"] == "SHDB"].merge(shdb_cov, on=["cohort", "record"], how="left")
        for s_tag, s_sub in [
            ("SHDB_PAF", sh_m[sh_m["AF_Type"] == "PAF"]),
            ("SHDB_PerAF", sh_m[sh_m["AF_Type"].astype(str).str.upper().str.contains("PER")]),
        ]:
            for r in compute_conditional_probs(s_sub, all_thresholds, all_windows, n_boot=500, seed=SEED + 10):
                r["cohort"] = s_tag
                r["sensitivity"] = "shdb_af_type"
                cond_rows.append(r)

    if not runs_cs.empty:
        for coh in ["SHDB", "LTAFDB"]:
            sub_cs, _ = filter_known_onset(runs_cs[runs_cs["cohort"] == coh])
            for r in compute_conditional_probs(sub_cs, all_thresholds, all_windows, n_boot=500, seed=SEED + 20):
                r["cohort"] = coh
                r["sensitivity"] = "continuity_nonsinus_gaps"
                cond_rows.append(r)

    # Super-contributor exclusion sensitivity (pooled, g0, known onset)
    sub_nosc = exclude_super_contributors(runs_g0)
    for r in compute_conditional_probs(sub_nosc, all_thresholds, all_windows, n_boot=500, seed=SEED + 1):
        r["cohort"] = "POOLED"
        r["sensitivity"] = "exclude_super_contributors"
        cond_rows.append(r)

    # Min-run sensitivity (pooled, g0, known onset)
    for mr in min_run_sens:
        sub_mr = runs_g0[runs_g0["dur_sec"] >= mr]
        for r in compute_conditional_probs(sub_mr, all_thresholds, all_windows, n_boot=500, seed=SEED + int(mr)):
            r["cohort"] = "POOLED"
            r["sensitivity"] = f"min_run_{mr}s"
            cond_rows.append(r)

    cond_df = add_provenance(pd.DataFrame(cond_rows), ctx)
    cond_df.to_csv(out_dir / "table1_conditional_probs.csv", index=False)

    # Short well-supported conditional termination horizon table (15 min, 30 min, 1 h, 6 h)
    short_sub = cond_df[
        cond_df["threshold_sec"].isin(short_thresholds)
        & cond_df["window_sec"].isin(short_windows)
        & cond_df["sensitivity"].isin(["primary_g0", "shdb_af_type", "continuity_nonsinus_gaps"])
    ].copy()
    short_sub["suppressed_lt20_ep_or_lt10_pt"] = (short_sub["n_at_risk_all"] < 20) | (short_sub["n_patients_at_risk"] < 10)
    short_sub.to_csv(out_dir / "table1_short_horizons.csv", index=False)

    # 3. Shared-gamma-frailty Weibull MLE (known-onset runs!)
    weib_rows: List[Dict[str, Any]] = []
    for coh in list(cohorts) + ["POOLED"]:
        sub = runs_g0 if coh == "POOLED" else runs_g0[runs_g0["cohort"] == coh]
        ev = (~sub["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
        fit = fit_shared_gamma_frailty_weibull(sub["dur_sec"].to_numpy(), ev, sub["patient"].to_numpy())
        fit["cohort"] = coh
        fit["sensitivity"] = "primary_g0"
        fit["spitaler_ref_shape"] = 0.767
        weib_rows.append(fit)

    # Super-contributor exclusion
    ev_nosc = (~sub_nosc["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
    fit_nosc = fit_shared_gamma_frailty_weibull(sub_nosc["dur_sec"].to_numpy(), ev_nosc, sub_nosc["patient"].to_numpy())
    fit_nosc["cohort"] = "POOLED"
    fit_nosc["sensitivity"] = "exclude_super_contributors"
    fit_nosc["spitaler_ref_shape"] = 0.767
    weib_rows.append(fit_nosc)

    # Min-run sensitivity (per cohort + POOLED, properly left-truncated at t_min=mr, known-onset only)
    for mr in min_run_sens:
        for coh in list(cohorts) + ["POOLED"]:
            sub_base = runs_g0 if coh == "POOLED" else runs_g0[runs_g0["cohort"] == coh]
            sub_mr = sub_base[sub_base["dur_sec"] >= mr]
            ev_mr = (~sub_mr["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
            fit_mr = fit_shared_gamma_frailty_weibull(
                sub_mr["dur_sec"].to_numpy(), ev_mr, sub_mr["patient"].to_numpy(), t_min=float(mr)
            )
            fit_mr["cohort"] = coh
            fit_mr["sensitivity"] = f"min_run_{mr}s"
            fit_mr["spitaler_ref_shape"] = 0.767
            weib_rows.append(fit_mr)

    # Gap-family sensitivity (known onset only)
    for g in gap_family:
        if g == 0:
            continue
        gp = stage2_dir / f"runs_g{g}.parquet"
        if not gp.exists():
            continue
        df_g, _ = filter_known_onset(pd.read_parquet(gp))
        ev_g = (~df_g["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
        fit_g = fit_shared_gamma_frailty_weibull(df_g["dur_sec"].to_numpy(), ev_g, df_g["patient"].to_numpy())
        fit_g["cohort"] = "POOLED"
        fit_g["sensitivity"] = f"gap_{g}s"
        fit_g["spitaler_ref_shape"] = 0.767
        weib_rows.append(fit_g)

    weib_df = add_provenance(pd.DataFrame(weib_rows), ctx)
    weib_df.to_csv(out_dir / "tableS1_weibull_frailty.csv", index=False)

    # 3a. Nested Model Comparison (Constant patient-specific hazard vs Duration-dependent Weibull + Frailty)
    # Use proper delayed-entry (t_min=30.0) + administrative censoring at 24h (86400.0) for common-support comparison!
    sub_30_24h = runs_g0[runs_g0["dur_sec"] > 30.0].copy()
    sub_30_24h["boundary_censored_end"] = sub_30_24h["boundary_censored_end"].astype(bool) | (sub_30_24h["dur_sec"] > 86400.0)
    sub_30_24h["dur_sec"] = np.minimum(sub_30_24h["dur_sec"], 86400.0)

    comp_strata: List[Tuple[str, pd.DataFrame, float]] = [
        ("IRIDIA_primary_PAF", runs_g0[runs_g0["cohort"] == "IRIDIA"], 0.0),
        ("SHDB_all", runs_g0[runs_g0["cohort"] == "SHDB"], 0.0),
        ("LTAFDB_harmonized", runs_g0[runs_g0["cohort"] == "LTAFDB"], 0.0),
        ("POOLED_all", runs_g0, 0.0),
        ("POOLED_no_super_contributors", sub_nosc, 0.0),
        ("POOLED_left_trunc_30s_admin_24h", sub_30_24h, 30.0),
        ("IRIDIA_left_trunc_30s_admin_24h", sub_30_24h[sub_30_24h["cohort"] == "IRIDIA"], 30.0),
    ]
    if not shdb_cov.empty:
        sh_m = runs_g0[runs_g0["cohort"] == "SHDB"].merge(shdb_cov, on=["cohort", "record"], how="left")
        comp_strata.extend([
            ("SHDB_PAF", sh_m[sh_m["AF_Type"] == "PAF"], 0.0),
            ("SHDB_PerAF", sh_m[sh_m["AF_Type"].astype(str).str.upper().str.contains("PER")], 0.0),
            ("SHDB_prior_ablation_yes", sh_m[sh_m["Previous_AF_Ablation"].astype(str).str.lower() == "true"], 0.0),
            ("SHDB_prior_ablation_no", sh_m[sh_m["Previous_AF_Ablation"].astype(str).str.lower() == "false"], 0.0),
            ("SHDB_any_AAD_yes", sh_m[sh_m["Antiarrhythmic_Drug_nonBB"].notna() | sh_m["Antiarrhythmic_Drug_BB"].notna()], 0.0),
            ("SHDB_any_AAD_no", sh_m[sh_m["Antiarrhythmic_Drug_nonBB"].isna() & sh_m["Antiarrhythmic_Drug_BB"].isna()], 0.0),
        ])
    if not runs_cs.empty:
        for coh in ["SHDB", "LTAFDB"]:
            cs_k, _ = filter_known_onset(runs_cs[runs_cs["cohort"] == coh])
            comp_strata.append((f"{coh}_continuity_nonsinus_gaps", cs_k, 0.0))
    if not runs_sn.empty:
        for coh in ["SHDB", "LTAFDB"]:
            sn_k, _ = filter_known_onset(runs_sn[runs_sn["cohort"] == coh])
            comp_strata.append((f"{coh}_continuity_except_normal", sn_k, 0.0))

    def _weighted_median(vals: np.ndarray, weights: np.ndarray) -> float:
        if len(vals) == 0:
            return float("nan")
        order = np.argsort(vals)
        v_s, w_s = vals[order], weights[order]
        cw = np.cumsum(w_s) / np.sum(w_s)
        idx = int(np.searchsorted(cw, 0.5, side="left"))
        return float(v_s[min(idx, len(v_s) - 1)])

    comp_rows = []
    hier_rows = []
    for s_name, s_df, t_min_val in comp_strata:
        if len(s_df) < 10:
            continue
        ev_s = (~s_df["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
        durs_s = s_df["dur_sec"].to_numpy(dtype=float)
        res_m = fit_nested_hazard_models(durs_s, ev_s, s_df["patient"].to_numpy(), t_min=t_min_val)
        res_m["stratum"] = s_name
        res_m["left_trunc_sec"] = float(t_min_val)
        comp_rows.append(res_m)

        # Report BOTH patient-level median-of-medians AND inverse-episode-count weighted quantile median
        unc_s = s_df[ev_s > 0.5]
        if len(unc_s):
            ep_med_dur = float(unc_s["dur_sec"].median())
            pt_med_of_meds = float(unc_s.groupby("patient")["dur_sec"].median().median())
            pt_counts = unc_s.groupby("patient")["dur_sec"].transform("count").to_numpy(float)
            pw_quant_med = _weighted_median(unc_s["dur_sec"].to_numpy(float), 1.0 / pt_counts)
        else:
            ep_med_dur = pt_med_of_meds = pw_quant_med = float("nan")
        hier_rows.append({
            "stratum": s_name,
            "n_runs_known_onset": int(len(s_df)),
            "n_complete": int(np.sum(ev_s)),
            "n_patients": int(s_df["patient"].nunique()),
            "episode_weighted_median_dur_sec": ep_med_dur,
            "patient_median_of_medians_sec": pt_med_of_meds,
            "patient_weighted_quantile_median_sec": pw_quant_med,
            "frailty_wb_shape_k": res_m.get("frailty_wb_k", np.nan),
            "frailty_wb_theta": res_m.get("frailty_wb_theta", np.nan),
            "p_lrt_dur_dep_given_frailty": res_m.get("p_lrt_dur_dep_given_frailty", np.nan),
        })

    add_provenance(pd.DataFrame(comp_rows), ctx).to_csv(out_dir / "model_comparison_frailty.csv", index=False)
    add_provenance(pd.DataFrame(hier_rows), ctx).to_csv(out_dir / "cohort_hierarchy_summary.csv", index=False)

    # 3b. Output #4 — S10 Side-by-Side: Weibull shape vs min_run_sec curve (known onset only)
    min_run_curve_grid = [0, 30, 60, 120, 300, 600, 1800, 3600]
    curve_rows: List[Dict[str, Any]] = []
    for coh in list(cohorts) + ["POOLED"]:
        sub_base = runs_g0 if coh == "POOLED" else runs_g0[runs_g0["cohort"] == coh]
        for mr in min_run_curve_grid:
            sub_mr = sub_base[sub_base["dur_sec"] >= mr]
            durs_mr = sub_mr["dur_sec"].to_numpy(dtype=float)
            ev_mr = (~sub_mr["boundary_censored_end"].astype(bool)).astype(float).to_numpy()
            pats_mr = sub_mr["patient"].to_numpy()

            fit_lt = fit_shared_gamma_frailty_weibull(durs_mr, ev_mr, pats_mr, t_min=float(mr))
            fit_sh = fit_shared_gamma_frailty_weibull(durs_mr, ev_mr, pats_mr, t_min=0.0)
            if mr > 0:
                ex_mask = durs_mr > float(mr)
                fit_ex = fit_shared_gamma_frailty_weibull(
                    durs_mr[ex_mask] - float(mr), ev_mr[ex_mask], pats_mr[ex_mask], t_min=0.0
                )
            else:
                fit_ex = fit_lt

            row = dict(fit_lt)
            row["cohort"] = coh
            row["min_run_sec"] = float(mr)
            row["shape_k_left_trunc"] = fit_lt["shape_k"]
            row["shape_ci_lo_left_trunc"] = fit_lt["shape_ci_low"]
            row["shape_ci_hi_left_trunc"] = fit_lt["shape_ci_high"]
            row["frailty_theta_left_trunc"] = fit_lt["frailty_theta"]
            row["shape_k_shifted_origin"] = fit_sh["shape_k"]
            row["shape_ci_lo_shifted_origin"] = fit_sh["shape_ci_low"]
            row["shape_ci_hi_shifted_origin"] = fit_sh["shape_ci_high"]
            row["frailty_theta_shifted_origin"] = fit_sh["frailty_theta"]
            row["untruncated_shape_k"] = fit_sh["shape_k"]
            row["shape_k_excess"] = fit_ex["shape_k"]
            row["shape_ci_lo_excess"] = fit_ex["shape_ci_low"]
            row["shape_ci_hi_excess"] = fit_ex["shape_ci_high"]
            row["frailty_theta_excess"] = fit_ex["frailty_theta"]
            row["spitaler_ref_shape"] = 0.767
            curve_rows.append(row)
    curve_df = add_provenance(pd.DataFrame(curve_rows), ctx)
    curve_df.to_csv(out_dir / "weibull_shape_vs_min_run.csv", index=False)

    # 4. Output #3 — Table 1 at Spitaler's Grid
    sp_grid_df = add_provenance(build_spitaler_grid_table(cond_df, cohorts), ctx)
    sp_grid_df.to_csv(out_dir / "table1_spitaler_grid.csv", index=False)

    # 5. Output #2 — Nonparametric + Multi-Knot Frailty Hazard Profile (replaces dropped Table S2)
    hp_rows: List[Dict[str, Any]] = []
    for coh in list(cohorts) + ["POOLED"]:
        sub = runs_g0 if coh == "POOLED" else runs_g0[runs_g0["cohort"] == coh]
        hp_rows.extend(fit_multiknot_pe_frailty(sub, coh))
        hp_rows.extend(compute_nelson_aalen_kernel_hazard(sub, coh, sigma=0.45, n_grid=60, n_boot=500, seed=SEED))

    # Full verified published Spitaler et al. (Heart Rhythm 2026) Table 3 equivalent 24h hazard reference points (1h..7d)
    for th, sp in sorted(SPITALER_TABLE3.items()):
        p_pub = float(sp["p"])
        h_s = float(-np.log(1.0 - p_pub) / 86400.0)
        h_lo = float(-np.log(1.0 - float(sp["ci_lo"])) / 86400.0)
        h_hi = float(-np.log(1.0 - float(sp["ci_hi"])) / 86400.0)
        hp_rows.append(
            {
                "cohort": "SPITALER_REF",
                "estimator": "spitaler_published_ref",
                "converged": True,
                "interval_start_sec": float(th * 3600),
                "interval_end_sec": float(th * 3600 + 86400),
                "mid_sec": float(th * 3600),
                "n_at_risk_start": int(sp["n_risk"]),
                "n_events": int(sp["n_term"]),
                "exposure_sec": 86400.0,
                "hazard_per_sec": h_s,
                "hazard_ci_lo": h_lo,
                "hazard_ci_hi": h_hi,
                "hazard_per_hour": h_s * 3600.0,
                "frailty_theta": 0.0,
                "suppressed_lt20": False,
            }
        )

    hp_df = add_provenance(pd.DataFrame(hp_rows), ctx)
    hp_df.to_csv(out_dir / "hazard_profile.csv", index=False)

    de_ver = add_provenance(verify_delayed_entry_frailty_likelihood(SEED + 31), ctx)
    de_ver.to_csv(out_dir / "delayed_entry_likelihood_verification.csv", index=False)

    print(
        f"[Aim 1] Saved table1 ({len(cond_df)} rows), table1_spitaler_grid ({len(sp_grid_df)} rows), "
        f"tableS1 ({len(weib_df)} rows), delayed_entry_ver ({len(de_ver)} rows), weibull_shape_vs_min_run ({len(curve_df)} rows), "
        f"hazard_profile ({len(hp_df)} rows), km_curves ({len(km_df)} rows) in {out_dir}"
    )


if __name__ == "__main__":
    main()
