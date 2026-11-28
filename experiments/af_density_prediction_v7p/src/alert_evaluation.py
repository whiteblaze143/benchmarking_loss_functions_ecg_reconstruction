"""Continuous alert forecasting evaluation for Task P01."""
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def evaluate_p01_alerts(
    predictions_df: pd.DataFrame,
    probability_threshold: float = 0.5,
    forecast_horizon_s: float = 1800.0,
) -> Dict[str, float]:
    """Computes clinical alert metrics: episode detection fraction, lead time, false alarm rate."""
    df = predictions_df.copy()
    df["alert"] = df["predicted_prob"] >= probability_threshold

    # True sustained episodes from positive landmarks
    pos_df = df[df["target_sustained_30m"]].copy()
    neg_df = df[~df["target_sustained_30m"]].copy()

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    if n_pos > 0:
        episodes_alerted = float(pos_df["alert"].mean())
    else:
        episodes_alerted = 0.0

    # Total non-AF monitoring hours evaluated
    total_monitoring_hours = (len(df) * 300.0) / 3600.0  # 300s per grid point
    false_alerts = int(neg_df["alert"].sum())
    false_alert_rate_per_hour = float(false_alerts / total_monitoring_hours) if total_monitoring_hours > 0 else 0.0

    # Time in alarm status
    alarm_burden_pct = float(df["alert"].mean() * 100.0)

    # Lead time for true alerts
    # If alert is fired at landmark t for episode starting at event_time_s: lead_time = event_time_s - t
    true_alerts = pos_df[pos_df["alert"]].copy()
    if len(true_alerts) > 0 and "event_time_s" in true_alerts.columns:
        lead_times_m = (true_alerts["event_time_s"] - true_alerts["landmark_time_s"]) / 60.0
        lead_times_m = np.clip(lead_times_m, 0.0, forecast_horizon_s / 60.0)
        median_lead_time_m = float(np.median(lead_times_m))
    else:
        median_lead_time_m = np.nan

    return {
        "episodes_alerted_fraction": episodes_alerted,
        "median_lead_time_minutes": median_lead_time_m,
        "false_alerts_per_monitoring_hour": false_alert_rate_per_hour,
        "alarm_burden_percent": alarm_burden_pct,
        "total_monitoring_hours": total_monitoring_hours,
        "operating_threshold": probability_threshold,
    }
