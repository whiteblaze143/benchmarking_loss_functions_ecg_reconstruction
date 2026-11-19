"""
Temporal Markov / HMM Smoothing & Pareto Abstention Calibration Suite.
Designed for Continuous Ambulatory Holter and Wearable Telemetry Monitoring.
Solves window-level chatter, suppresses burden MAE, and eliminates false recurrence alarms.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit, expit


class TemporalMarkovSmoother:
    """
    Two-state Hidden Markov Model (HMM) for continuous AF sequence modeling:
      State 0: Normal Sinus Rhythm / Ectopy (NSR)
      State 1: Atrial Fibrillation / Atrial Flutter (AF)
    
    Incorporates physiological state-transition persistence:
      P(AF_{t+1} | AF_t) = 1 - p_offset (high persistence)
      P(AF_{t+1} | NSR_t) = p_onset (low onset rate)
    """

    def __init__(
        self,
        p_onset: float = 0.001,       # Probability of transition NSR -> AF per 30s window (~1-3 per day)
        p_offset: float = 0.020,      # Probability of transition AF -> NSR per 30s window (mean episode ~50 windows = 25 min)
        train_prior_af: float = 0.50, # Training fold class balance (typically balanced 50/50)
        test_prior_af: float = 0.05,  # Real-world deployment prior AF prevalence (~5%)
        min_episode_windows: int = 1, # 1 window = 30s; 12 windows = 6 minutes
        temperature: float = 1.0,
    ):
        self.p_onset = p_onset
        self.p_offset = p_offset
        self.train_prior_af = train_prior_af
        self.test_prior_af = test_prior_af
        self.min_episode_windows = min_episode_windows
        self.temperature = temperature

        # Transition matrix in log space
        # A[i, j] = P(S_{t+1} = j | S_t = i)
        self.log_A = np.array([
            [np.log(max(1e-8, 1.0 - p_onset)), np.log(max(1e-8, p_onset))],
            [np.log(max(1e-8, p_offset)), np.log(max(1e-8, 1.0 - p_offset))],
        ], dtype=np.float64)

        # Initial state prior distribution in log space
        self.log_pi = np.array([
            np.log(max(1e-8, 1.0 - test_prior_af)),
            np.log(max(1e-8, test_prior_af)),
        ], dtype=np.float64)

    def viterbi_decode(
        self,
        probs: np.ndarray,
        abstain_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Global Maximum A Posteriori (MAP) state sequence via Viterbi algorithm.
        Handles abstained / missing windows seamlessly by propagating transition priors.
        Args:
            probs: 1D array of predicted AF probabilities P(AF_t | x_t) in [0, 1].
            abstain_mask: Optional boolean array (True if window was abstained).
        Returns:
            smoothed_states: 1D binary array (1 = AF, 0 = NSR).
        """
        T = len(probs)
        if T == 0:
            return np.zeros(0, dtype=int)

        # Compute log emission probabilities log P(x_t | S_t)
        # Using Bayes' rule adjusting for training balance:
        # P(x_t | S_t = 1) ∝ P(S_t = 1 | x_t) / P_train(S_t = 1)
        # P(x_t | S_t = 0) ∝ P(S_t = 0 | x_t) / P_train(S_t = 0)
        eps = 1e-6
        clipped_p = np.clip(probs, eps, 1.0 - eps)
        if self.temperature != 1.0:
            scaled_logit = logit(clipped_p) / self.temperature
            clipped_p = expit(scaled_logit)

        log_emit = np.zeros((T, 2), dtype=np.float64)
        log_emit[:, 0] = np.log(1.0 - clipped_p) - np.log(1.0 - self.train_prior_af)
        log_emit[:, 1] = np.log(clipped_p) - np.log(self.train_prior_af)

        # For abstained windows, emission is uninformative (log likelihood ratio = 0)
        if abstain_mask is not None:
            log_emit[abstain_mask, :] = 0.0

        # Viterbi forward pass
        viterbi_vars = np.zeros((T, 2), dtype=np.float64)
        backpointers = np.zeros((T, 2), dtype=int)

        viterbi_vars[0] = self.log_pi + log_emit[0]

        for t in range(1, T):
            for state in range(2):
                # Transition candidates from previous states
                candidates = viterbi_vars[t - 1] + self.log_A[:, state]
                best_prev = int(np.argmax(candidates))
                viterbi_vars[t, state] = candidates[best_prev] + log_emit[t, state]
                backpointers[t, state] = best_prev

        # Backtracking
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = int(np.argmax(viterbi_vars[-1]))
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointers[t + 1, best_path[t + 1]]

        # Enforce minimum episode duration constraint (anti-chatter filter)
        if self.min_episode_windows > 1:
            best_path = self._filter_short_episodes(best_path, self.min_episode_windows)

        return best_path

    def forward_backward_smooth(
        self,
        probs: np.ndarray,
        abstain_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute smoothed posterior marginal probabilities gamma_t = P(S_t = 1 | x_{1:T}).
        """
        T = len(probs)
        if T == 0:
            return np.zeros(0, dtype=np.float32)

        eps = 1e-6
        clipped_p = np.clip(probs, eps, 1.0 - eps)
        log_emit = np.zeros((T, 2), dtype=np.float64)
        log_emit[:, 0] = np.log(1.0 - clipped_p) - np.log(1.0 - self.train_prior_af)
        log_emit[:, 1] = np.log(clipped_p) - np.log(self.train_prior_af)

        if abstain_mask is not None:
            log_emit[abstain_mask, :] = 0.0

        # Forward pass (in log space using logaddexp)
        log_alpha = np.zeros((T, 2), dtype=np.float64)
        log_alpha[0] = self.log_pi + log_emit[0]

        for t in range(1, T):
            for j in range(2):
                log_alpha[t, j] = (
                    np.logaddexp(
                        log_alpha[t - 1, 0] + self.log_A[0, j],
                        log_alpha[t - 1, 1] + self.log_A[1, j],
                    )
                    + log_emit[t, j]
                )

        # Backward pass
        log_beta = np.zeros((T, 2), dtype=np.float64)
        log_beta[-1] = 0.0  # log(1.0)

        for t in range(T - 2, -1, -1):
            for i in range(2):
                log_beta[t, i] = np.logaddexp(
                    self.log_A[i, 0] + log_emit[t + 1, 0] + log_beta[t + 1, 0],
                    self.log_A[i, 1] + log_emit[t + 1, 1] + log_beta[t + 1, 1],
                )

        # Posterior marginals
        log_gamma = log_alpha + log_beta
        log_norm = np.logaddexp(log_gamma[:, 0], log_gamma[:, 1])
        smoothed_prob_af = np.exp(log_gamma[:, 1] - log_norm)

        return smoothed_prob_af.astype(np.float32)

    @staticmethod
    def _filter_short_episodes(states: np.ndarray, min_len: int) -> np.ndarray:
        """Remove any isolated contiguous AF runs shorter than min_len windows."""
        filtered = states.copy()
        T = len(states)
        i = 0
        while i < T:
            if filtered[i] == 1:
                start = i
                while i < T and filtered[i] == 1:
                    i += 1
                end = i
                if (end - start) < min_len:
                    filtered[start:end] = 0
            else:
                i += 1
        return filtered


class ParetoAbstentionCalibrator:
    """
    Pareto Frontier Calibration for Chow's Quality-Aware Selective Abstention Gate.
    Explores trade-offs between:
      1) Signal Coverage (% of monitoring time retained)
      2) Burden Estimation MAE (Mean Absolute Error against ground truth)
      3) False Alert Rate on Normal Sinus Rhythm (NSR) Controls
    """

    def __init__(
        self,
        tau_qual_grid: list[float] | None = None,
        tau_conf_grid: list[float] | None = None,
    ):
        self.tau_qual_grid = tau_qual_grid or [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
        self.tau_conf_grid = tau_conf_grid or [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    def calibrate(
        self,
        records_data: list[dict],
        smoother: TemporalMarkovSmoother | None = None,
    ) -> dict[str, object]:
        """
        Evaluate full grid of (tau_qual, tau_conf) across records.
        records_data format:
          List of dicts: {
             'rec_id': str,
             'is_control': bool,     # True for 33 non-AF controls
             'gt_burden_pct': float, # Ground truth burden
             'raw_probs': np.ndarray,
             'sqi_scores': np.ndarray,
          }
        """
        grid_results = []

        for t_q in self.tau_qual_grid:
            for t_c in self.tau_conf_grid:
                total_windows = 0
                accepted_windows = 0
                burden_errors = []
                control_false_af_windows = 0
                control_accepted_windows = 0
                control_patient_alarms = 0
                total_control_patients = 0

                for rec in records_data:
                    probs = rec["raw_probs"]
                    sqi = rec["sqi_scores"]
                    is_ctrl = rec["is_control"]
                    gt_b = rec.get("gt_burden_pct", rec.get("GT_Burden_Pct", 0.0))
                    n_w = len(probs)
                    if n_w == 0:
                        continue

                    conf = np.maximum(probs, 1.0 - probs)
                    abstain = (sqi < t_q) | (conf < t_c)

                    total_windows += n_w
                    n_acc = int(np.sum(~abstain))
                    accepted_windows += n_acc

                    # Smoothed prediction on accepted windows
                    if smoother is not None:
                        pred_states = smoother.viterbi_decode(probs, abstain_mask=abstain)
                    else:
                        pred_states = (probs >= 0.50).astype(int)

                    # Calculate predicted burden on accepted windows
                    if n_acc > 0:
                        acc_pred = pred_states[~abstain]
                        pred_b = (float(np.sum(acc_pred == 1)) / n_acc) * 100.0
                    else:
                        pred_b = 0.0

                    if not is_ctrl and gt_b is not None:
                        burden_errors.append(abs(pred_b - gt_b))

                    if is_ctrl:
                        total_control_patients += 1
                        if n_acc > 0:
                            n_false_af = int(np.sum(acc_pred == 1))
                            control_false_af_windows += n_false_af
                            control_accepted_windows += n_acc
                            if n_false_af > 0:
                                control_patient_alarms += 1

                coverage_pct = round(100.0 * (accepted_windows / max(1, total_windows)), 2)
                mean_mae = round(float(np.mean(burden_errors)), 2) if burden_errors else 0.0
                control_far_pct = (
                    round(100.0 * (control_false_af_windows / max(1, control_accepted_windows)), 3)
                    if control_accepted_windows > 0
                    else 0.0
                )
                control_specificity = (
                    round(100.0 * (1.0 - (control_patient_alarms / max(1, total_control_patients))), 2)
                    if total_control_patients > 0
                    else 100.0
                )

                grid_results.append({
                    "tau_qual": t_q,
                    "tau_conf": t_c,
                    "coverage_pct": coverage_pct,
                    "burden_mae_pct": mean_mae,
                    "control_far_pct": control_far_pct,
                    "control_patient_specificity_pct": control_specificity,
                })

        valid_pts = [p for p in grid_results if p["coverage_pct"] >= 80.0]
        if not valid_pts:
            valid_pts = grid_results

        best_config = sorted(valid_pts, key=lambda x: (x["burden_mae_pct"], x["control_far_pct"]))[0]

        return {
            "all_grid_points": grid_results,
            "optimal_operating_point": best_config,
        }
