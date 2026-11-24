"""Core P-Wave Morphological Variability (PMV) Extraction & Alignment Module.

Implements the exact signal processing and variability metrics from:
1. Noriega et al. IEEE TBME 2026 (DOI: 10.1109/TBME.2026.3742161)
2. Noriega et al. J Electrocardiol 2026 (DOI: 10.1016/j.jelectrocard.2026.154425)
3. Kurogouchi et al. Int J Cardiol 2003 (DOI: 10.1016/S0167-5273(02)00358-3)

Features:
- 12th-order zero-phase Butterworth bandpass filter (0.5–40 Hz).
- Strict 140 ms pre-QRS extraction window ([-220 ms, -80 ms]).
- Beat SNR quality gating (> 20 dB QRS-to-noise ratio).
- Sample-wise MAD energy: D(k) = sum d_k(n)^2.
- Exact Noriega denominator: E_i = median of individual P-wave energies across reference baseline.
- Fixed spatial PCA projection vector across entire episode/recording.
- PR-interval jitter detection (peak-to-valley > 0.025 and peak separation >= 50 ms) and template alignment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal


@dataclass
class PMVSubsegmentResult:
    """Results of PMV and comparator metrics for a single 120s subsegment."""
    time_center_s: float
    num_beats: int
    is_valid: bool
    status: str
    
    # Core PMV metrics
    mad_energy_D: float  # Absolute MAD energy: sum d(n)^2
    ref_energy_Ei: float  # Fixed reference individual P-wave median energy
    pmv_V: float  # Published Noriega PMV: D / Ei
    local_median_energy_Em: float  # Local median template energy: sum m(n)^2
    pmv_mad_amplitude_mean: float  # Mean of d(n)
    pmv_mad_amplitude_max: float  # Max of d(n)
    
    # PR-aligned PMV metrics
    pr_aligned: bool
    pmv_aligned_D: float
    pmv_aligned_V: float
    mean_abs_pr_shift_ms: float
    
    # Comparators
    p_amp_dispersion_iqr: float
    p_amp_dispersion_std: float
    p_duration_dispersion_std: float
    p_arc_length_dispersion_std: float
    pca_lambda2_ratio: float  # lambda2 / lambda1
    
    # Rhythm context
    mean_rr_ms: float
    sdnn_ms: float
    rmssd_ms: float
    mean_hr_bpm: float


class PMVExtractor:
    """Exact implementation of Noriega et al. 2026 P-wave Morphological Variability."""

    def __init__(
        self,
        fs: float = 200.0,
        subsegment_len_s: float = 120.0,
        subsegment_step_s: float = 60.0,
        pre_qrs_start_s: float = -0.220,
        pre_qrs_end_s: float = -0.080,
        min_beats_per_subsegment: int = 20,
        butter_order: int = 12,
        bandpass_freqs: Tuple[float, float] = (0.5, 40.0),
        enable_pr_alignment: bool = True,
        max_pr_shift_ms: float = 20.0,
        min_beat_snr_db: float = 20.0,
    ):
        self.fs = float(fs)
        self.subsegment_len_s = float(subsegment_len_s)
        self.subsegment_step_s = float(subsegment_step_s)
        self.pre_qrs_start_s = float(pre_qrs_start_s)
        self.pre_qrs_end_s = float(pre_qrs_end_s)
        self.min_beats = int(min_beats_per_subsegment)
        self.butter_order = int(butter_order)
        self.bandpass_freqs = bandpass_freqs
        self.enable_pr_alignment = enable_pr_alignment
        self.max_pr_shift_ms = float(max_pr_shift_ms)
        self.min_beat_snr_db = float(min_beat_snr_db)

        # Precompute window length in samples (140 ms window)
        self.window_duration_s = self.pre_qrs_end_s - self.pre_qrs_start_s
        self.window_len_samples = int(round(self.window_duration_s * self.fs))
        self.pre_qrs_start_samples = int(round(self.pre_qrs_start_s * self.fs))
        self.pre_qrs_end_samples = self.pre_qrs_start_samples + self.window_len_samples

        # Build 12th order Butterworth bandpass filter
        nyq = 0.5 * self.fs
        low = self.bandpass_freqs[0] / nyq
        high = min(self.bandpass_freqs[1] / nyq, 0.99)
        self.sos = signal.butter(
            self.butter_order,
            [low, high],
            btype="bandpass",
            output="sos"
        )

    def filter_signal(self, ecg: np.ndarray) -> np.ndarray:
        """Apply zero-phase 12th-order Butterworth bandpass filter (0.5-40 Hz)."""
        if ecg.ndim == 1:
            return signal.sosfiltfilt(self.sos, ecg)
        elif ecg.ndim == 2:
            return signal.sosfiltfilt(self.sos, ecg, axis=-1)
        else:
            raise ValueError(f"Expected 1D or 2D ECG array, got shape {ecg.shape}")

    def compute_reference_energy(self, p_waves: np.ndarray) -> float:
        """Exact Noriega published denominator:
        Median energy of INDIVIDUAL P-waves across the baseline ensemble.
        E_i = median_j { sum_n p_j(n)^2 }
        """
        if len(p_waves) == 0:
            return np.nan
        individual_energies = np.sum(p_waves**2, axis=-1)
        Ei = float(np.median(individual_energies))
        return Ei

    def extract_p_wave_ensemble(
        self,
        ecg_lead: np.ndarray,
        qrs_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract P-wave snippets for given QRS indices on a single lead.
        
        Applies beat SNR gating (> 20 dB) and removes linear drift.
        """
        valid_p = []
        valid_qrs = []
        n_samples = len(ecg_lead)

        # Baseline noise estimate across lead
        lead_std = np.std(ecg_lead) + 1e-8

        # Compute median RR interval for ectopic / premature beat rejection
        if len(qrs_indices) >= 5:
            rr_diffs = np.diff(qrs_indices)
            med_rr = float(np.median(rr_diffs))
        else:
            med_rr = None

        for idx_i, qrs_idx in enumerate(qrs_indices):
            start_idx = qrs_idx + self.pre_qrs_start_samples
            end_idx = start_idx + self.window_len_samples

            if start_idx < 0 or end_idx > n_samples:
                continue

            # Sinus-beat regularity check: reject premature beats or compensatory pauses
            if med_rr is not None and idx_i > 0:
                prev_rr = qrs_idx - qrs_indices[idx_i - 1]
                if prev_rr < 0.70 * med_rr or prev_rr > 1.35 * med_rr:
                    continue

            snippet = ecg_lead[start_idx:end_idx]

            # Quality check: non-finite or non-physiological voltage
            if np.any(np.isnan(snippet)) or np.any(np.isinf(snippet)):
                continue
            span = np.ptp(snippet)
            # Physiological P-wave voltage range: 5 uV to 0.80 mV (rejects flatlines & motion artifacts)
            if span < 0.005 or span > 0.80:
                continue

            # Beat SNR check: QRS peak-to-peak amplitude vs isoelectric baseline noise
            qrs_start = max(0, qrs_idx - int(round(0.04 * self.fs)))
            qrs_end = min(n_samples, qrs_idx + int(round(0.06 * self.fs)))
            qrs_amp = np.ptp(ecg_lead[qrs_start:qrs_end])
            
            # Isoelectric PR segment (between P end and Q onset: -80 ms to -20 ms)
            pr_start = end_idx
            pr_end = max(pr_start + 2, qrs_idx - int(round(0.02 * self.fs)))
            if pr_end <= n_samples and pr_end - pr_start >= 3:
                noise_std = float(np.std(signal.detrend(ecg_lead[pr_start:pr_end])))
            else:
                noise_std = float(np.std(np.diff(snippet)) / np.sqrt(2.0))
            noise_std = max(noise_std, 1e-6)

            snr_db = 20.0 * np.log10(max(qrs_amp / noise_std, 1e-3))
            if snr_db < self.min_beat_snr_db:
                continue

            # Detrend snippet
            snippet_detrend = signal.detrend(snippet, type="linear")

            valid_p.append(snippet_detrend)
            valid_qrs.append(qrs_idx)

        if len(valid_p) == 0:
            return np.empty((0, self.window_len_samples), dtype=np.float64), np.empty(0, dtype=int)

        return np.array(valid_p, dtype=np.float64), np.array(valid_qrs, dtype=int)

    def compute_spatial_pca_projection(
        self,
        ecg_2lead: np.ndarray,
        qrs_indices: np.ndarray,
        fixed_eigenvector: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Spatial PCA projection across 2 leads.
        
        If fixed_eigenvector is provided, projects using that fixed reference vector.
        """
        assert ecg_2lead.shape[0] == 2, "ecg_2lead must have shape (2, n_samples)"
        p_lead0, qrs0 = self.extract_p_wave_ensemble(ecg_2lead[0], qrs_indices)
        p_lead1, qrs1 = self.extract_p_wave_ensemble(ecg_2lead[1], qrs_indices)

        # Intersect valid QRS indices between both leads
        common_qrs, idx0, idx1 = np.intersect1d(qrs0, qrs1, return_indices=True)
        if len(common_qrs) < self.min_beats:
            return (
                np.empty((0, self.window_len_samples)),
                np.empty(0, dtype=int),
                np.nan,
                np.empty(0)
            )

        p0_sub = p_lead0[idx0]
        p1_sub = p_lead1[idx1]

        if fixed_eigenvector is not None:
            u1 = fixed_eigenvector / (np.linalg.norm(fixed_eigenvector) + 1e-12)
            if u1[0] < 0:
                u1 = -u1
            projected = u1[0] * p0_sub + u1[1] * p1_sub
            return projected, common_qrs, 0.0, u1

        # Form spatial correlation matrix: R = sum(P_i @ P_i^T) / (N * L)
        N, L = p0_sub.shape
        R = np.zeros((2, 2), dtype=np.float64)
        for i in range(N):
            P_i = np.vstack([p0_sub[i], p1_sub[i]])
            R += P_i @ P_i.T
        R /= (N * L)

        eigenvalues, eigenvectors = np.linalg.eigh(R)
        lambda1 = eigenvalues[1]
        lambda2 = max(eigenvalues[0], 0.0)
        u1 = eigenvectors[:, 1]

        if u1[0] < 0:
            u1 = -u1

        lambda2_ratio = float(lambda2 / (lambda1 + 1e-12))
        projected = u1[0] * p0_sub + u1[1] * p1_sub
        return projected, common_qrs, lambda2_ratio, u1

    def align_pr_jitter(
        self,
        p_waves: np.ndarray,
        template: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """Align P-waves to template using cross-correlation within [-max_shift, +max_shift].
        
        Strict Noriega criteria:
        1. Peak-to-valley ratio of normalized ensemble std > 0.025.
        2. Prominent peak separation >= 50 ms.
        """
        N, L = p_waves.shape
        if N < 2:
            return p_waves, 0.0, False

        ensemble_std = np.std(p_waves, axis=0)
        max_std = np.max(ensemble_std)
        if max_std < 1e-8:
            return p_waves, 0.0, False

        norm_std = ensemble_std / max_std
        ptv = np.ptp(norm_std)
        if ptv < 0.025:
            return p_waves, 0.0, False

        # Peak separation criterion (>= 50 ms between at least two distinct peaks)
        min_distance_samples = max(2, int(round(0.050 * self.fs)))
        peaks, _ = signal.find_peaks(norm_std, distance=min_distance_samples, height=0.3)
        if len(peaks) < 2:
            # If fewer than 2 distinct separated peaks exist, PR jitter is absent
            return p_waves, 0.0, False

        max_shift_samples = int(round(self.max_pr_shift_ms * 1e-3 * self.fs))
        shifts = []
        aligned_p = np.zeros_like(p_waves)

        template_norm = template - np.mean(template)
        template_std = np.std(template_norm) + 1e-12

        for i in range(N):
            beat = p_waves[i]
            beat_norm = beat - np.mean(beat)
            beat_std = np.std(beat_norm) + 1e-12

            best_shift = 0
            best_corr = -1.0

            for s in range(-max_shift_samples, max_shift_samples + 1):
                if s < 0:
                    b_win = beat_norm[-s:]
                    t_win = template_norm[:len(b_win)]
                elif s > 0:
                    b_win = beat_norm[:-s]
                    t_win = template_norm[s:]
                else:
                    b_win = beat_norm
                    t_win = template_norm

                if len(b_win) < 4:
                    continue

                corr = np.sum(b_win * t_win) / (np.sqrt(np.sum(b_win**2) * np.sum(t_win**2)) + 1e-12)
                if corr > best_corr:
                    best_corr = corr
                    best_shift = s

            shifts.append(best_shift)

            if best_shift == 0:
                aligned_p[i] = beat
            elif best_shift < 0:
                aligned_p[i] = np.pad(beat[-best_shift:], (0, -best_shift), mode="edge")
            else:
                aligned_p[i] = np.pad(beat[:-best_shift], (best_shift, 0), mode="edge")

        mean_abs_shift_ms = float(np.mean(np.abs(shifts)) / self.fs * 1000.0)
        return aligned_p, mean_abs_shift_ms, True

    def calculate_pmv_from_ensemble(
        self,
        p_waves: np.ndarray,
        fixed_reference_energy: Optional[float] = None
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute sample-wise MAD energy D, fixed reference energy Ei, and PMV V.
        
        Returns:
            D: absolute MAD energy: sum d_k(n)^2
            Ei: fixed reference energy: median_j { sum p_j(n)^2 } across reference segment
            V: published PMV = D / Ei
            Em: local template energy = sum m_k(n)^2
            mad_mean: mean(d_k(n))
            mad_max: max(d_k(n))
        """
        if len(p_waves) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Sample-wise median P-wave: m_k(n) = median_i { p_i(n) }
        m_k = np.median(p_waves, axis=0)

        # Sample-wise MAD: d_k(n) = median_i { |p_i(n) - m_k(n)| }
        abs_diffs = np.abs(p_waves - m_k[np.newaxis, :])
        d_k = np.median(abs_diffs, axis=0)

        # Absolute MAD energy: D(k) = sum(d_k(n)^2)
        D = float(np.sum(d_k**2))

        # Local template energy
        Em = float(np.sum(m_k**2))

        # Denominator Ei: require valid fixed reference energy; NO silent fallback!
        if fixed_reference_energy is not None and not np.isnan(fixed_reference_energy) and fixed_reference_energy > 1e-6:
            Ei = float(fixed_reference_energy)
            V = float(D / Ei)
        else:
            Ei = np.nan
            V = np.nan

        mad_mean = float(np.mean(d_k))
        mad_max = float(np.max(d_k))

        return D, Ei, V, Em, mad_mean, mad_max

    def compute_comparators(
        self,
        p_waves: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Compute conventional P-wave comparators across the subsegment beats."""
        if len(p_waves) < 2:
            return np.nan, np.nan, np.nan, np.nan

        ptp_amps = np.ptp(p_waves, axis=-1)
        amp_iqr = float(np.percentile(ptp_amps, 75) - np.percentile(ptp_amps, 25))
        amp_std = float(np.std(ptp_amps))

        durations = []
        for beat in p_waves:
            thresh = 0.10 * np.max(np.abs(beat))
            active = np.where(np.abs(beat) >= thresh)[0]
            dur_s = (active[-1] - active[0] + 1) / self.fs if len(active) > 0 else 0.0
            durations.append(dur_s)
        dur_std = float(np.std(durations))

        arc_lengths = np.sum(np.abs(np.diff(p_waves, axis=-1)), axis=-1)
        arc_std = float(np.std(arc_lengths))

        return amp_iqr, amp_std, dur_std, arc_std

    def compute_subsegment_pmv(
        self,
        ecg: np.ndarray,
        qrs_indices: np.ndarray,
        time_center_s: float,
        fixed_eigenvector: Optional[np.ndarray] = None,
        fixed_reference_energy: Optional[float] = None,
    ) -> PMVSubsegmentResult:
        """Compute full PMV metrics on a 120s subsegment using fixed reference energy Ei."""
        if len(qrs_indices) >= 2:
            rr_intervals_s = np.diff(qrs_indices) / self.fs
            mean_rr_ms = float(np.mean(rr_intervals_s) * 1000.0)
            sdnn_ms = float(np.std(rr_intervals_s) * 1000.0)
            rmssd_ms = float(np.sqrt(np.mean(np.diff(rr_intervals_s)**2)) * 1000.0) if len(rr_intervals_s) >= 2 else 0.0
            mean_hr_bpm = float(60.0 / (np.mean(rr_intervals_s) + 1e-12))
        else:
            mean_rr_ms, sdnn_ms, rmssd_ms, mean_hr_bpm = np.nan, np.nan, np.nan, np.nan

        # P-wave extraction
        if ecg.ndim == 2 and ecg.shape[0] == 2:
            p_waves, valid_qrs, lambda2_ratio, u1 = self.compute_spatial_pca_projection(
                ecg, qrs_indices, fixed_eigenvector=fixed_eigenvector
            )
        else:
            lead = ecg if ecg.ndim == 1 else ecg[0]
            p_waves, valid_qrs = self.extract_p_wave_ensemble(lead, qrs_indices)
            lambda2_ratio = np.nan

        num_beats = len(p_waves)
        if num_beats < self.min_beats:
            return PMVSubsegmentResult(
                time_center_s=time_center_s,
                num_beats=num_beats,
                is_valid=False,
                status=f"INSUFFICIENT_BEATS_{num_beats}_LT_{self.min_beats}",
                mad_energy_D=np.nan,
                ref_energy_Ei=np.nan,
                pmv_V=np.nan,
                local_median_energy_Em=np.nan,
                pmv_mad_amplitude_mean=np.nan,
                pmv_mad_amplitude_max=np.nan,
                pr_aligned=False,
                pmv_aligned_D=np.nan,
                pmv_aligned_V=np.nan,
                mean_abs_pr_shift_ms=np.nan,
                p_amp_dispersion_iqr=np.nan,
                p_amp_dispersion_std=np.nan,
                p_duration_dispersion_std=np.nan,
                p_arc_length_dispersion_std=np.nan,
                pca_lambda2_ratio=lambda2_ratio,
                mean_rr_ms=mean_rr_ms,
                sdnn_ms=sdnn_ms,
                rmssd_ms=rmssd_ms,
                mean_hr_bpm=mean_hr_bpm,
            )

        # Standard PMV with fixed reference energy Ei
        D, Ei, V, Em, mad_mean, mad_max = self.calculate_pmv_from_ensemble(
            p_waves, fixed_reference_energy=fixed_reference_energy
        )

        # PR alignment
        if self.enable_pr_alignment:
            m_k = np.median(p_waves, axis=0)
            aligned_p, mean_shift_ms, did_align = self.align_pr_jitter(p_waves, m_k)
            if did_align:
                D_align, _, V_align, _, _, _ = self.calculate_pmv_from_ensemble(
                    aligned_p, fixed_reference_energy=fixed_reference_energy
                )
            else:
                D_align, V_align = D, V
        else:
            did_align = False
            mean_shift_ms = 0.0
            D_align, V_align = D, V

        amp_iqr, amp_std, dur_std, arc_std = self.compute_comparators(p_waves)

        # Enforce validity: must have valid fixed reference and non-null values
        if fixed_reference_energy is None or np.isnan(fixed_reference_energy) or fixed_reference_energy < 1e-6:
            is_valid = False
            status = "MISSING_FIXED_REFERENCE"
        elif V is None or np.isnan(V) or np.isinf(V):
            is_valid = False
            status = "INVALID_PMV_NAN_OR_INF"
        else:
            is_valid = True
            status = "VALID"

        return PMVSubsegmentResult(
            time_center_s=time_center_s,
            num_beats=num_beats,
            is_valid=is_valid,
            status=status,
            mad_energy_D=D,
            ref_energy_Ei=Ei,
            pmv_V=V,
            local_median_energy_Em=Em,
            pmv_mad_amplitude_mean=mad_mean,
            pmv_mad_amplitude_max=mad_max,
            pr_aligned=did_align,
            pmv_aligned_D=D_align,
            pmv_aligned_V=V_align,
            mean_abs_pr_shift_ms=mean_shift_ms,
            p_amp_dispersion_iqr=amp_iqr,
            p_amp_dispersion_std=amp_std,
            p_duration_dispersion_std=dur_std,
            p_arc_length_dispersion_std=arc_std,
            pca_lambda2_ratio=lambda2_ratio,
            mean_rr_ms=mean_rr_ms,
            sdnn_ms=sdnn_ms,
            rmssd_ms=rmssd_ms,
            mean_hr_bpm=mean_hr_bpm,
        )

    def extract_sliding_pmv_series(
        self,
        ecg: np.ndarray,
        qrs_indices: np.ndarray,
        start_time_s: float = 0.0,
        end_time_s: Optional[float] = None,
        fixed_eigenvector: Optional[np.ndarray] = None,
        fixed_reference_energy: Optional[float] = None,
    ) -> List[PMVSubsegmentResult]:
        """Extract continuous PMV time series across a recording segment.
        
        Slides 120s windows every 60s, using fixed spatial PCA and fixed reference energy Ei.
        """
        n_samples = ecg.shape[-1]
        total_duration_s = n_samples / self.fs
        if end_time_s is None or end_time_s > total_duration_s:
            end_time_s = total_duration_s

        results = []
        cur_start_s = start_time_s
        while cur_start_s + self.subsegment_len_s <= end_time_s + 1e-6:
            cur_end_s = cur_start_s + self.subsegment_len_s
            time_center_s = 0.5 * (cur_start_s + cur_end_s)

            start_idx = int(round(cur_start_s * self.fs))
            end_idx = int(round(cur_end_s * self.fs))

            sub_qrs_mask = (qrs_indices >= start_idx) & (qrs_indices < end_idx)
            sub_qrs = qrs_indices[sub_qrs_mask]

            if ecg.ndim == 1:
                ecg_sub = ecg[start_idx:end_idx]
            else:
                ecg_sub = ecg[:, start_idx:end_idx]

            rel_qrs = sub_qrs - start_idx

            res = self.compute_subsegment_pmv(
                ecg_sub,
                rel_qrs,
                time_center_s=time_center_s,
                fixed_eigenvector=fixed_eigenvector,
                fixed_reference_energy=fixed_reference_energy,
            )
            results.append(res)

            cur_start_s += self.subsegment_step_s

        return results
