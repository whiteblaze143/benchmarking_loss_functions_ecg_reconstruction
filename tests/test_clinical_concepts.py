#!/usr/bin/env python3
"""Unit tests for Module C: Clinical Concepts & Observability Auditing."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from experiments.echonext_operator_benchmark.measurement_information import (
    compute_noiseless_observable_np,
)
from experiments.echonext_operator_benchmark.clinical_concepts import (
    audit_concepts_observability,
    extract_reference_ecg_concepts,
    CLINICAL_CONCEPTS,
    CONCEPT_KEYS,
)


def test_concept_observability_lead_configurations():
    """Verifies that observability mathematically distinguishes between limb and precordial setups."""
    d = 8
    
    # 1. Full 8 Canonical Leads (Q_8 = I_8)
    Q_8 = np.eye(8, dtype=np.float32)
    P_8 = Q_8.T @ Q_8
    audit_8 = audit_concepts_observability(P_8)
    for k in CONCEPT_KEYS:
        assert audit_8[k]["status"] == "OBSERVABLE", f"{k} should be observable under Q_8"
        assert np.isclose(audit_8[k]["nullspace_residue"], 0.0, atol=1e-5)

    # 2. Limb Leads Only (Leads I, II -> rows 0 and 1)
    Q_limb = np.zeros((2, 8), dtype=np.float32)
    Q_limb[0, 0] = 1.0  # Lead I
    Q_limb[1, 1] = 1.0  # Lead II
    P_limb = Q_limb.T @ Q_limb
    audit_limb = audit_concepts_observability(P_limb)
    
    # Frontal axis MUST be fully observable from limb leads
    assert audit_limb["frontal_qrs_axis_deg"]["status"] == "OBSERVABLE"
    assert np.isclose(audit_limb["frontal_qrs_axis_deg"]["observable_fraction"], 1.0, atol=1e-5)
    
    # Anterior ST elevation and precordial R progression MUST be UNOBSERVABLE from limb leads
    assert audit_limb["st_elevation_v2_v3_uv"]["status"] == "UNOBSERVABLE"
    assert np.isclose(audit_limb["st_elevation_v2_v3_uv"]["nullspace_residue"], 1.0, atol=1e-5)
    
    assert audit_limb["precordial_r_progression"]["status"] == "UNOBSERVABLE"
    assert np.isclose(audit_limb["precordial_r_progression"]["nullspace_residue"], 1.0, atol=1e-5)

    # 3. Precordial Leads Only (V1..V6 -> rows 2..7)
    Q_precordial = np.zeros((6, 8), dtype=np.float32)
    for i in range(6):
        Q_precordial[i, i + 2] = 1.0
    P_precordial = Q_precordial.T @ Q_precordial
    audit_prec = audit_concepts_observability(P_precordial)
    
    # Precordial concepts MUST be fully observable
    assert audit_prec["st_elevation_v2_v3_uv"]["status"] == "OBSERVABLE"
    assert audit_prec["precordial_r_progression"]["status"] == "OBSERVABLE"
    
    # Frontal axis MUST be UNOBSERVABLE from chest leads alone
    assert audit_prec["frontal_qrs_axis_deg"]["status"] == "UNOBSERVABLE"
    assert np.isclose(audit_prec["frontal_qrs_axis_deg"]["nullspace_residue"], 1.0, atol=1e-5)


def test_reference_concept_extractor_simulation():
    """Verifies that extract_reference_ecg_concepts extracts plausible numbers from synthetic heartbeat."""
    sampling_rate = 250.0
    T = 2500  # 10 seconds
    t = np.arange(T) / sampling_rate
    
    # Construct a synthetic multi-lead ECG with 1.0s R-R intervals (60 bpm)
    ecg = np.zeros((8, T), dtype=np.float32)
    # 10 beats at t = 0.5, 1.5, 2.5, ..., 9.5
    beat_times = np.arange(0.5, 9.5, 1.0)
    for tb in beat_times:
        ib = int(tb * sampling_rate)
        # QRS spike (+1.0 mV on Lead II, +1.2 mV on V5, +0.3 mV on V1)
        w_qrs = int(0.05 * sampling_rate)  # 50ms half-width -> ~100ms QRS
        if ib - w_qrs >= 0 and ib + w_qrs < T:
            # Lead I
            ecg[0, ib - w_qrs : ib + w_qrs] += 0.8 * np.hanning(2 * w_qrs)
            # Lead II
            ecg[1, ib - w_qrs : ib + w_qrs] += 1.0 * np.hanning(2 * w_qrs)
            # V1
            ecg[2, ib - w_qrs : ib + w_qrs] += 0.3 * np.hanning(2 * w_qrs)
            # V2
            ecg[3, ib - w_qrs : ib + w_qrs] += 0.6 * np.hanning(2 * w_qrs)
            # V3
            ecg[4, ib - w_qrs : ib + w_qrs] += 0.9 * np.hanning(2 * w_qrs)
            # V5
            ecg[6, ib - w_qrs : ib + w_qrs] += 1.4 * np.hanning(2 * w_qrs)
            # V6
            ecg[7, ib - w_qrs : ib + w_qrs] += 1.1 * np.hanning(2 * w_qrs)
            
            # T wave: 250ms after R peak, 120ms duration
            it = ib + int(0.25 * sampling_rate)
            wt = int(0.08 * sampling_rate)
            if it + wt < T:
                ecg[1, it - wt : it + wt] += 0.3 * np.hanning(2 * wt)
                ecg[6, it - wt : it + wt] += 0.4 * np.hanning(2 * wt)

    concepts = extract_reference_ecg_concepts(ecg, sampling_rate=sampling_rate)
    
    # Check values
    assert 55.0 <= concepts["heart_rate_bpm"] <= 65.0, f"Heart rate: {concepts['heart_rate_bpm']}"
    assert 60.0 <= concepts["qrs_duration_ms"] <= 130.0, f"QRS duration: {concepts['qrs_duration_ms']}"
    assert 300.0 <= concepts["qtc_interval_ms"] <= 550.0, f"QTc: {concepts['qtc_interval_ms']}"
    assert -90.0 <= concepts["frontal_qrs_axis_deg"] <= 90.0, f"Frontal axis: {concepts['frontal_qrs_axis_deg']}"
    assert concepts["precordial_r_progression"] > 1.0, f"R progression: {concepts['precordial_r_progression']}"
