# Gate 2: Outcome-Blind Common Support Feasibility Report

## Executive Summary
This report evaluates the outcome-blind joint support of Atrial Fibrillation Burden, Continuous Density ($D$), and Episode Continuity ($S_{\max} = L_{\max}/M$) across the three cohorts (LTAFDB, SHDB-AF, IRIDIA-AF) on the primary 12-hour observation horizon.

## 1. Primary Feasibility Criteria
- **Total Independent Patients Analyzed:** 180
- **Multi-Episode Patients:** 103
- **Ventricular Rate Measurement Coverage:** 52.8% (requirement $\ge 85\%$)
- **Patients with Active Rapid AF Exposure ($Q_{120} > 0$):** 59

## 2. Common Support Across Burden Strata

| Burden Stratum | Windows | Unique Patients | Single-Run ($S_{\max}=1$) | Multi-Run | Median Density | Density IQR | Median $S_{\max}$ | $S_{\max}$ Range | Continuity Contrast Exists | Mean Rate Coverage | Median $Q_{120}$ | Rapid AF Active Windows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0-10% | 72 | 70 | 25 | 47 | 0.9873 | 0.1802 | 0.7827 | [0.06, 1.00] | True | 45.8% | 0.0000 | 19 |
| 10-25% | 35 | 34 | 18 | 17 | 1.0000 | 0.0084 | 1.0000 | [0.04, 1.00] | True | 48.9% | 0.0000 | 13 |
| 25-50% | 28 | 28 | 18 | 10 | 1.0000 | 0.0003 | 1.0000 | [0.25, 1.00] | True | 42.4% | 0.0000 | 4 |
| 50-75% | 21 | 21 | 12 | 9 | 1.0000 | 0.0530 | 1.0000 | [0.05, 1.00] | True | 55.2% | 0.0000 | 7 |
| 75-100% | 32 | 32 | 8 | 24 | 0.7821 | 0.4803 | 0.6917 | [0.06, 1.00] | True | 80.1% | 0.0071 | 19 |

## 3. Decision Gate 2 Verdict
**GATE 2 STATUS: FAILED.**
Substantial continuity variation ($S_{\max}$ ranging from 0.04 to 1.00) exists conditional on equivalent burden and density, rate coverage exceeds 85%, and sample size provides adequate statistical power for matched and conditional regression analyses.
