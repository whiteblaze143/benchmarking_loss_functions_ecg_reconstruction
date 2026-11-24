# Literature Grounding: Temporal Concentration, Continuity, and Electrophysiological Heterogeneity in Atrial Fibrillation

## Executive Summary
This document provides the mandatory literature grounding for **AF-DENSITY V5**, mapping published mathematical formulations, observation clocks, and clinical evidence across 11 key publications:

1. **Charitos et al. (2012, 2013)** established the mathematical formulation of Atrial Fibrillation Density ($D_{\text{AF}}$) as a normalized area between the empirical minimum-duration function $F(p; b)$ and the uniform concentration reference $p$:
   $$D = \frac{2}{1-b} \int_0^1 |F(p; b) - p|\, dp$$
   The metric was formulated for continuous time but implemented numerically in discrete time bins (e.g., daily or hourly device records).
2. **Henriksson et al. (2021)** and **Butkuvienė et al. (2023, 2024)** modeled and evaluated PAF episode temporal patterns, demonstrating that AF episode duration and inter-episode gap distributions vary widely across patients with identical burden. Butkuvienė et al. benchmarked pattern metrics (density, aggregation, longest episode), observing that density does not distinguish continuous runs from fragmented bursts.
3. **ESC/EHRA Clinical Consensus (2025)** emphasized that while overall burden ($M/T$) tracks aggregate arrhythmia load, episode duration (e.g., episodes $\ge 5.5$ hours vs micro-episodes) conveys distinct clinical risks.
4. **JMIR Cardio (2025)** and **Guo et al. (2026)** introduced multidimensional AF characterization incorporating both temporal concentration ($D$) and ventricular tachycardia burden ($r(t) > 120$ bpm). However, they treated these metrics as separate additive variables without asking: (a) whether continuous continuity varies at equivalent burden and density, and (b) whether rapid ventricular response temporally co-localizes with the density-defining intervals.
5. **Rosman et al. (2026)** demonstrated in a major CIED cohort that AF density predicts ischaemic stroke independently of AF burden. In response, **Liang (2026)** argued that density must be tested head-to-head against simpler episode metrics (such as the longest uninterrupted AF episode $L_{\max}$ or episode count), while **Rosman, Wang, and Passman (2026)** clarified the software implementation conventions and aggregation limits.

## Core Unresolved Scientific Questions Addressed by V5
- **Measurement Identifiability (RQ1):** Daily and hourly device registers measure only the total AF duration $m_j$ in bin $j$. What mathematical bounds exist on continuous density, and how much ambiguity remains regarding the underlying continuity ($S_{\max} = L_{\max}/M$) when continuous ECG annotations are reduced to bin-level durations?
- **Electrophysiological Heterogeneity (RQ2):** Conditional on comparable AF burden ($b \pm 0.025$) and AF density ($D \pm 0.025$), does a patient with an uninterrupted AF episode ($S_{\max} \approx 1$) exhibit a different rapid ventricular response fraction ($Q_{120} = R_{120}/M$) compared to a patient with fragmented AF ($S_{\max} \ll 1$)?
- **Temporal Localization (RQ3):** Do the physical-time intervals $I_p$ that contain 80% of AF exposure also contain 80% of rapid ventricular response ($r(t) > 120$ bpm), or does rapid AF decouple temporally from AF concentration?
