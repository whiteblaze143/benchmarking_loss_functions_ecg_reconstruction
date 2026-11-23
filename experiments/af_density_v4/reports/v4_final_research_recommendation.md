# T-CAIREM V4: Final Research Recommendation & Transition Roadmap

## To: Scientific Advisory Committee & T-CAIREM Leadership
**Program:** Continuous SetOperator / AF-Density Research  
**Subject:** Program Conclusion, Architectural Pivot, and Clinical Transition Recommendation  

---

### 1. Core Scientific Verdict
The transition from binary AF classification to the **measurement science of temporal AF pattern reconstruction** has established three definitive principles:

1. **Rhythm Irregularity is the Primary Signal of Paroxysmal AF:**
   Standalone Gradient-Boosted Decision Trees (`RR-GBDT`) trained on 36-D local RR interval statistics achieve state-of-the-art discrimination across 504,777 windows (AUROC 0.9911, BAcc 0.9629, Brier 0.0212, BSS +0.8853). It outperforms all proposed neural continuous SetOperators and deep convolutional ResNets across both clean and noisy Holters.
2. **Charitos AF Density Captures Orthogonal Disease Phenotypes:**
   In clinical cohorts with identical AF burden (e.g. $b \approx 15\%$, density ranges from 0.30 to 1.00 (discordance ratio up to 9.98). Reporting burden alone conceals whether a patient suffers from 50 highly fragmented micro-episodes or one single sustained episode.
3. **Continuous SetOperator Architecture on Unaligned Waveforms is a Non-Viable Pathway:**
   The speculative continuous SetOperator ($A001$) fails to beat chance within patients (AUROC 0.5905). Shuffling morphology in multimodal $G061$ leaves AUROC unchanged at 0.8686, while removing RR destroys performance (AUROC 0.5057). Furthermore, $G061$ suffers from catastrophic lead asymmetry (AUROC 0.1885 on Lead 1).

---

### 2. Operational Recommendation for Clinical Deployment (Policy Q2 / Q3)
We recommend deploying **Policy Q3 (Continuous Quality-Weighted Fusion)**:
- Primary Engine: Standalone `RR-GBDT` with R-peak validity gating.
- Waveform Fallback: 1D Convolutional ResNet (`RawECGNet`) seamlessly weighted when signal quality index drops below 0.60.
- Quality Reporting: Mandate selective density reporting with **Density Identifiability Width ($[D_{\min}, D_{\max}]$)** whenever observation completeness falls below 80%. Never impute missing time with small epsilons or assume sinus rhythm.

### 3. Protected Test Partition Status
The protected IRIDIA held-out test partition (`test_windows.npz`, 32 patients, 131,667 windows) remains **STRICTLY QUARANTINED ON HOLD**. Zero test data has been exposed, preserving uncompromised integrity for final prospective validation.
