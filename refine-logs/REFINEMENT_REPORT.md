# Refinement Report: Methodological Evolutions & Architectural Synthesis

## 1. Evolution of the Methodological Thesis
The research plan evolved through iterative refinement from a generic operator-learning formulation into a clinically targeted, noise-resilient post-ablation surveillance system:

1. **Initial Concept:** Generic SetOperator for continuous lead-transformation and invariant spatial ECG reconstruction.
   - *Limitation Identified:* While theoretically sound for spatial vectorcardiography, standard spatial dipole models do not inherently capture temporal rhythm irregularity (the primary clinical hallmark of atrial fibrillation).
2. **Intermediate Pivot:** Incorporating temporal RR intervals via independent dual streams.
   - *Limitation Identified:* Processing waveforms and RR intervals separately led to contradictory predictions during paroxysmal episodes with borderline ventricular rates.
3. **Final Refined Architecture:** **Quality-Aware Multimodal SetOperator (Phase-KME + Sequential RR + Selective Abstention Gate).**
   - *Key Synthesis:* Cross-attention directly couples phase-normalized P-wave morphology with sequential RR interval dynamics. Furthermore, integrating Chow's selective classification using bSQI and prediction entropy provides mathematical guarantees against noise-induced false alarms.

---

## 2. Explicit Ablation Decisions

| Ablation Component | Alternative Tested | Final Architectural Choice | Clinical & Theoretical Justification |
| :--- | :--- | :--- | :--- |
| **Morphological Representation** | Raw 1D-CNN vs. Phase-KME | **Phase-KME (16 phase bins)** | Phase normalization achieves heart-rate invariance and eliminates baseline wander drift while isolating P-wave absence and f-waves. |
| **Rhythm Modeling** | Aggregate HRV metrics vs. Sequential RR | **Sequential 1D-CNN / BiLSTM on RR sequence** | Sequential models preserve local tachyarrhythmia dynamics and sudden onset/offset patterns essential for episode bounding. |
| **Multimodal Fusion** | Late Logit Averaging vs. Cross-Attention | **Cross-Modal Multi-Head Attention** | Enables morphological tokens (absence of P-waves) to dynamically attend to and modulate irregular RR tokens. |
| **Noise Handling** | Noise Denoising Filter vs. Selective Abstention | **Selective Classification Gate (Chow's Rule)** | Filtering severe N2 noise distorts authentic ECG morphology; safe abstention eliminates false alarms without introducing reconstruction artifacts. |

---

## 3. Methodological Guardrails
- **Zero Training Contamination:** Models are benchmarked directly in zero-shot transfer mode using pre-trained weights (AFtoolkit SavedModels for ArNet2/RawECGNet; SetOperator checkpoints) on completely unseen external patient cohorts (SHDB-AF and Noise Holter).
- **Consensus Blanking Enforcement:** Analysis code explicitly segregates recordings by days elapsed from first ablation ($>90$ days vs. $\le 90$ days).
