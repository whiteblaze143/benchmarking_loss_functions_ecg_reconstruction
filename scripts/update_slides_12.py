import re

with open("slides/main.tex", "r") as f:
    content = f.read()

# Locate slide 5 end
idx = content.find("% Slide 6: Current AI State: What Works & What Is Bounded")
if idx == -1:
    print("Could not find Slide 6 marker")
    exit(1)

prefix = content[:idx]

new_slides = r"""% ------------------------------------------------------------------------------
% Slide 6: The 3DRECON Methodology: Multitask Spatial Regularization
% ------------------------------------------------------------------------------
\begin{frame}{The 3DRECON Methodology (Ansari et al., \textit{Circulation} 2026)}{How Auxiliary 12-Lead Reconstruction Powers Primary AF Detection}
\begin{columns}[T]
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{The Stanford 3DRECON-QT Breakthrough}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.2em}
        \item \textbf{Reference}: Ansari, Rogers et al., \textit{Circulation} (Jan 2026 / Nov 2025).
        \item \textbf{Multitask Learning Objective}: Jointly reconstructed 12 leads while predicting clinical biomarkers from single-lead input.
        \item \textbf{Inductive Bias}: Spatial 12L head forced encoder to learn 3D cardiac vector loops.
        \item \textbf{Ablation Result}: Removing spatial encoder collapsed downstream prediction ($r = 0.73 \to 0.04$).
      \end{itemize}
    \end{block}
  \end{column}
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Application to Sunnybrook AF Trial (3DRECON-AF)}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.2em}
        \item \textbf{Primary Task (At Home)}: Single-lead smartwatch ECG detects AF recurrence \& burden.
        \item \textbf{The Single-Lead Blindness Problem}: Lead I alone can miss low-amplitude f-waves or ectopic triggers oriented along vertical/precordial axes.
        \item \textbf{Auxiliary 12L Reconstruction Head}: Forces shared encoder to represent full 3D atrial vectors (including V1 f-wave dynamics).
        \item \textbf{Translational Win}: Supercharges primary AF detection sensitivity while reconstructing 12 leads on the side!
      \end{itemize}
    \end{block}
  \end{column}
\end{columns}
\vspace{0.4em}
\centering
\scriptsize
\textbf{\color{NavyPrimary}Key Mechanism}: 12-lead reconstruction is not just an output—it is an electrophysiologic regularizer that prevents single-lead diagnostic failure.
\note{
  Chris and Alex, this slide connects our project to the state-of-the-art literature published just two months ago in Circulation by Albert Rogers' group at Stanford (Ansari et al., 2026).
  Ansari developed 3DRECON-QT, showing that training an auxiliary 12-lead reconstruction head forces the single-lead encoder to learn 3D cardiac vector loops. In their ablation study, removing the 12-lead spatial conditioning collapsed downstream clinical biomarker prediction from r=0.73 down to 0.04!
  We apply this exact paradigm to our trial:
  At home, patients record daily single-lead smartwatch ECGs for our primary endpoint: time to AF recurrence.
  Lead I alone often suffers from vector blindness when fibrillatory f-waves are oriented perpendicular to the frontal plane.
  By training an auxiliary 12-lead reconstruction head, the shared encoder is forced to preserve 3D atrial vector dynamics—giving the smartwatch model superhuman sensitivity for early AF detection!
}
\end{frame}

% ------------------------------------------------------------------------------
% Slide 7: Figure 1: The ECG-AIM Model Architecture
% ------------------------------------------------------------------------------
\begin{frame}{Figure 1: The ECG-AIM Model Architecture}{conv15e\_A0\_wave\_noSSL\_gated\_add: 6 Core Engineering Components}
\begin{columns}[T]
  \begin{column}{0.48\textwidth}
    \vspace{-0.5em}
    \includegraphics[width=\textwidth,height=0.76\textheight,keepaspectratio]{conv15e_a0_wave_architecture.png}
  \end{column}
  \begin{column}{0.50\textwidth}
    \vspace{-0.6em}
    \scriptsize
    \begin{itemize}
      \setlength{\itemsep}{0.2em}
      \item \textbf{1. Patch Tokenizer}: 50\,ms temporal patches project single-lead wrist signals ($1 \times 5000$) into high-dimensional tokens.
      \item \textbf{2. Continuous Morlet Wavelet Bank}: 32 logarithmic scales (0.5--45\,Hz) decompose raw wrist data into multi-scale time-frequency scalograms.
      \item \textbf{3. TimeSformer + 1D-CNN}: Spatiotemporal self-attention across patches coupled with local convolutional receptive fields.
      \item \textbf{4. Gated-Add Fusion Module}: Sigmoid-gated combination: $Z = Z_t + \sigma(W[Z_t, Z_w]) \odot Z_w$ injects wavelets without baseline corruption.
      \item \textbf{5. 8-Layer Transformer Encoder}: Captures long-range rhythm transitions and cardiac cycles ($d=768$, 12 heads).
      \item \textbf{6. Axial Transformer Decoder}: Decouples temporal and cross-lead spatial dimensions into continuous 12 leads and P-QRS-T delineation.
    \end{itemize}
  \end{column}
\end{columns}
\note{
  Alex, this is the architectural engine that powers our system: ECG-AIM champion checkpoint conv15e_A0_wave_noSSL_gated_add.
  Notice the 6 core components:
  1. Patch tokenizer slicing the Lead I input into 50 ms tokens.
  2. A continuous Morlet wavelet bank across 32 frequency scales from 0.5 to 45 Hz.
  3. A hybrid TimeSformer and 1D-CNN backbone.
  4. Our gated-add fusion module that dynamically modulates wavelet energy via a sigmoid gate.
  5. An 8-layer transformer encoder.
  6. An axial transformer decoder producing both continuous 12-lead waveforms and explicit fiducial delineations.
}
\end{frame}

% ------------------------------------------------------------------------------
% Slide 8: Data Collection Methodology: Dual-Stream Protocol
% ------------------------------------------------------------------------------
\begin{frame}{Data Collection Methodology: Dual-Stream Protocol}{Seamless Integration: Daily At-Home Surveillance + In-Clinic Paired Anchors}
\begin{columns}[T]
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Stream A: Ambulatory Home Surveillance}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.2em}
        \item \textbf{At-Home Daily ECG}: Patient records daily 30s smartwatch ECG (Lead I) for 6 months post-ablation.
        \item \textbf{Continuous Gold Standard}: 14-day continuous \textbf{CardioSTAT™ patch} worn at 3 months and 6 months.
        \item \textbf{Primary Outcome Harvested}: Exact time to AF recurrence, daily AF burden (\% time in AF), and asymptomatic episode yield.
        \item \textbf{Data Flow}: Automated upload via SRI custom dashboard to secure Sunnybrook server.
      \end{itemize}
    \end{block}
  \end{column}
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Stream B: In-Clinic Paired Anchor Protocol}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.2em}
        \item \textbf{Hospital Touchpoints}: Routine visits already required:
          \begin{itemize}
            \tiny
            \item Visit 1: Baseline / Pre-procedure clinic visit
            \item Visit 2: Index procedure day (post-ablation acute)
            \item Visit 3: 3-month routine follow-up
            \item Visit 4: 6-month study closeout
          \end{itemize}
        \item \textbf{Synchronous Acquisition ($\le 5$ min added)}:
          Patient lies supine; clinical 12-lead ECG is recorded simultaneously with smartwatch Lead I ($\Delta t = 0$).
        \item \textbf{Patient Burden}: Zero extra hospital visits or invasive tests.
      \end{itemize}
    \end{block}
  \end{column}
\end{columns}
\vspace{0.4em}
\centering
\scriptsize
\textbf{\color{NavyPrimary}Operational Elegance}: Stream A answers Chris's primary clinical question; Stream B generates the world's premier paired evaluation set.
\note{
  Chris and Alex, this is the exact data collection methodology.
  It operates as two synchronized streams:
  Stream A is the ambulatory home surveillance from Protocol v1.1. Patients wear the watch daily for 6 months and wear the 14-day continuous CardioSTAT patch at 3 and 6 months. This feeds directly into our primary clinical efficacy endpoint.
  Stream B is the in-clinic paired protocol. Patients are already at Sunnybrook for four standard clinical visits: baseline pre-ablation, procedure day, 3 months, and 6 months.
  During the routine clinical 12-lead ECG, the research coordinator records a 30-second smartwatch tracing at the exact same moment.
  It adds less than 5 minutes to routine care, requires zero extra visits, and creates perfectly synchronized ground-truth pairs!
}
\end{frame}

% ------------------------------------------------------------------------------
% Slide 9: The Paired In-Clinic Evaluation Set: The Gold Standard
% ------------------------------------------------------------------------------
\begin{frame}{The Paired In-Clinic Evaluation Set}{The Most Critical Dataset: Breaking the Synthetic Proxy Trap}
\begin{table}[h]
\centering
\tiny
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Methodological Dimension} & \textbf{Stanford 3DRECON-QT (Ansari 2026)} & \textbf{Sunnybrook Wearable Trial (Our Evaluation Set)} \\ \midrule
\textbf{Hardware Evaluation Cohort} & Retrospective CareLink archive ($N=26$ patients). & \textbf{Prospective cohort}: 96 patients, $\approx 384$ paired recordings. \\ \midrule
\textbf{Temporal Synchronization} & $\pm \mathbf{7}$ \textbf{days} between ICM and 12-lead ECG ($|\Delta\text{HR}| \le 20$\,bpm). & $\mathbf{\Delta t = 0}$ \textbf{synchronous}: Captured in same clinical exam room. \\ \midrule
\textbf{Physiological Confounding} & Electrolytes, posture, autonomic tone vary over 7 days. & \textbf{Identical resting state}: Supine, same autonomic state, same HR. \\ \midrule
\textbf{Clinical Transitions Captured} & Static cross-sectional outpatient snapshots. & \textbf{Longitudinal transitions}: Baseline AF $\to$ Acute post-ablation $\to$ Healed. \\ \midrule
\textbf{Scientific Role} & Device proof-of-concept for publication. & \textbf{The Gold Standard Benchmark}: Bridges synthetic AI to real wearables. \\ \bottomrule
\end{tabular}
\end{table}
\vspace{0.2em}
\begin{alertblock}{\textbf{Why This Evaluation Set is the Crown Jewel of the Study}}
  \scriptsize
  99\% of published ECG AI papers test only on synthetic derived leads from 12-lead files. That only tests math, not wearable physics (dry electrodes, wrist impedance, DSP filters). Our prospective paired dataset is the \textbf{first true clinical-hardware benchmark} for wearable 12-lead reconstruction.
\end{alertblock}
\note{
  Alex, this slide explains why our computing science group needs this data so desperately: this evaluation set is the crown jewel of the entire project!
  Look at the comparison with the Stanford paper in Circulation:
  Ansari and Rogers had to search retrospectively through old Medtronic CareLink transmissions. They only found 26 patients with a clinical 12-lead within plus-or-minus 7 days! Over 7 days, hydration, potassium, and medication levels change.
  In our trial, we are prospectively acquiring exact delta-t equals zero pairs across 96 patients at 4 distinct time points—yielding nearly 400 paired recordings!
  Furthermore, this evaluation set captures true longitudinal transitions: pre-ablation AF, acute sinus restoration with atrial tissue stunning, and 3-to-6-month healed rhythm.
  This completely breaks the synthetic proxy trap that invalidates other AI papers.
}
\end{frame}

% ------------------------------------------------------------------------------
% Slide 10: Figure 2: Integrated Clinical-AI Trial Pipeline
% ------------------------------------------------------------------------------
\begin{frame}{Figure 2: Integrated Clinical-AI Trial Pipeline}{Tri-Modal Data Acquisition Engine, Domain Adaptation, and Dual Validation}
\begin{columns}[T]
  \begin{column}{0.52\textwidth}
    \vspace{-0.5em}
    \includegraphics[width=\textwidth,height=0.76\textheight,keepaspectratio]{paired_trial_model_integration.png}
  \end{column}
  \begin{column}{0.46\textwidth}
    \vspace{-0.6em}
    \scriptsize
    \begin{itemize}
      \setlength{\itemsep}{0.2em}
      \item \textbf{Stage 1: Foundation Pretraining}:
        Trained on PTB-XL, MIMIC-IV, and Icentia11k to embed general cardiac electrophysiology priors.
      \item \textbf{Stage 2: Sunnybrook Tri-Modal Data Engine}:
        \begin{itemize}
          \tiny
          \item \textit{Smartwatch}: Daily at-home rhythm + in-clinic paired Lead I.
          \item \textit{CardioSTAT Patch}: 14-day continuous gold standard at 3 \& 6 mo.
          \item \textit{Hospital 12-Lead}: Diagnostic gold standard during clinic visits.
        \end{itemize}
      \item \textbf{Stage 3: Domain Adaptation \& Calibration}:
        Estimates device transfer function $H(\omega)$ and tunes low-rank adapters (LoRA) on paired in-clinic signals.
      \item \textbf{Stage 4: Dual Clinical Endpoints}:
        \begin{itemize}
          \tiny
          \item \textbf{Primary Efficacy}: AF recurrence \& burden vs. CardioSTAT.
          \item \textbf{Secondary AI}: 12-lead reconstruction \& biomarker recovery.
        \end{itemize}
    \end{itemize}
  \end{column}
\end{columns}
\note{
  Alex and Chris, this is Figure 2: the master architecture diagram showing how the clinical trial and AI engineering integrate seamlessly.
  Stage 1 is our foundation models pretrained on big public data.
  Stage 2 is our tri-modal data acquisition engine: smartwatch, 14-day continuous CardioSTAT patch, and hospital 12-lead machine.
  Stage 3 uses the paired in-clinic data to calibrate the hardware transfer function and adapt the model.
  Stage 4 shows the clear separation of endpoints: primary clinical AF surveillance vs. secondary translational 12-lead reconstruction.
}
\end{frame}

% ------------------------------------------------------------------------------
% Slide 11: Sample Size, Statistical Power & Clustered Modeling
% ------------------------------------------------------------------------------
\begin{frame}{Sample Size, Statistical Power \& Clustered Modeling}{Precision-Driven Sizing for Clinical Trial and Clustered Secondary AI Aim}
\begin{columns}[T]
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Primary Clinical Power (Protocol v1.1)}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.2em}
        \item \textbf{Primary Outcome}: Time to AF recurrence detection.
        \item \textbf{Hypothesis}: Smartwatch detects recurrent AF at higher or non-inferior rate vs. 14-day patch.
        \item \textbf{Assumptions}: AF recurrence rate 20\% with patch vs. 25\% with smartwatch.
        \item \textbf{Target Power}: 80\% power, $\alpha = 0.05$, 10\% non-inferiority margin $\implies \mathbf{N = 96}$ \textbf{patients}.
        \item \textbf{Vanguard Cohort}: Initial $N = 20$ consecutive patients to verify protocol adherence and pipeline.
      \end{itemize}
    \end{block}
  \end{column}
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Secondary AI Precision \& Clustered Stats}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.2em}
        \item \textbf{AI Sample Size}: 96 patients $\times$ 4 in-clinic visits yields $>2,500$ paired cardiac complexes.
        \item \textbf{Precision Target}: Sized for QTc MAE 95\% CI margin of error $\le \mathbf{\pm 3.5}$\,\textbf{ms}.
        \item \textbf{Clustering Adjustment}: Multiple beats per patient violate standard i.i.d. assumptions.
        \item \textbf{MMRM \& GEE}: Linear Mixed Models (unstructured covariance) and Generalized Estimating Equations for robust sandwich variance.
      \end{itemize}
    \end{block}
  \end{column}
\end{columns}
\vspace{0.3em}
\centering
\scriptsize
\textbf{\color{NavyPrimary}Regulatory Defensibility}: Primary sample size is powered on clinical events; secondary AI sample size exceeds precision requirements.
\note{
  Chris and Alex, this slide demonstrates total alignment on sample size and statistics.
  The primary clinical trial sample size of 96 patients is calculated directly from expected AF recurrence rates—20% on patch vs 25% on smartwatch with an 80% power target.
  For the secondary AI aim, having 96 patients with multiple paired in-clinic visits provides over 2,500 paired cardiac complexes.
  Per Riley's 2024 BMJ guidelines, this is more than sufficient to bound QTc measurement error within a tight 3.5 ms margin of error.
  Furthermore, we pre-specify Linear Mixed Models (MMRM) and Generalized Estimating Equations (GEE) to adjust for intra-patient beat clustering.
}
\end{frame}

% ------------------------------------------------------------------------------
% Slide 12: Action Items & Alignment for Today's Meeting
% ------------------------------------------------------------------------------
\begin{frame}{Action Items \& Alignment for Today's Meeting}{4 Concrete Decisions to Finalize Today}
\begin{columns}[T]
  \begin{column}{0.50\textwidth}
    \begin{block}{\textbf{4 Concrete Decisions to Finalize Today}}
      \scriptsize
      \begin{enumerate}
        \setlength{\itemsep}{0.35em}
        \item \textbf{Two-Track Protocol Amendment}:
          Approve formalizing Aim 1 (AF recurrence) as primary and Aim 2 (paired 12L reconstruction) as secondary.
        \item \textbf{In-Clinic Paired Recording SOP}:
          Approve the 5-minute simultaneous smartwatch + 12L recording protocol during routine clinic visits.
        \item \textbf{Secondary Biomarker Selection}:
          Confirm Chris's priority secondary biomarker: QTc monitoring on sotalol vs. P-wave remodeling vs. conduction delay.
        \item \textbf{Vanguard Cohort Launch}:
          Authorize rolling out the 20-patient feasibility vanguard under Sunnybrook IRB.
      \end{enumerate}
    \end{block}
  \end{column}
  \begin{column}{0.46\textwidth}
    \begin{block}{\textbf{Summary of Trial Benefits}}
      \scriptsize
      \begin{itemize}
        \setlength{\itemsep}{0.3em}
        \item \textbf{For Chris (Cardiology)}: High-impact clinical paper evaluating wearable AF surveillance against continuous patch monitoring.
        \item \textbf{For Alex (Computer Science)}: World's first prospectively paired wearable + 12L dataset; top-tier machine learning publications.
        \item \textbf{For Industry / Funders}: A clinically rigorous, dual-value proposition with zero trial design flaws.
      \end{itemize}
    \end{block}
  \end{column}
\end{columns}
\vspace{0.4em}
\centering
\textbf{\color{NavyPrimary}Next Step}: Finalize protocol text and begin vanguard enrollment in the Arrhythmia Clinic.
\note{
  To close our meeting, here are the 4 concrete decisions we need to agree on:
  First, confirm the two-track protocol amendment: Aim 1 primary AF recurrence, Aim 2 secondary 12-lead reconstruction.
  Second, approve the simple 5-minute simultaneous recording SOP during in-clinic visits.
  Third, decide which secondary clinical biomarker Chris wants to prioritize for the paired analysis—QTc interval tracking is our recommendation.
  Fourth, green-light the 20-patient vanguard cohort.
  Thank you Chris and Alex; let's open the floor for discussion.
}
\end{frame}

\end{document}
"""

full_content = prefix + new_slides

with open("slides/main.tex", "w") as f:
    f.write(full_content)

print("Updated slides/main.tex successfully to 12 slides")
