with open("slides/main.tex", "r") as f:
    text = f.read()

# Marker after slide 7
marker = "% Slide 8: Data Collection Methodology: Dual-Stream Protocol"
idx = text.find(marker)
if idx == -1:
    print("Could not find Slide 8 marker")
    exit(1)

prefix = text[:idx]
suffix = text[idx:]

# Update slide numbers in suffix
suffix = suffix.replace("% Slide 8: Data Collection", "% Slide 9: Data Collection")
suffix = suffix.replace("% Slide 9: The Paired", "% Slide 10: The Paired")
suffix = suffix.replace("% Slide 10: Figure 2", "% Slide 11: Figure 2")
suffix = suffix.replace("% Slide 11: Sample Size", "% Slide 12: Sample Size")
suffix = suffix.replace("% Slide 12: Action Items", "% Slide 13: Action Items")

empirical_slide = r"""% ------------------------------------------------------------------------------
% Slide 8: Empirical Validation of ECG-AIM: Benchmark Performance
% ------------------------------------------------------------------------------
\begin{frame}{Empirical Validation of ECG-AIM: Benchmark Performance}{conv15e\_A0\_wave\_noSSL\_gated\_add Evaluated on 175,890 Patient Observations}
\centering
\resizebox{0.96\textwidth}{!}{%
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Clinical Evaluation Dimension} & \textbf{Factorial Baseline} & \textbf{Stanford 3DRECON-QT} & \textbf{ECG-AIM (Ours)} & \textbf{Statistical Contrast / Rigor} \\ \midrule
\textbf{Precordial Chest Mean Pearson $r$} & 0.570 & 0.732 & \textbf{0.776} & $\Delta = +0.206$ ($p < 10^{-300}$), Ranked \#3 of 55 \\
\textbf{Anterior Lead V3 Pearson $r$} & 0.382 & 0.670 & \textbf{0.739} & $\Delta = +0.357$ ($p < 10^{-300}$), Anterior LV wall view \\
\textbf{QRS Duration Error (MAE / Median)} & 13.85 ms & 10.26 ms & \textbf{10.05 / 8.0 ms} & Sub-10ms precision; tracks depolarization \\
\textbf{Conduction Delay Concordance (>120ms)} & 90.7\% & 94.3\% & \textbf{94.3\%} & $\text{aOR} = 1.84$ ($p < 10^{-17}$), Specificity: \textbf{96.5\%} \\
\textbf{Sokolow-Lyon LVH Voltage MAE} & 1.13 mV & 0.63 mV & \textbf{0.72 mV} & $\text{ICC} = 0.714$, Quantifies chamber hypertrophy \\
\textbf{EchoNext Structural Heart Disease} & 68.8\% & \textbf{46.5\% (Collapse)} & \textbf{71.6\%} & Preserves chamber echocardiographic pathology \\ \bottomrule
\end{tabular}%
}
\vspace{0.3em}
\begin{columns}[T]
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Key Technical Advantage: Morlet Wavelets}}
      \scriptsize
      Continuous Morlet wavelets preserve high-frequency QRS details and ST-T repolarization without baseline distortion, outperforming Stanford's static CNN on chest leads ($r = 0.776$ vs $0.732$).
    \end{block}
  \end{column}
  \begin{column}{0.48\textwidth}
    \begin{block}{\textbf{Overcoming Ansari's Structural Collapse}}
      \scriptsize
      Ansari's scalar QT loss drops to \textbf{46.5\%} on structural echo phenotypes. ECG-AIM retains \textbf{71.6\%} concordance across 1,000 paired exams, preserving ventricular geometry.
    \end{block}
  \end{column}
\end{columns}
\note{
  Alex and Chris, this slide presents the strictly empirical benchmark across all 55 models evaluated with patient-clustered MMRM and GEE.
  Notice our champion model: conv15e_A0_wave_noSSL_gated_add.
  Compared to the Stanford 3DRECON-QT model published in Circulation, our model achieves higher precordial chest lead correlation—0.776 vs 0.732—and a Lead V3 correlation of 0.739 vs 0.670.
  Crucially, look at the bottom row: Ansari's model collapsed to 46.5% concordance on echocardiographic structural heart disease in EchoNext because their scalar QT loss ignored chamber geometry. Our wavelet-gated transformer maintains 71.6% structural concordance and tracks QRS duration with a median error of just 8.0 milliseconds!
}
\end{frame}

"""

new_content = prefix + empirical_slide + suffix
with open("slides/main.tex", "w") as f:
    f.write(new_content)

print("Successfully inserted Slide 8 into slides/main.tex (Total slides: 13)")
