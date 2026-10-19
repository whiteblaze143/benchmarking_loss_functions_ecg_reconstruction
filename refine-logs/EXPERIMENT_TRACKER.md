# Experiment Tracker: Paper 07 — 168 GPU-Hour SetOperator Architectural Ablation Campaign

**Total Budget**: 168.0 GPU-hours  
**GPU-Hours Consumed**: ~2.5 h (Baseline training underway)  
**GPU-Hours Remaining**: ~165.5 h  
**Status**: ACTIVE  

## Live Leaderboard & Execution State

| ID | Experiment Name | Group | Active Clf Params | Status | Best Epoch | Val AUROC | Val AUPRC | Dev Grassmann AUROC | GPU-h | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **A0** | Original SetOperator | A | 1,044,161 | **RUNNING** | 2 | 0.7462 | 0.6891 | *(eval pending)* | ~2.5 | Epoch 8/50 in `train_so_span_s42` |
| **A1** | G1 Projective Geometry | A | 1,046,465 | **QUEUED** | — | — | — | — | — | Auto-launches via `g1_launch_controller` |
| **A2** | Capacity-Matched Raw-$q$ | A | 1,046,497 | PLANNED | — | — | — | — | — | Mandatory control for G1 (+32 params) |
| **D2** | Tabular-Only Baseline | D | ~2,113 | PLANNED | — | — | — | — | — | Clinical covariates baseline |
| **D1** | ECG-Only SetOperator | D | 1,027,585 | PLANNED | — | — | — | — | — | Pure ECG morphology baseline |
| **B2** | Phase-Position Readout | B | 1,044,177 | PLANNED | — | — | — | — | — | Softmax phase weighting |
| **A3** | Remove Operator Identity | A | 1,039,425 | PLANNED | — | — | — | — | — | Constant/ablated operator token |
| **B1** | Global-KME (No Phase) | B | 1,044,161 | PLANNED | — | — | — | — | — | Phase-agnostic baseline |
| **C1** | No Cross-View Transformer| C | 647,425 | PLANNED | — | — | — | — | — | Per-token MLP + set pooling |
| **C2** | Mean Set Pooling | C | 970,049 | PLANNED | — | — | — | — | — | Masked mean pooling |
| **A4** | Projective-Only (Unsigned)| A | 1,045,953 | PLANNED | — | — | — | — | — | Unsigned $qq^\top \in \mathbb{R}^{36}$ |
| **A5** | Diagonal Quadratic | A | 1,044,673 | PLANNED | — | — | — | — | — | $[q, q_1^2..q_8^2] \in \mathbb{R}^{16}$ |
| **A6** | G2 Relational Attention | A | 1,044,169 | PLANNED | — | — | — | — | — | Attention bias $\beta (q_i^\top q_j)^2$ |
| **A7** | Diagonal Attention Control | A | 1,044,169 | PLANNED | — | — | — | — | — | Matched control for G2 |
| **B3** | Noncircular PhaseCNN | B | 1,044,161 | PLANNED | — | — | — | — | — | Zero-padded Conv1d |
| **B4** | No Local Convolutions | B | 1,044,161 | PLANNED | — | — | — | — | — | Per-phase MLP |
| **B5** | Disordered Phase Cells | B | 1,044,161 | PLANNED | — | — | — | — | — | Negative control |
| **C3** | 1-Layer Transformer | C | 846,721 | PLANNED | — | — | — | — | — | Depth = 1 |
| **C4** | Single-Head Attention | C | 1,044,161 | PLANNED | — | — | — | — | — | $n_{\text{head}} = 1$ |
| **D3** | Late Fusion Adapter | D | 1,045,953 | PLANNED | — | — | — | — | — | Late concatenation |
| **B6** | Raw-Block Representation | B | 1,044,161 | PLANNED | — | — | — | — | — | Reshaped raw waveform control |
| **E1** | G1 + Phase-Position Readout| E | 1,046,481 | PLANNED | — | — | — | — | — | Combined operator + phase |
| **E2** | Best Geom + Best Set | E | TBD | PLANNED | — | — | — | — | — | Factorial combination |
