# Experiment Plan: Paper 07 — Continuous Lead-Span Functional ECG

## 1. Variant Matrix & Comparators

| Variant | Operator Knowledge | Reconstructive Task | Scientific Role |
| :--- | :---: | :---: | :--- |
| `projective_continuous` | Continuous $q$, exact $\mathbb{Z}_2$ symmetry ($g_q$) | No | **Primary model** |
| `projective_continuous_aux` | Continuous $q$, exact $\mathbb{Z}_2$ symmetry | Yes | Auxiliary predictive model |
| `continuous_mlp` | Raw $q$ MLP (unconstrained symmetry) | No | Symmetry ablation |
| `q_ablated_set` | No operator geometry (signal-only) | No | Does $q$ matter beyond set pooling? |
| `categorical_ids` | Discrete Lead IDs | No | Standard discrete token baseline |
| `nearest_known_operator` | Unseen $q \to \arg\max_k |q^\top e_k|$ | No | Strong realistic discrete baseline |
| `linear_q_encoder` | Fixed/linear $q$ mapping | No | MLP complexity control |
| `GraphECG_geometry` | 3D electrode/edge geometry | Optional | Direct prior-art comparator |
| `analytic_pinv` | $x_{\text{pinv}} = Q^\top (Q Q^\top)^\dagger y$ | Yes | Analytic reconstruction oracle |
| `LMMSE_operator` | $\Sigma_x Q^\top (Q \Sigma_x Q^\top + \sigma^2 I)^{-1} y$ | Yes | Statistical linear reconstruction baseline |

---

## 2. Core Hypotheses

- **$H_1$ (Continuous Operator Relevance)**:
  $$\text{AUROC}(\text{projective\_continuous}) > \text{AUROC}(\text{q\_ablated\_set})$$
- **$H_2$ (Continuous Generalization vs Strong Discrete Baselines)**:
  $$\text{AUROC}_{\text{unseen}}(\text{projective\_continuous}) > \text{AUROC}_{\text{unseen}}(\text{nearest\_known\_operator}) > \text{AUROC}_{\text{unseen}}(\text{categorical\_ids})$$
- **$H_3$ (Structural Gauge Invariance)**:
  $$\|\hat{y}(\mathcal{D}) - \hat{y}(T_{\pm}\mathcal{D})\|_\infty < 10^{-12}$$
- **$H_{\text{aux-repr}}$ (Auxiliary Representation Benefit)**:
  Auxiliary query prediction improves held-out operator prediction and sparse-context ($m \le 3$) calibration without degrading full-context diagnosis.

---

## 3. Observability Stratification Protocol
Evaluate performance not just by context size $m$, but stratified by:
1. Measurement rank: $\operatorname{rank}_\epsilon(Q)$
2. Measurement coherence: $\mu(Q) = \max_{i \neq j} |q_i^\top q_j|$
3. Minimum singular value of source projection: $\sigma_{\min}(Q L)$
4. Projective distance to training bank: $d_{\pm}(q, \mathcal{Q}_{\text{train}}) = \min_{p \in \mathcal{Q}_{\text{train}}} \arccos |q^\top p|$ across bands $\delta \in \{5^\circ, 15^\circ, 30^\circ, 45^\circ\}$.
