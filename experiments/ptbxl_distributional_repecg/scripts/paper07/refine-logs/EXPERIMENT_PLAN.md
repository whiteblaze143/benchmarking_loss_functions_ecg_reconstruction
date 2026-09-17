# Experiment Plan: Paper 07 — Continuous Measurement-Operator ECG

## 1. Variant Matrix

| Variant | Operator Representation | Auxiliary Task | Hypothesis Role |
| :--- | :---: | :---: | :--- |
| `continuous_primary` | Continuous $q \in \mathbb{S}^7$ | No | Core continuous operator baseline |
| `categorical_primary` | Discrete Lead IDs | No | Discrete token control |
| `continuous_auxiliary` | Continuous $q \in \mathbb{S}^7$ | Recon + Orient | **Primary full model** |
| `categorical_auxiliary` | Discrete Lead IDs | Recon only | Categorical reconstructive control |

---

## 2. Core Hypotheses & Evaluation Protocol

### Hypothesis 1 ($H_{\text{cont}}$: Continuous Geometric Generalization)
$$\text{AUROC}_{\text{unseen}}(\text{continuous\_primary}) > \text{AUROC}_{\text{unseen}}(\text{categorical\_primary})$$
On unseen operator projections (derived limb leads, interpolations, dense projections), continuous geometric embeddings maintain high diagnostic accuracy, whereas categorical embeddings collapse due to `UNKNOWN_ID` mapping.

### Hypothesis 2 ($H_{\text{aux}}$: Reconstructive Regularization)
$$\text{AUROC}(\text{continuous\_auxiliary}) > \text{AUROC}(\text{continuous\_primary})$$
Reconstructing unseen target lead responses forces the patient representation $z$ to retain full spatiotemporal cardiac field geometry.

### Hypothesis 3 ($H_{\text{orient}}$: Polarity Invariance)
$$\mathcal{L}_{\text{orientation}}(\text{continuous\_auxiliary}) < 10^{-4}$$
Flipping operator polarity $q \to -q$ leaves diagnostic predictions invariant.

---

## 3. Hyperparameter Grid
- **Learning Rates**: $1 \times 10^{-4}, 3 \times 10^{-4}, 1 \times 10^{-3}$
- **Weight Decays**: $1 \times 10^{-5}, 1 \times 10^{-4}, 1 \times 10^{-3}$
- **Batch Size**: 64
- **Max Epochs**: 100 with Early Stopping (patience = 10)
- **Loss Weights**: $\lambda_{\text{recon}} = 0.1, \lambda_{\text{orient}} = 0.05$
