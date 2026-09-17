# Final Proposal: Paper 08 — Phase-Token Attention & Dynamic Routing vs Equivalence Clustering

## 1. Problem Anchor & Prior Art Demarcation

### 1.1 The Prior Art Landscape: Beat-Synchronous Transformers & ECG Tokenizers
Recent literature (2024–2026) has widely adopted Transformer architectures for ECG:
- **Beat-Synchronous & Patch Transformers**: HeartLang (CinC 2024), Beat-Synchronous Transformer (arXiv 2025), and PulseAttention (2025) demonstrate that segmenting ECGs into beat-aligned tokens outperforms fixed arbitrary time-series patching.
- **Discrete ECG Tokenization**: HeartLLM (AAAI 2025/2026) and ECG-Byte (2025/2026) discretize continuous ECG waveforms into discrete token vocabularies for language modeling.
- **Novelty Boundary**:
  Paper 08 does not claim the generic novelty of applying a Transformer or tokenization to ECGs.
  Instead, Paper 08 focuses on a precise physiological question in the phase domain:
  $$\boxed{ \text{Dynamic Cross-Phase Routing vs Local Conduction & Equivalence Clustering} }$$
  Over the sequence of 16 cyclic phase-cell kernel mean embeddings:
  $$\mathcal{Z} = [z_1, z_2, \dots, z_{16}]^\top \in \mathbb{R}^{16 \times D}$$
  where each $z_\phi = \mu_{P(\cdot \mid \phi)}$ is an RKHS distributional representation of cardiac phase $\phi \in \mathbb{Z}_{16}$.

The central scientific hypotheses are:
1. **Dynamic vs Static Routing**: Does the network genuinely utilize patient-specific dynamic attention weights $A_{ij}(x) = \text{softmax}(Q_i K_j^\top / \sqrt{d})$, or does an input-invariant static routing matrix $\bar{A}_{ij}$ capture the necessary inter-phase interactions?
2. **Local Conduction vs Distant Cross-Phase Routing**: Can pathology be detected with strictly local cyclic attention (window $\pm 2$ on $C_{16}$), or is long-range routing (e.g. P-wave phase $\phi \in [0, 3]$ attending directly to ST-T segment $\phi \in [10, 14]$) mandatory for diagnosis?
3. **Phase-Order Sensitivity**: Does the model rely on the cyclic temporal sequence of cardiac phases via positional embeddings, or does it collapse into a permutation-invariant set aggregator?
4. **Continuous RKHS Geometry vs Statistically Verified Discrete Equivalence**: Does continuous RKHS feature geometry preserve critical diagnostic variance that discrete vector quantization (`kmeans_dictionary` or `equivalence.py` complete-linkage clusters) discards?

---

## 2. Mathematical Formalization

### 2.1 Phase-Token Transformer Architecture
Let $Z = [z_1, \dots, z_{16}]^\top \in \mathbb{R}^{16 \times D_{\text{in}}}$ be the input phase-cell KMEs ($D_{\text{in}} = 256$).
1. **Token Embedding & Cyclic Positional Encoding**:
   $$H_0 = [e_{\text{cls}}; Z W_{\text{in}}] + E_{\text{pos}} \in \mathbb{R}^{17 \times d_{\text{model}}}$$
   where $e_{\text{cls}} \in \mathbb{R}^{1 \times d_{\text{model}}}$ is a learned classification token, and $E_{\text{pos}} \in \mathbb{R}^{17 \times d_{\text{model}}}$ provides phase-index awareness.
2. **Multi-Head Self-Attention with Configurable Routing Mask**:
   For each head $h \in \{1, \dots, H\}$:
   $$A^{(h)} = \text{softmax}\left( \frac{Q^{(h)} (K^{(h)})^\top}{\sqrt{d_k}} + M \right)$$
   where $M \in \mathbb{R}^{17 \times 17}$ is a spatial attention mask:
   - **Global Dynamic (`full`)**: $M_{ij} = 0$ for all $i, j \in \{0, \dots, 16\}$.
   - **Local Banded (`local_banded`)**: $M_{0j} = M_{i0} = 0$ (CLS interacts globally), and for phase tokens $i, j \in \{1, \dots, 16\}$:
     $$M_{ij} = \begin{cases} 0 & \text{if } \min(|i - j|, 16 - |i - j|) \le w \\ -\infty & \text{otherwise} \end{cases}$$
     with default bandwidth $w = 2$.
   - **Static Routing (`static_attention`)**: $A^{(h)} = \text{softmax}(\bar{W}^{(h)})$, where $\bar{W}^{(h)} \in \mathbb{R}^{17 \times 17}$ is a learned parameter matrix independent of input $x$.
   - **Uniform Attention (`uniform_attention`)**: $A^{(h)}_{ij} = 1/17$, removing all selective routing while preserving feedforward transformations.
   - **Phase-Agnostic Set (`phase_agnostic_set`)**: $E_{\text{pos}} = 0$, enforcing permutation invariance across phase tokens.

### 2.2 Statistical Equivalence Clustering (`equivalence.py`)
To rigorously evaluate whether continuous phase representations can be reduced to a discrete alphabet of cardiac states without loss of clinical information:
1. **Pairwise Dissimilarity**:
   $$D_{ij} = \mathbb{E}_{x \sim P}[\|z_i(x) - z_j(x)\|_2]$$
2. **Simultaneous Upper Confidence Bounds**:
   Using max-t bootstrap over patient records:
   $$\hat{D}_{ij}^{\text{upper}} = \hat{D}_{ij} + q_{1-\alpha}\left( \max_{u, v} (\hat{D}_{uv} - D_{uv}^{*b}) \right)$$
   controlling family-wise error rate (FWER) at level $\alpha = 0.05$.
3. **Complete-Linkage Equivalence Merging**:
   Merge clusters $C_a, C_b$ if $\max_{i \in C_a, j \in C_b} \hat{D}_{ij}^{\text{upper}} < \delta$.
4. **Split-Half Reliability**:
   Measure odd-versus-even heartbeat token distribution agreement via normalized Jensen-Shannon divergence:
   $$R_{\text{odd/even}} = 1 - \frac{\text{JSD}(p_{\text{odd}}, p_{\text{even}})^2}{\ln 2} \in [0, 1]$$

---

## 3. Benchmark Hierarchy & Strong Comparators

| Variant | Attention Routing | Positional Encoding | Representation | Scientific Role |
| :--- | :--- | :--- | :--- | :--- |
| `global_dynamic` (`full`) | Full $17 \times 17$ dynamic | Learned | Continuous KME | Primary Model |
| `local_banded` | Banded cyclic ($w=2$) | Learned | Continuous KME | Locality vs global routing test |
| `static_attention` | Learned input-invariant $\bar{A}$ | Learned | Continuous KME | Dynamic routing falsification |
| `uniform_attention` | Fixed $A_{ij} = 1/17$ | Learned | Continuous KME | Feature transformation control |
| `scrambled_phases` | Full dynamic | Scrambled at eval | Continuous KME | Phase-order sensitivity test |
| `phase_agnostic_set` | Full dynamic | None ($E_{\text{pos}} = 0$) | Continuous KME | Permutation invariance control |
| `kmeans_tokens` | Full dynamic | Learned | Quantized 64-codebook | Discrete dictionary baseline |
| `linear_probe` | Linear pooling | None | Continuous KME | Non-transformer linear baseline |
| `cnn_matched_control` | 1D Cyclic CNN ($C_{16}$) | Implicit in conv | Continuous KME | Inductive bias matched control (Paper 02) |
| `random_feature_control` | Full dynamic | Learned | Frozen random proj | Representation sanity baseline |

---

## 4. Key Hypotheses & Verifiable Gates

1. **$H_{\text{dynamic}}$ (Dynamic Routing Necessity)**:
   $$\text{AUROC}(\text{global\_dynamic}) - \text{AUROC}(\text{static\_attention}) \ge +0.010$$
   If this difference is $< 0.005$, the claim that cardiac diagnosis requires patient-specific dynamic attention is falsified.
2. **$H_{\text{locality}}$ (Local Conduction Sufficiency)**:
   If $\text{AUROC}(\text{local\_banded}) \approx \text{AUROC}(\text{global\_dynamic})$, then distant cross-phase attention is largely redundant, and local conduction transitions dominate.
3. **$H_{\text{order}}$ (Phase Order Exploitation)**:
   $$\text{AUROC}(\text{global\_dynamic}) - \text{AUROC}(\text{scrambled\_phases}) \ge +0.020$$
   Scrambling phase positions must measurably degrade performance, proving the network uses temporal phase order.
4. **$H_{\text{continuous}}$ (Continuous RKHS Value)**:
   $$\text{AUROC}(\text{global\_dynamic}) - \text{AUROC}(\text{kmeans\_tokens}) \ge +0.010$$
   Proves continuous distributional representations retain clinical variance lost during discrete vector quantization.
