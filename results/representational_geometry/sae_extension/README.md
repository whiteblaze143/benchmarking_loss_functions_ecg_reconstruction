# Post-Hoc Sparse Autoencoder (SAE) Extension Protocol (Section 21)

## Status: Protocol Specification & Hook Configuration (Secondary Phase)

As specified in **PRD Section 21**:
> *"Because Li et al. actually analyze SAE feature dictionaries, add a secondary, clearly separated phase. Do not begin here. If direct latent analysis identifies interesting hooks, train frozen post-hoc SAEs on those activations... Do not conflate direct Transformer-channel analysis with SAE feature analysis."*

### Candidate Hooks for Post-Hoc SAE Dictionaries
Based on the Scale I–III audit conclusions, the following 6 candidate representations provide optimal divergence for SAE dictionary analysis:
1. `conv15e_A0_raw_s42_l0` at Hook **H1** (Raw millivolt temporal backbone representations before conditioning).
2. `conv15e_R7_morlet_mag_ueg_real_s42_l0` at Hook **H3** (Wavelet + temporal fused representation).
3. `D2_current_id_l1_s42_l0` at Hook **H4** (Categorical learned embedding conditioned latent).
4. `D3_theta_mul_l1_s42_l0` at Hook **H4** (True physical spherical angle continuous conditioned latent).
5. `D5_permuted_theta_mul_l1_s42_l0` at Hook **H4** (Circularly permuted coordinates).
6. `D8_random12_mul_l1_s42_l0` at Hook **H4** (Gaussian random keys).

### Standardized SAE Training Architecture (Frozen Protocol)
- **Architecture**: Single-layer Top-K SAE (Gao et al., 2024 / Anthropic 2024) with encoder $f(x) = \text{TopK}(W_e(x - b_d) + b_e, k=32)$ and decoder $\hat{x} = W_d f(x) + b_d$.
- **Expansion Ratio**: $8\times$ (latent dim $D = 768 \implies K = 6,144$ dictionary features).
- **Target Sparsity**: $L_0 = 32$ active features per token (99.5% sparsity).
- **Training Set**: $N=17,418$ Fold 1–8 training records, extracted activations ($200$ tokens $\times 12$ leads).
- **Evaluation**: Zero-shot feature crystal detection, parallelogram residuals, and precordial monosemanticity on Fold 9.
