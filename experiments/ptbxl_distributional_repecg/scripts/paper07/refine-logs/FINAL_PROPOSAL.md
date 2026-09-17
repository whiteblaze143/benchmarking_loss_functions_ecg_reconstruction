# Final Proposal: Paper 07 — Continuous Lead-Span Functional ECG

## 1. Problem Anchor & Prior Art Demarcation

### 1.1 The Prior Art Landscape: GraphECG & LAEF
A comprehensive audit against the literature and open-source repositories reveals that Paper 07 cannot claim novelty from generic flexible-lead modeling:
- **GraphECG** (Sri et al., 2024–2025): Public repository and technical documentation explicitly model ECG leads as directed edges on an electrode graph, supporting variable-sized lead subgraphs, arbitrary custom electrode-pair measurements, bidirectional reversed leads carrying negated signals, spherical-harmonic geometric coordinate encoding, and coordinate-conditioned auxiliary lead reconstruction.
- **LAEF** (August 2026 preprint): Implements lead-agnostic graph processing natively on variable-size lead subsets pre-trained with stochastic lead sampling, demonstrating strong 1- and 2-lead robustness.

### 1.2 The Defensible Scientific Object: Continuous Lead-Span Functional ECG
We reject the ungrounded claim that $q \in S^7$ represents an "arbitrary physical lead orientation on the human torso." Instead, the mathematical object of Paper 07 is:
$$\boxed{ \text{Continuous lead-span measurement functionals over an 8-dimensional ECG basis} }$$
Let $x(t) = [I, II, V_1, \dots, V_6]^\top \in \mathbb{R}^8$ be the standard 8-dimensional algebraically independent ECG physical voltage basis. We define a continuous linear functional:
$$\ell_q(x) = q^\top x(t), \quad q \in \mathbb{R}^8$$
The model observes finite, unordered samples of the patient-specific operator-response function:
$$\boxed{ F_x: q \mapsto r_q = \mu_{P([q^\top x, q^\top \dot{x}] \mid \theta)} }$$
$$\mathcal{D}_C = \big\{(q_1, F_x(q_1)), \dots, (q_m, F_x(q_m))\big\}, \quad m \in [1, 6]$$
The core scientific question is:
$$\boxed{ \text{Does explicit continuous measurement-functional information improve inference beyond signal-only set aggregation and strong discrete/geometric baselines?} }$$

---

## 2. Mathematical Formalization & Physical Distinctions

### 2.1 $S^7$ is Coefficient Geometry, Not Thoracic Geometry
A new physical body-surface electrode configuration is represented exactly by some $q \in S^7$ *if and only if*:
$$\ell_{\text{new}} \in \operatorname{span}\{\ell_1, \dots, \ell_8\}$$
Precordial displacements (> 1.5–2 cm), posterior leads (V7–V9), right-sided leads (V3R/V4R), or Mason-Likar placements do not lie strictly in the span of the standard 8-lead basis. We therefore partition evaluated operators into:
1. $\mathcal{Q}_{\text{clinical}}$: canonical leads (I, II, V1–V6) and algebraically exact derived limb leads (III = II - I, aVR, aVL, aVF).
2. $\mathcal{Q}_{\text{synthetic}}$: held-out lead-span functionals (interior interpolations $(1-\alpha)I + \alpha V_2$, dense random unit vectors).
3. $\mathcal{Q}_{\text{physical OOD}}$: true non-standard body-surface configurations (reserved for external BSPM validation).

Coefficient distance is defined on projective space $\mathbb{RP}^7$:
$$d_{\mathbb{RP}}(q, p) = \arccos |q^\top p|$$
and covariance-induced Mahalanobis distance:
$$d_\Sigma^2(q_1, q_2) = (q_1 - q_2)^\top \Sigma_x (q_1 - q_2)$$

### 2.2 Exact $\mathbb{Z}_2$ Gauge Symmetry on Projective Space $\mathbb{RP}^7$
For any measurement pair $(q, x_q)$, reversing the electrode polarity yields:
$$(q, x_q) \mapsto (-q, -x_q)$$
These represent two coordinate descriptions of the same unoriented measurement axis. Diagnostic classification must satisfy:
$$f\big(\dots, (q, x_q), \dots\big) = f\big(\dots, (-q, -x_q), \dots\big)$$
The unoriented physical measurement space is $\mathbb{RP}^7 = S^7 / \{q \sim -q\}$.

**Resolution of the Non-Odd Phase-KME Bug**:
While raw atoms $a_q(t) = [x_q(t), \dot{x}_q(t)]$ satisfy $a_{-q}(t) = -a_q(t)$, a nonlinear Nyström/KME feature map $\psi(a)$ is generally **not** odd ($\psi(-a) \neq -\psi(a)$ for RBF/IMQ kernels). To eliminate approximate KL regularization and enforce exact $\mathbb{Z}_2$ gauge invariance by construction, we define the projective atom:
$$g_q(t) = [q \, x_q(t), \; q \, \dot{x}_q(t)] = [q q^\top x(t), \; q q^\top \dot{x}(t)] \in \mathbb{R}^{16}$$
Under polarity reversal:
$$(-q) x_{-q} = (-q)(-x_q) = q x_q \implies g_{-q}(t) \equiv g_q(t)$$
This guarantees $r(q, x_q) = r(-q, -x_q)$ to machine precision ($< 10^{-14}$) by architectural design.

### 2.3 Observability-Governed Source Reconstructive Inversion
For $m \in [1, 6]$ measurements $Q \in \mathbb{R}^{m \times 8}$, $\operatorname{rank}(Q) \le m < 8$, so the nullspace is non-trivial. Under a low-rank cardiac source model $x(t) = L s(t)$ with $s(t) \in \mathbb{R}^r$ ($r = 3$), the observed signal is $y(t) = Q L s(t)$. Exact recovery of any query operator $x_{q^*}(t) = q^{*\top} x(t)$ is mathematically solvable *if and only if*:
$$\boxed{ \operatorname{rank}(Q L) = r = 3 }$$
- Observable context ($\operatorname{rank}(Q L) = 3$): $s(t) = (Q L)^\dagger y(t)$, yielding exact recovery error $< 10^{-12}$.
- Unobservable context ($\operatorname{rank}(Q L) < 3$): at least one source dimension is invisible to the lead set, and exact reconstruction must provably fail.

---

## 3. Benchmark Hierarchy & Strong Comparators

We replace the binary `continuous_primary` vs strawman `categorical_primary` framing with the 10-variant benchmark hierarchy:
1. `projective_continuous`: continuous $q$, exact structural $\mathbb{Z}_2$ symmetry ($g_q$), primary model.
2. `projective_continuous_aux`: continuous $q$, exact polarity symmetry + auxiliary query-response prediction.
3. `continuous_mlp`: raw continuous $q$ MLP, unconstrained symmetry ablation.
4. `q_ablated_set`: set model with NO operator geometry (signal-only control).
5. `categorical_ids`: standard discrete token baseline.
6. `nearest_known_operator`: realistic discrete baseline mapping unseen $q$ to $\arg\max_k |q^\top e_k|$.
7. `linear_q_encoder`: linear projection of $q$ (MLP complexity ablation).
8. `GraphECG_geometry`: 3D electrode/edge geometry comparator.
9. `analytic_pinv`: exact linear algebra pseudo-inverse reconstruction baseline.
10. `LMMSE_operator`: train-covariance-aware linear reconstruction baseline.
