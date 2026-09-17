# Review Summary: Paper 07 — Continuous Lead-Span Functional ECG

## Adversarial Peer Review Dialogue

### Reviewer Critique 1: "GraphECG and LAEF already solved flexible-lead ECG. Why do we need Paper 07?"
**Response**:
GraphECG and LAEF operate on fixed 3D electrode coordinates and discrete lead graphs. When given a novel, synthetic linear functional $q \in S^7$ (e.g. an interior interpolation between precordial and limb leads, or an oblique linear combination), discrete/graph models must either snap to nearest discrete nodes or fail. Paper 07 formulates the problem as sampling a patient-specific continuous function defined on the dual of the 8-dimensional ECG lead span, with exact $\mathbb{Z}_2$ projective gauge symmetry ($\mathbb{RP}^7 = S^7 / \{q \sim -q\}$).

### Reviewer Critique 2: "Is $S^7$ a physical lead-orientation sphere on the body?"
**Response**:
No. We have explicitly corrected this. $q \in S^7$ is a **lead-span synthetic functional**, not an arbitrary physical electrode orientation. A physical lead corresponds to some $q$ if and only if $\ell_{\text{new}} \in \operatorname{span}\{\ell_1, \dots, \ell_8\}$. Large precordial displacements (> 2 cm) or posterior leads (V7–V9) do not lie strictly in this span and are categorized as $\mathcal{Q}_{\text{physical OOD}}$.

### Reviewer Critique 3: "Phase-KME breaks odd symmetry: $r_{-q} \neq -r_q$."
**Response**:
Acknowledged and resolved. While raw atoms $a_q = [x_q, \dot{x}_q]$ are odd under sign reversal ($a_{-q} = -a_q$), nonlinear kernel feature maps $\psi(a)$ are not odd. We resolved this structurally by constructing the projective measurement atom:
$$g_q(t) = [q x_q(t), \; q \dot{x}_q(t)] = [q q^\top x(t), \; q q^\top \dot{x}(t)] \in \mathbb{R}^{16}$$
Under polarity reversal: $(-q) x_{-q} = (-q)(-x_q) = q x_q$, making $g_{-q} \equiv g_q$ identically invariant by construction.

### Reviewer Critique 4: "Can $m \le 6$ measurements invert a generic 8-D ECG?"
**Response**:
No. For $m < 8$, $\operatorname{rank}(Q) \le m < 8$, so the nullspace is non-trivial. Exact reconstruction is only mathematically possible under a low-rank source model $x(t) = L s(t)$ ($r = 3$) when the observability condition $\operatorname{rank}(Q L) = 3$ is satisfied. We reformulated the reconstruction evaluation around this exact observability gate.

---

## Verdict
**PRODUCTION-LOCKED**. Prior art collision resolved, mathematical terminology grounded in lead-span functionals, exact $\mathbb{Z}_2$ gauge symmetry implemented, and non-strawman benchmark hierarchy established.
