# Refinement Report: Paper 08

## 1. Mathematical Specifications

### 1.1 Cyclic Banded Mask
The 16 phase cells reside on the discrete torus $C_{16}$. For phase tokens $i, j \in \{1, \dots, 16\}$:
$$\text{dist}_{C_{16}}(i, j) = \min(|i - j|, 16 - |i - j|)$$
For bandwidth $w \in \mathbb{N}$:
$$M_{ij} = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \text{ or } \text{dist}_{C_{16}}(i, j) \le w \\ -\infty & \text{otherwise} \end{cases}$$
This ensures the CLS token (index 0) pools from all phase tokens, while phase-to-phase routing is strictly localized to neighboring cardiac phases.

### 1.2 Static Attention Routing Matrix
Rather than computing $A(x) = \text{softmax}(Q(x) K(x)^\top / \sqrt{d})$, the static attention variant uses:
$$A^{(h)} = \text{softmax}(\bar{W}^{(h)})$$
where $\bar{W}^{(h)} \in \mathbb{R}^{17 \times 17}$ is a learned parameter tensor.
This preserves identical parameter capacity and non-linear feature transformation depth while destroying input-dependent dynamic routing.

### 1.3 Permutation Invariance Under $E_{\text{pos}} = 0$
When positional embeddings are zeroed and mean pooling (or symmetric attention) is applied, for any permutation $\pi \in S_{16}$:
$$f(\pi \mathcal{Z}) = f(\mathcal{Z})$$
verified to machine precision ($< 10^{-6}$).

---

## 2. Verification Outcomes
- Mask sparsity and correct cyclic boundary wrap-around verified.
- Static attention weights verified invariant across diverse inputs.
- Dynamic attention verified to have non-zero variance across distinct ECG morphologies.
- Complete-linkage clustering verified with simultaneous bootstrap bounds.
