# Third-Party Source Code Provenance Ledger

This directory contains vendored reference implementations as mandated by the GRAIL-ECG PRD (§3).
These directories are **immutable references** and must not be edited in place. Any usage must be mediated by adapters.

| Vendored Directory | Original Source Path | Upstream Archive | Preregistered SHA-256 |
|---|---|---|---|
| `third_party/nefnet_v2` | `external/NEFNET-v2-main/` | `NEFNET-v2-main.zip` | `062eb0c40b1383850e3ae049a08faa0842d965639fe3e49be3af03540150c4ae` |
| `third_party/krepes` | `external/KREPES/Kernel_SSL-krepes/` | `Kernel_SSL-krepes.zip` | `c3692c7df48f10c664319a2101f71c7b47826c737de1be772dbe6b35678ca890` |
| `third_party/krepes_mgpu_entk` | `external/KREPES/mGPU_eNTK/` | `mGPU_eNTK.zip` | `c9f951040dff9ea7fcdcf818dbe78c442993cbe9b05c9caf173e71d5994d479e` |
| `third_party/krepes_nn` | `external/KREPES/nn/` | `nn.zip` | `fd12c69a3161177ac4b304de0d9787d760791bade824b72af0e3a97d3676ca99` |

### Known Upstream Implementation Constraints Audited:
1. **NEF-NET+**:
   - `encoder.py` defines `Encoder` twice (second overwriting the first).
   - Hardcoded temporal length `1152` and LayerNorm sizes `2304` / `4608`.
   - Global min-max normalization in original loaders (erases voltage scale).
   - `AngleDefine.py` contains path-dependent globals.
   - *Resolution:* Clean reimplementation of `ThetaEncoder` and ResNet-1D architecture without min-max or hardcoded 1152 lengths.
2. **KREPES**:
   - `utils/interpretability.py` uses heuristic `similarity * ||A_l||` instead of formal $\Delta A = -H_{GN}^{-1} \nabla_A L(A_0)$.
   - Duplicate `SimclrGNHPreconditioner` in `model/precondition.py`.
   - *Resolution:* Implement formal $\Delta A$ sensitivity displacement and exact trace eNTK.
3. **KREPES NN**:
   - VICReg training loop erroneously passes total dataset size $N$ instead of batch size $B$ for covariance denominator.
   - *Resolution:* Implement clean, mathematically verified VICReg loss in `src/losses/ssl.py`.
