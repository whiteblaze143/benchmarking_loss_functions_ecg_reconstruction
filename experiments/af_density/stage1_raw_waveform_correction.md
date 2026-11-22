# Stage 1 Raw Waveform Ingestion & Correction Specification

**T-CAIREM Continuous SetOperator AF-Density Research Program**  
**Author:** Representation & Failure Diagnostician (`@diagnostician`)  
**Date:** 2026-11-22  
**Status:** Priority 0 (P0.2 & P0.3) Corrective Action Verified

---

## 1. Executive Summary & Incident Disposition

During the Stage 1 review, critical audit of `experiments/af_density/run_continuous_setoperator_queue.py` revealed that raw-waveform experiments (`H071` and `I082`) did not receive authentic recorded electrograms. Instead, the runner script generated a pseudo-signal:

```python
raw_mock = np.zeros((len(resp), 2, 6000), dtype=np.float32)
for c in range(2):
    raw_mock[:, c, :] = np.repeat(resp[:, c, :, 0], 375, axis=1)
```

This synthetic construct tiled the first coordinate of the 16-cell Phase-KME response representation across 375 samples ($16 \times 375 = 6,000$ samples), producing an artificial 16-step staircase waveform. As a result:
1. `H071` (Per-Channel Raw ResNet SetOperator) was trained and evaluated on step-function artifacts lacking true QRS complexes, f-waves, P-wave absences, and baseline wander, resulting in a degenerate AUROC of $0.4995$.
2. The poor performance of `H071` cannot be used to reject raw-waveform representations or ResNet fusion.
3. In accordance with directive **P0.2**, all synthetic/mock fallbacks are permanently deprecated.
4. In accordance with directive **P0.3**, `I082` is formally disambiguated into:
   - **`I082-v1`**: SetOperator Residual MLP Head (capacity control, Phase-KME only).
   - **`I082-v2`**: Genuine Dual-Branch Raw ResNet + SetOperator Residual Fusion.

---

## 2. Authentic Raw ECG Data Specification

### 2.1 File System Storage & Schema
Authentic raw ambulatory ECG recordings are stored in the IRIDIA-AF dataset repository:
- **Root Directory**: `/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5/`
- **Subdirectory Pattern**: `ecg_hdf5/record_{record_id}/record_{record_id}_ecg_{daily_file_index:02d}.h5`
- **Manifest Mapping**: Each `file_index` in `file_manifest.csv` explicitly provides the absolute `h5_path`.

### 2.2 Internal HDF5 Architecture
Each HDF5 file contains a top-level dataset `'ecg'`:
- **Dimensions**: `(N_samples, 2)` where $N_{\text{samples}} \approx 1.58 \times 10^7$ to $1.73 \times 10^7$ samples ($\sim$22 to 24 hours).
- **Sampling Rate**: Exactly $200\text{ Hz}$ ($5.0\text{ ms}$ interval).
- **Data Type**: `float64` (calibrated voltage in millivolts/microvolts).
- **Channel Allocation**:
  - Column 0: Nominal Clinical Lead I.
  - Column 1: Nominal Clinical Lead II.

### 2.3 Window Slicing Semantics
Windows are defined by the verified partition archives (`train_windows.npz` and `val_windows.npz`):
- **Window Length**: Exactly $6,000\text{ samples}$ ($30.00\text{ seconds}$).
- **Interval**: Sliced as `ecg[assigned_start : assigned_end, :]`.
- **Consistency**: Across all 702,491 dataset windows, $\text{assigned\_end} - \text{assigned\_start} \equiv 6,000$.
- **Transposition**: To match PyTorch 1D Convolution conventions `(Batch, Channels, Time)`, the slice is cast to `float32` and transposed:
  $$\mathbf{x}_{\text{raw}} \in \mathbb{R}^{B \times 2 \times 6000}$$

---

## 3. Empirical Verification of Authentic Waveforms

The diagnostic verification script sampled authentic windows from both training and validation cohorts:

### 3.1 Training Cohort: Patient 000 (`file_index = 0`)
- **File**: `/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5/record_000/record_000_ecg_00.h5`
- **Total Samples**: $17,278,200$ samples ($24.00\text{ hours}$).
- **Window $[6000 : 12000]$ (Samples $30.0\text{s}$ to $60.0\text{s}$)**:
  - Lead I: Range $[-1.2300, +4.4800]\text{ mV}$, Mean $+0.0664\text{ mV}$, Std $0.6974\text{ mV}$.
  - Lead II: Range $[-2.4000, +1.5600]\text{ mV}$, Mean $+0.1390\text{ mV}$, Std $0.3910\text{ mV}$.
  - Dynamic range and signal morphology exhibit physiological QRS spike amplitude ($\sim 4.5\text{ mV}$) and baseline stability.

### 3.2 Validation Cohort: Patient 019 (`file_index = 33`)
- **File**: `/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1/ecg_hdf5/record_003/record_003_ecg_00.h5`
- **Total Samples**: $15,897,790$ samples ($22.08\text{ hours}$).
- **Window $[6000 : 12000]$**:
  - Lead I: Range $[-12.4900, +11.8600]\text{ mV}$, Mean $+0.6224\text{ mV}$, Std $2.3749\text{ mV}$.
  - Lead II: Range $[-20.6900, +5.0300]\text{ mV}$, Mean $-3.6569\text{ mV}$, Std $2.8089\text{ mV}$.
  - High dynamic range and substantial baseline wander characteristic of ambulatory Holter recordings.

---

## 4. High-Performance HDF5 Streaming Protocol

Because slicing 4,096 windows individually from disk could incur random I/O latency, the Stage 1B queue implements an optimized caching strategy:

1. **File Handle Caching**: Open HDF5 file handles are cached in a thread-safe dict `_H5_CACHE` during dataset preparation, eliminating repeated filesystem opens.
2. **File-Grouped Extraction**: Windows are sorted and grouped by `file_index`. Slices from the same recording are extracted sequentially to benefit from OS kernel page cache.
3. **RAM Footprint**:
   $$\text{Total Memory} = 4,096 \times 2 \times 6,000 \times 4\text{ bytes} \approx 196.6\text{ MB}$$
   This memory requirement easily fits into both host system RAM ($>64\text{ GB}$ available) and GPU VRAM ($40\text{ GB}$ A100).
4. **Validation Segment Pre-Loading**: For the 15 informative validation segments, raw waveforms are loaded into GPU tensors during validation initialization, ensuring rapid validation forward passes without disk bottlenecks.

---

## 5. Architectural Integration for Stage 1B

| Model Identifier | Scientific Identity | Input Modality | Input Dimensions |
| :--- | :--- | :--- | :--- |
| **`A001`** | Continuous SetOperator Baseline | Phase-KME | $(B, 2, 16, 64)$ |
| **`A002`** | Categorical SetOperator Baseline | Phase-KME | $(B, 2, 16, 64)$ |
| **`A003`** | Operator-Ablated SetOperator | Phase-KME | $(B, 2, 16, 64)$ |
| **`A004`** | Continuous 2D Polar SetOperator | Phase-KME | $(B, 2, 16, 64)$ |
| **`D032-v1`** | Dual-Moment Phase SetOperator | Phase-KME (Mean + Var) | $(B, 2, 16, 64)$ |
| **`F053`** | No-Transformer Set Operator | Phase-KME | $(B, 2, 16, 64)$ |
| **`G061`** | Multimodal RR + Phase SetOperator | Phase-KME + RR | $(B, 2, 16, 64) \oplus (B, 36)$ |
| **`H071-v2`** | Genuine Raw ResNet SetOperator | **Authentic Raw ECG** | $(B, 2, 6000)$ |
| **`I082-v1`** | Residual MLP Head Capacity Control | Phase-KME | $(B, 2, 16, 64)$ |
| **`I082-v2`** | Genuine Raw ResNet + SetOperator Fusion | **Authentic Raw ECG** + Phase | $(B, 2, 6000) \oplus (B, 2, 16, 64)$ |

---

## 6. Zero Test Data Leakage Guarantee

In strict compliance with directive **P0.1**:
- Training windows are drawn exclusively from `data_partitions/train_windows.npz` (files $0 \dots 180$, $N=504,777$).
- Validation segments are drawn exclusively from `data_partitions/val_windows.npz` (files $181 \dots 240$, $N=166,047$).
- Zero held-out test records ($N=131,667$) are accessed, opened, or loaded into memory (`test_access: false`).
