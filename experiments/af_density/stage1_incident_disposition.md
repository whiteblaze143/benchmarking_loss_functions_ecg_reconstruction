# Incident Disposition: Mixed-Partition Array Reads and Held-Out Test Data Isolation

**Date**: 2026-11-22  
**Status**: REMEDIATED & PHYSICALLY ISOLATED  
**Classification**: P0.1 Source-Integrity Remediation  
**Auditor**: Data Provenance & Integrity Auditor (`integrity`)  
**Program**: T-CAIREM Continuous SetOperator AF-Density Research Program  

---

## 1. Executive Summary

During the Priority 0 audit of the exploratory Stage 1 training queue (`train_stage1_gpu_queue.py`), a methodological breach of the strict test data isolation rule was identified. Specifically, in `prepare_validation_dataset`:

```python
with np.load(W / "windows.npz") as z:
    windows = {k: z[k] for k in z.files}
```

This comprehension loaded the entire monolithic `windows.npz` archive (802,491 30-second windows across 289 files) into process memory, including target labels for all three partitions: train (504,777), validation (166,047), and protected held-out test (131,667 windows). Similarly, `reference_intervals.csv` (413 intervals across 152 patients) was parsed globally into memory, containing 71 held-out test reference intervals.

While downstream logic explicitly subsetted indices using `file_manifest.csv` and `patient_splits.csv` (so that no test metrics, test losses, or test predictions were ever calculated or printed), loading protected test labels into the development heap violates the core research protocol's zero-access boundary. In-memory subsetting after a global read fails the standard of independent custody.

---

## 2. Root Cause Analysis

1. **Monolithic Data Archives**: The initial waveform preprocessing stage (`waveform_protocol.py`) bundled all 802,491 extracted windows into a single `windows.npz` and all 413 reference intervals into `reference_intervals.csv` across train, validation, and test cohorts.
2. **Post-Load Subsetting**: Development scripts relied on dictionary comprehension and boolean indexing (`lookup[ids]`) after loading the entire mixed archive into heap memory.
3. **Absence of Filesystem-Level Separation**: There were no dedicated partition-specific filesystem directories physically isolating the held-out test data from the development workspace.

---

## 3. Corrective Actions Executed (P0.1 Remediation)

### 3.1 Physically Separated Partition Artifacts
The monolithic archives have been decommissioned for development use and partitioned into strictly isolated files in `data_partitions/`:
- **`data_partitions/train_windows.npz`** (`375e839c...`):
  - Exactly 504,777 windows across 181 files and 90 patients.
  - Zero validation or test windows present.
- **`data_partitions/val_windows.npz`** (`7dfb370f...`):
  - Exactly 166,047 windows across 60 files and 30 patients.
  - Zero train or test windows present.
- **`data_partitions/train_reference_intervals.csv`** (`af763298...`):
  - Exactly 247 training AF episode reference intervals across 181 training files.
- **`data_partitions/val_reference_intervals.csv`** (`f0a0f3f4...`):
  - Exactly 95 validation AF episode reference intervals across 60 validation files.
- **Held-Out Test Exclusion**:
  - The 131,667 test windows and 71 test reference intervals have been physically excluded from all development data paths.

### 3.2 Operating-System Level Isolation
- Development and Stage 1B training queues (`run_stage1b_corrective_queue.py` and subsequent stages) are strictly bound to read exclusively from `data_partitions/train_windows.npz`, `data_partitions/val_windows.npz`, and the corresponding reference interval CSVs.
- Monolithic mixed-partition archives are quarantined from the training process.

### 3.3 Patient Population Reconciliation
- Generated **`stage1_corrected_training_population.csv`** (`e76f6dad...`) documenting the exact training population across all 90 patients, 99 records, 181 daily files, and 504,777 windows (76,281 AF windows, 428,496 non-AF windows, mean AF burden 15.11%).
- Generated **`stage1_input_provenance_reconciliation.json`** (`5ba0564b...`) documenting input shapes, sampling rates, and channel identities for authentic raw ECG streaming (6,000 samples @ 200 Hz).

---

## 4. Verification and Audit Witness

An automated integrity audit executed on `data_partitions/` confirmed:
1. `set(train_windows.file_index) & set(test_files) == set()` (0 test files in train).
2. `set(val_windows.file_index) & set(test_files) == set()` (0 test files in val).
3. `set(train_reference_intervals.file_index) & set(test_files) == set()` (0 test files in train intervals).
4. `set(val_reference_intervals.file_index) & set(test_files) == set()` (0 test files in val intervals).
5. No held-out test data is present in or readable by development loaders.

---

## 5. Disposition Verdict

**Verdict**: **HOLD / DEVELOPMENT-ONLY CLEARED FOR STAGE 1B REPLICATION**.  
- The mixed-partition array read vulnerability is fully remediated and sealed.
- Stage 1B corrective replication may proceed using authentic HDF5 raw waveforms and physically separated `data_partitions/`.
- No evaluation on held-out test data shall be permitted until Stages 2–7 are fully executed, checkpoints locked, and an external validation protocol frozen.
