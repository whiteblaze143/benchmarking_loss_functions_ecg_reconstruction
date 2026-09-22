# Foundation-Model Comparator Plan

**Problem:** Determine whether real externally pretrained ECG encoders alter the fixed-12-lead or physical-Lead-I EchoNext representation result relative to GraphECG, without manufacturing absent leads.

**Method thesis:** Admit a foundation model only when its checkpoint, preprocessing, and input-lead contract are evidenced; measure its frozen representation using the identical EchoNext `echonext_shd_12` five-fold probe protocol used for GraphECG.

## Claim map

| Claim | Minimum convincing evidence | Anti-claim |
|---|---|---|
| C1: A foundation encoder can be compared fairly to GraphECG on EchoNext only under its native input contract. | Same 5,442 records, 12 outcomes, KFold seed 42, linear probe, and immutable embedding artifact. | A score arose from proxy padding, a different cohort, or a released outcome head. |
| C2: Full-12 and Lead-I are distinct settings. | Explicit full-12 ceiling and separately labelled true one-channel Lead-I result. | Full-12 performance establishes robustness to absent leads. |

## Must-run queue stages

| Run ID | System | Scope | Required output | Gate |
|---|---|---|---|---|
| FM-00 | ECGFounder archive admission | Strict checksum, extraction, strict state load | `ecgfounder_admission/admission.json` | Both releases load into author architectures. |
| FM-01 | ECGFounder 12-lead | Full-12 EchoNext frozen representation | `ecgfounder_12lead_echonext/result.json` | Full-12 comparator only. |
| FM-02 | ECGFounder 1-lead | Physical Lead I only, EchoNext frozen representation | `ecgfounder_leadI_echonext/result.json` | No padding or multi-lead interpretation. |
| FM-03 | ECG-FM | Read-only archived payload provenance audit | `ecgfm_payload_audit.json` | Not an evaluation. |
| FM-04 | HuBERT-ECG | Official checkpoint acquisition and immutable revision capture | `hubert_ecg_large/admission.json` | Evaluation awaits a later extractor gate. |

## Explicit exclusions

- ECG-FM wrapper zero-padding for absent leads;
- HuBERT bridge zero-padding for absent leads;
- CSFM random-tensor stub and CLEF without a usable artifact;
- task-specific EchoNext Minimodel as a frozen-representation baseline;
- ICM, fabricated channels, duplicated leads, or unsupported multi-lead subsets.

## Execution dependency and safety contract

The queue waits for the active strict Braid-vs-GraphECG completion marker, then
for related P07/Braid/statistics GPU processes to exit and `nvidia-smi` to show
no compute processes. Every output goes to a new timestamped artifact root.

ECGFounder preprocessing is declared: invert known EchoNext dataset z-scoring,
resample the released 250-Hz signal to the author filter's 500-Hz contract,
apply the author filter, then apply author per-record z-scoring. This is kept
separate from GraphECG preprocessing and written to the result JSON.

## Interpretation rules

- ECGFounder-12L compares only to GraphECG full-12 EchoNext (`0.7050` macro AUROC); it is not a subset result.
- ECGFounder-1L compares only to a physical Lead-I comparator under the same protocol.
- HuBERT and ECG-FM are not evaluated merely by passing their acquisition/audit stage.
- No inferential superiority claim follows from five-fold probe scores alone.

## Failure interpretation

A hash/state-load failure, unsupported input shape, data-provenance mismatch, or
missing official HuBERT revision terminates that model branch. It never invokes
a substitute encoder, synthetic waveform, or proxy input.
