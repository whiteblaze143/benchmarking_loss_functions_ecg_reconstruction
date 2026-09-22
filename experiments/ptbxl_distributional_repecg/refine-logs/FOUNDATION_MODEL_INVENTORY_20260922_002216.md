# Foundation-Model Weight and Loader Inventory

**Recorded:** 2026-09-22 00:22 EDT  
**Scope:** Inventory only. No comparator extraction, fine-tuning, or evaluation
was launched. A model is not eligible for a reduced-lead comparison merely
because a local wrapper can pad missing channels with zeros.

## Reference EchoNext protocol to match

The existing GraphECG result is the `echonext_shd_12` protocol in
`scripts/evaluation/evaluate_all_datasets_zeroshot.py`:

- 5,442 fixed EchoNext test records;
- 12 SHD endpoints;
- representations extracted from the waveform and evaluated with the same
  shuffled five-fold (`random_state=42`) linear-probe CV;
- GraphECG artifact: 192-dimensional representation, macro AUROC `0.7050`
  (`outputs/external_dataset_evaluations/zeroshot_echonext_graphecg.json`).

This is a representation-transfer benchmark, not a direct use of a released
EchoNext outcome head. Any comparator must write a separate immutable output
and preserve its encoder checkpoint hash, preprocessing, embedding definition,
and probe splits.

## Inventory

| Candidate | Checkpoint / access state | Loader code | Native lead contract | Eligibility now |
|---|---|---|---|---|
| ECGFounder, 12-lead | **Verified recoverable.** `project_other_models.tgz` checksum passed; member `12_lead_ECGFounder.pth`, 369,942,585 bytes. The direct checkout path is currently absent. | Official `finetune_model.py:ft_12lead_ECGFounder`; removes the released `dense` head and supports linear probing. | Exactly 12 input channels. | **Admit only to full-12 EchoNext representation evaluation** after extraction to an immutable artifact and preprocessing parity audit. Not eligible for reduced subsets. |
| ECGFounder, 1-lead | **Verified recoverable.** Same checksum-passed archive; member `1_lead_ECGFounder.pth`, 369,807,481 bytes. | Official `finetune_model.py:ft_1lead_ECGFounder`, a true 1-channel architecture. | Exactly one input channel; no channel padding required. | **Candidate for single-lead cells only** after proving from the release/data contract that a chosen physical lead may be supplied. It cannot stand in for two-, three-, six-, or arbitrary subset inputs. |
| ECG-FM | No local pretrained checkpoint at the wrapper's expected path `checkpoints/mimic_iv_ecg_physionet_pretrained.pt`. Archive contains only `mimic_iv_ecg_finetuned.pt`, 1,081,641,499 bytes, with payload/provenance not yet inspected. | `src/ecg_fm_classifier.py` uses Fairseq Signals `load_model_and_task`. | Current wrapper appends all-zero channels whenever fewer than 12 are supplied. | **Do not admit** to either full-12 or reduced-lead benchmark until the archived payload is inspected and its pretraining provenance is established. Reduced-lead path is expressly excluded as a zero-pad proxy. |
| HuBERT-ECG | No local checkpoint or cache found. Official checkout exists and its README documents Hugging Face access through `AutoModel.from_pretrained`; the legacy identifier redirects to the maintained author namespace. | `unified_latents/engineering/src/models/hubert_bridge.py`; native repository also includes training/finetuning code. | Native dataset code loads and flattens exactly 12 leads. Our bridge zero-fills sparse inputs. | **Do not admit** to reduced subsets. A full-12 cell is possible only after a separately pinned official checkpoint download, license/environment audit, and unmodified encoder extraction test. No download has been performed. |
| CSFM | No source checkout, official loader, checkpoint artifact, or archive entry found. The sole local `CSFM_Prior` is an explicit smoke-test stub whose `encode` method returns `torch.randn` tensors. Its claimed downstream checkpoint `checkpoints/fm_v2_experiments/FM4_CSFM_Full/best_model.pt` is absent. | `unified_latents/engineering/models/fm_v2.py`; not an adapter to a real CSFM checkpoint. | Stub declares single-lead support but does not encode ECG data. | **Unavailable and permanently excluded from this benchmark until a real official model, checkpoint, and loader are independently established.** |
| CLEF | No local source directory, loader, checkpoint, cache, or archive entry found. Literature mentions are not usable artifacts. | None. | Unknown. | **Unavailable.** Do not queue or describe as evaluated. |
| EchoNext Minimodel | **Verified recoverable** in the checksum-passed archive: `weights.pt`, 4,272,845 bytes. Companion tabular transformer and waveform-normalization parameter files are present in the source checkout. | Author-provided inference entry point under `echonext_minimodel_repo/7-EchoNext Minimodel`. | Full 12-lead waveform plus seven tabular features. | **Not a foundation-model representation baseline.** May be an explicitly separate, outcome-specific native EchoNext supervised reference only after checking cohort/split and tabular-feature availability. It must never be placed in the frozen-encoder representation table. |

## Hard exclusions for the requested reduced-subset table

The following local implementation behavior is not a valid missing-lead
experiment and will not be used:

- `ECGFMClassifier.extract_embeddings` pads inputs below 12 channels with
  zeros;
- `HuBERTBridge.forward` creates a 12-lead zero tensor and inserts the active
  leads;
- the native HuBERT-ECG data loader itself reshapes exactly 12 leads.

Therefore, the only presently plausible non-proxy reduced-lead foundation
cell is ECGFounder 1-lead, and only for a *single* physical lead after its
official input contract is checked. There is currently no qualified
foundation model for multi-lead subsets.

## Required admission gates before any launch

1. Restore each selected archive member to a new immutable artifact directory;
   record archive SHA-256, member path, extracted SHA-256, and loader commit.
2. Inspect model payloads without training to establish architecture,
   pretraining versus fine-tuning status, input sampling rate, sequence length,
   and license.
3. Implement only author-contract preprocessing; prohibit zero filling,
   lead duplication, interpolation from absent leads, or synthetic channels.
4. Run a CPU/GPU load-and-shape audit with a real EchoNext batch. Failure is an
   exclusion, not a fallback implementation.
5. Freeze the exact EchoNext probe split indices and evaluate only admissible
   cells. Reduced-lead cells must use an architecture/checkpoint whose native
   input contract matches that subset.

## Decision

No foundation-model job is queued. The initial admissible target set is:

- ECGFounder 12-lead for the same full-12 EchoNext representation protocol;
- possibly ECGFounder 1-lead for one-lead-only EchoNext cells, subject to gate
  2--4;
- HuBERT-ECG full-12 only after a pinned official download and audit.

ECG-FM, CLEF, and all multi-lead reduced-subset foundation comparisons remain
blocked on concrete artifact/interface evidence.
