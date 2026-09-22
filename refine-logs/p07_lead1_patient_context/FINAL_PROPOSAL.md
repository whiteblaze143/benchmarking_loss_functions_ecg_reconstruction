# Proposal: Lead-I Reconstruction With Deployment-Available Patient Context

Train matched SetOperator variants on PTB-XL folds 1–7 to reconstruct the real 12-lead waveform from Lead I alone: one variant receives age and sex and one does not. Both retain diagnosis BCE plus IMQ MMD squared and select on fold-8 diagnosis macro-AUROC only. Context is encoded from train-standardized age plus recorded sex; records without either field are excluded, never imputed.

The term `patient-specific fine-tuning` is reserved for a later, self-supervised adaptation experiment with multiple ECGs per patient. It may use only the observed Lead I and deployment-available context, never withheld leads or task labels. Sunnybrook is not eligible for that claim because all 20 retained MRNs occur once.

For external task-native cells, context is admitted only when it is available at deployment and not identical to, or derived from, the target. ISP sex prediction uses age only; diagnosis codes and target labels are never inputs. No synthetic waveforms, metadata, or missing-value fallback is permitted.
