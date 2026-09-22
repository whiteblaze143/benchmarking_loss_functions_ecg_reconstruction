# Experiment Plan

1. Materialize exact PTB-XL age/sex context aligned to the existing folds 1–7/8 records; fit age scaling on folds 1–7 and exclude incomplete rows.
2. Train `lead1_unconditioned` and `lead1_age_sex_conditioned` with identical encoder/decoder capacity, seed, Lead-I input, target, and BCE + IMQ MMD² loss. The only difference is the context vector.
3. Select by fold-8 diagnostic macro-AUROC. Report held-out-lead reconstruction separately from labels.
4. Audit target-cohort metadata before task-native evaluation. Use only admissible cells: LUDB age+sex, Zhejiang sex-only, ISP age-only. Do not use Sunnybrook demographics or sex on ISP.
5. Run any record-level adaptation only after a separate pre-registered self-supervision protocol; call it transductive record adaptation, not patient-level personalization, unless repeat ECGs become available.
