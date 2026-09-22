#!/usr/bin/env bash
# Runs only after the active P07 GPU chain has completed.  No synthetic inputs,
# no ICM cell, no overwrite of existing Braid or GraphECG artifacts.
set -euo pipefail

project=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg
python=/home/mithunmanivannan/.venv/bin/python
artifact_root=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg
output_root="$artifact_root/braid_field/strict_graphecg_comparison_v2_cudnn_off"
terminal_log="$artifact_root/paper07_lead1_patient_context/task_native_ludb_age_sex.log"

until test -f "$terminal_log" && grep -q '\[Finished\]' "$terminal_log"; do
    sleep 30
done
while pgrep -f 'train_paper07_full12_aux_mmd.py' >/dev/null; do
    sleep 30
done

if test -e "$output_root"; then
    echo "refusing to overwrite existing strict Braid comparison: $output_root" >&2
    exit 1
fi
mkdir "$output_root"
cd "$project"

audit="$output_root/fold8_comparability.json"
"$python" scripts/braid_field/audit_strict_graphecg_comparability.py \
    --braid-validation "$artifact_root/braid_field/dataset/val.npz" \
    --canonical-fold8 outputs/ptbxl_masking_control/representation_ptbxl_full_fold8.npz \
    --setoperator-fold8 "$artifact_root/paper07_operator/development_representations/representation_val.npz" \
    --graphecg-checkpoint outputs/configuration_shift_evaluations/ptbxl/checkpoints/GraphECG.pt \
    --braid-checkpoint "$artifact_root/braid_field/training/field_full_only_best.pt" \
    --braid-checkpoint "$artifact_root/braid_field/training/field_braid_full_only_best.pt" \
    --braid-checkpoint "$artifact_root/braid_field/training/field_braid_event_full_only_best.pt" \
    --braid-checkpoint "$artifact_root/braid_field/training/field_braid_inv_full_only_best.pt" \
    --braid-checkpoint "$artifact_root/braid_field/training/field_braid_prob_full_only_best.pt" \
    --output "$audit"

for variant in field field_braid field_braid_event field_braid_inv field_braid_prob; do
    "$python" scripts/braid_field/evaluate_braid_field_shift.py \
        --checkpoint "$artifact_root/braid_field/training/${variant}_full_only_best.pt" \
        --dataset "$artifact_root/braid_field/dataset" \
        --split val --all-subsets --batch 32 --device cuda --disable-cudnn \
        --output "$output_root/${variant}_all_q8_subsets.json"
done

for variant in field field_braid field_braid_event field_braid_inv field_braid_prob; do
    "$python" scripts/task_native/run_paired_task_native_statistics.py \
        --prediction-root outputs/task_native_evaluation/aligned_predictions \
        --output-dir "$output_root/rdb_statistics_${variant}_vs_graphecg" \
        --models "graphecg,braid_${variant}" \
        --configurations Q8_indep,S6_precordial,S6_limb,S3_icu_v1,S3_icu_v5,S2_bipolar,S1_smartwatch_I,S1_lead_II
done

for variant in set_operator_robust set_operator_aux set_operator_fulllead set_operator_full12_aux_mmd set_operator_full12_aux_no_mmd; do
    "$python" scripts/task_native/run_paired_task_native_statistics.py \
        --prediction-root outputs/task_native_evaluation/aligned_predictions \
        --output-dir "$output_root/rdb_statistics_${variant}_vs_graphecg" \
        --models "graphecg,${variant}" \
        --configurations Q8_indep,S6_precordial,S6_limb,S3_icu_v1,S3_icu_v5,S2_bipolar,S1_smartwatch_I,S1_lead_II
done

printf '[Finished] strict Braid-vs-GraphECG Q8-subset and RDB paired statistics\n'
