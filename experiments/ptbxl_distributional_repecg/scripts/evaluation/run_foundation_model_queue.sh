#!/usr/bin/env bash
# Dependency-gated foundation-model queue. It uses no missing-lead proxy.
set -euo pipefail

project=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg
python=/home/mithunmanivannan/.venv/bin/python
archive=/data/mithunmanivannan/codex_artifacts/checkpoint_archives/home_models_20260916/project_other_models.tgz
archive_sha256=/data/mithunmanivannan/codex_artifacts/checkpoint_archives/home_models_20260916/project_other_models.tgz.sha256
ecgfounder_repo=/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/ecg_fm_integration/ecgfounder_repo
echonext=/home/mithunmanivannan/data/echonext
artifact_root=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison
strict_log=/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/braid_field/logs/strict_graphecg_comparison_queue.log
ecgfm=/data/mithunmanivannan/codex_artifacts/checkpoint_archives/home_models_20260916/fitbit_models.tgz
ecgfm_sha256=/data/mithunmanivannan/codex_artifacts/checkpoint_archives/home_models_20260916/fitbit_models.tgz.sha256

until test -f "$strict_log" && grep -q '\[Finished\] strict Braid-vs-GraphECG' "$strict_log"; do
    if ! tmux has-session -t strict_braid_graphecg_queue 2>/dev/null && ! grep -q '\[Finished\] strict Braid-vs-GraphECG' "$strict_log" 2>/dev/null; then
        echo 'upstream strict Braid queue stopped without its completion marker' >&2
        exit 1
    fi
    sleep 60
done

while pgrep -f 'train_paper07_full12_aux_mmd.py|evaluate_braid_field_shift.py|run_paired_task_native_statistics.py' >/dev/null; do
    sleep 60
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
    sleep 60
done

run_id=$(date +%Y%m%d_%H%M%S)
run_root="$artifact_root/$run_id"
mkdir -p "$run_root"
cd "$project"

"$python" scripts/evaluation/restore_ecgfounder_checkpoints.py \
    --archive "$archive" --archive-sha256 "$archive_sha256" --repo "$ecgfounder_repo" \
    --output "$run_root/ecgfounder_admission"

"$python" scripts/evaluation/evaluate_ecgfounder_echonext.py \
    --checkpoint "$run_root/ecgfounder_admission/ecgfounder_12_lead.pth" --repo "$ecgfounder_repo" \
    --echonext-dir "$echonext" --leads 12 --output "$run_root/ecgfounder_12lead_echonext"

"$python" scripts/evaluation/evaluate_ecgfounder_echonext.py \
    --checkpoint "$run_root/ecgfounder_admission/ecgfounder_1_lead.pth" --repo "$ecgfounder_repo" \
    --echonext-dir "$echonext" --leads 1 --lead-index 0 --output "$run_root/ecgfounder_leadI_echonext"

# ECG-FM remains audit-only: archive extraction/provenance is intentionally not a model evaluation.
sha256sum -c "$ecgfm_sha256"
mkdir "$run_root/ecgfm_archive_member"
tar -xzf "$ecgfm" -C "$run_root/ecgfm_archive_member" \
    home/mithunmanivannan/fitbit-clinical-dashboard/ECG-FM/ckpts/mimic_iv_ecg_finetuned.pt
"$python" scripts/evaluation/audit_ecgfm_checkpoint.py \
    --checkpoint "$run_root/ecgfm_archive_member/home/mithunmanivannan/fitbit-clinical-dashboard/ECG-FM/ckpts/mimic_iv_ecg_finetuned.pt" \
    --output "$run_root/ecgfm_payload_audit.json"

# HuBERT is acquisition-only until its author preprocessing and embedding gate passes.
"$python" scripts/evaluation/acquire_hubert_ecg.py --output "$run_root/hubert_ecg_large"
printf '[Finished] foundation-model admission, ECGFounder EchoNext cells, ECG-FM audit, and HuBERT acquisition\n'
