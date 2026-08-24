#!/bin/bash
#SBATCH --job-name=hpo_n125
#SBATCH --output=logs/hpo_n125_%j.out
#SBATCH --error=logs/hpo_n125_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu

# ============================================================================
# N=125 HPO with NSGA-II (pop=25, gen=5)
# Expected runtime: ~6-8 hours with 8× GPU parallelism via Optuna distributed
# ============================================================================

source /home/mithunmanivannan/ecg_recon_env/bin/activate
cd /home/mithunmanivannan

echo "=============================================="
echo "Starting N=125 HPO Study"
echo "Config: pop_size=25, n_generations=5"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=============================================="

# Run with N=125 trials
python scripts/m1_multiobj_hpo.py \
    --n-trials 125 \
    --epochs 15 \
    --subset 0.5

echo "=============================================="
echo "HPO Completed"
echo "End time: $(date)"
echo "=============================================="
