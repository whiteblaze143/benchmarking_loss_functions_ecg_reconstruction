#!/bin/bash
set -e

SESSION_NAME="vcg_baseline_M1"

# Create a new detached tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the tmux session for seed 42
tmux send-keys -t $SESSION_NAME "/home/mithunmanivannan/.venv/bin/python3 scripts/train_m1_pearson.py --seed 42 --factorial_mask 11111 --loss_protocol paper_parity --epochs 10 --batch_size 256 --run_name msvae_vcg_seed42 --checkpoint_path checkpoints/msvae_vcg_seed42.pt > refine-logs/vcg_seed42.log 2>&1" C-m

# Send commands to the tmux session for seed 200
tmux send-keys -t $SESSION_NAME "/home/mithunmanivannan/.venv/bin/python3 scripts/train_m1_pearson.py --seed 200 --factorial_mask 11111 --loss_protocol paper_parity --epochs 10 --batch_size 256 --run_name msvae_vcg_seed200 --checkpoint_path checkpoints/msvae_vcg_seed200.pt > refine-logs/vcg_seed200.log 2>&1" C-m

# Send commands to the tmux session for seed 201
tmux send-keys -t $SESSION_NAME "/home/mithunmanivannan/.venv/bin/python3 scripts/train_m1_pearson.py --seed 201 --factorial_mask 11111 --loss_protocol paper_parity --epochs 10 --batch_size 256 --run_name msvae_vcg_seed201 --checkpoint_path checkpoints/msvae_vcg_seed201.pt > refine-logs/vcg_seed201.log 2>&1" C-m

echo "Launched M1 training in tmux session: $SESSION_NAME"
