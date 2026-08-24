SESSION="m1_hpo_parallel"

# Kill existing session if it exists
tmux kill-session -t $SESSION 2>/dev/null

# Create new session
tmux new-session -d -s $SESSION -n "worker1"

# Window 1: Worker 1
tmux send-keys -t $SESSION:worker1 "/home/mithunmanivannan/ecg_recon_env/bin/python3 scripts/m1_multiobj_hpo.py --n-trials 170 --epochs 15" C-m

# Window 2: Worker 2
tmux new-window -t $SESSION -n "worker2"
tmux send-keys -t $SESSION:worker2 "/home/mithunmanivannan/ecg_recon_env/bin/python3 scripts/m1_multiobj_hpo.py --n-trials 170 --epochs 15" C-m

# Window 3: Worker 3
tmux new-window -t $SESSION -n "worker3"
tmux send-keys -t $SESSION:worker3 "/home/mithunmanivannan/ecg_recon_env/bin/python3 scripts/m1_multiobj_hpo.py --n-trials 160 --epochs 15" C-m

echo "Started Tmux Session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
