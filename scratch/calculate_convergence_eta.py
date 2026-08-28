import json, time
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Read actual elapsed time per epoch from log
log_path = Path("refine-logs/convergence_10e/logs/001_conv10e_conv_control_s42_l1.log")
lines = []
if log_path.is_file():
    with open(log_path) as f:
        lines = f.readlines()

# Scan completed epochs
runs_dir = Path("refine-logs/convergence_10e/runs")
total_completed_epochs = 0
for r in runs_dir.iterdir():
    m = r / "metrics.jsonl"
    if m.is_file():
        with open(m) as f:
            epochs = [json.loads(line) for line in f if line.strip()]
            total_completed_epochs += len(epochs)

total_jobs = 24
target_epochs_per_job = 10
total_planned_epochs = total_jobs * target_epochs_per_job
remaining_epochs = total_planned_epochs - total_completed_epochs

# Epoch duration: 7 min 16s train + ~1.5 min val = ~8.75 min (525 seconds)
sec_per_epoch = 525.0
sec_per_job = sec_per_epoch * 10 # 5250 s = 87.5 min = 1.458 hrs

total_remaining_seconds = remaining_epochs * sec_per_epoch
total_remaining_hours = total_remaining_seconds / 3600.0

now_utc = datetime.now(timezone.utc)
eta_utc = now_utc + timedelta(seconds=total_remaining_seconds)

# EDT is UTC - 4
edt_offset = timedelta(hours=-4)
now_edt = now_utc + edt_offset
eta_edt = eta_utc + edt_offset

print("="*75)
print("  10-EPOCH CONVERGENCE PANEL EXECUTION ETA BREAKDOWN")
print("="*75)
print(f"Total Jobs in Panel:            {total_jobs} (12 models x 2 leads)")
print(f"Total Planned Epochs:           {total_planned_epochs} epochs")
print(f"Completed Epochs:               {total_completed_epochs} epochs")
print(f"Remaining Epochs:               {remaining_epochs} epochs")
print(f"Empirical Time per Epoch:       ~{sec_per_epoch/60:.1f} minutes (training + full validation)")
print(f"Empirical Time per 10E Job:     ~{sec_per_job/60:.1f} minutes ({sec_per_job/3600:.2f} hours)")
print(f"Total Remaining Compute Time:   {total_remaining_hours:.1f} hours ({total_remaining_hours/24:.2f} days)")
print(f"Current Local Time (EDT):       {now_edt.strftime('%A, %b %d, %Y at %I:%M %p EDT')}")
print(f"Projected Completion (EDT):     {eta_edt.strftime('%A, %b %d, %Y at %I:%M %p EDT')}")
print("="*75)
