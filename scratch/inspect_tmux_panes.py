import subprocess

sessions = [
    "wavelet_chained_supervisor",
    "queue_3arch_cpu",
    "onelead_rdb_eval",
    "onelead_checkpoint_archiver",
    "checkpoint_archiver_3arch",
    "smoke_test",
    "book_review_render_round1"
]

for s in sessions:
    print(f"\n========================================================")
    print(f" TMUX SESSION: {s}")
    print(f"========================================================")
    try:
        out = subprocess.check_output(["tmux", "capture-pane", "-p", "-t", s], stderr=subprocess.STDOUT, text=True)
        lines = [line for line in out.strip().split("\n") if line.strip()]
        for l in lines[-10:]:
            print(l)
    except Exception as e:
        print("Error capturing pane:", e)
