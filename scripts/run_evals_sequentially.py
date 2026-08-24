import subprocess, sys, time, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

eval_scripts = [
    ("LUDB Evaluation", "scripts/evaluate_ludb_delineation.py", "refine-logs/ludb_eval.log"),
    ("ISP Evaluation", "scripts/evaluate_isp_delineation.py", "refine-logs/isp_eval.log"),
    ("MMD Evaluation", "scripts/evaluate_temporal_mmd.py", "refine-logs/mmd_eval.log"),
]

def run_script(name, script_path, log_path):
    logging.info(f"Starting {name}...")
    cmd = f"CUDA_VISIBLE_DEVICES='' ~/.venv/bin/python3 {script_path} > {log_path} 2>&1"
    res = subprocess.run(cmd, shell=True)
    if res.returncode == 0:
        logging.info(f"Finished {name} successfully.")
    else:
        logging.error(f"{name} failed with returncode {res.returncode}.")

def main():
    while True:
        for name, script_path, log_path in eval_scripts:
            run_script(name, script_path, log_path)
            time.sleep(5)
        logging.info("Completed one full pass over all evals. Sleeping 60s before next check...")
        time.sleep(60)

if __name__ == "__main__":
    main()
