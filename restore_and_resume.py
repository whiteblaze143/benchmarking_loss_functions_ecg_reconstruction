import os
import time
import subprocess
import shutil

def wait_for_downloads():
    print("Waiting for wget processes to finish...")
    while True:
        try:
            output = subprocess.check_output(["pgrep", "-f", "wget"], text=True)
            if output.strip():
                print(f"wget is still running... (PIDs: {output.strip()})")
                time.sleep(60)
            else:
                break
        except subprocess.CalledProcessError:
            print("No wget processes found. Downloads should be complete.")
            break

def extract_ptbxl():
    ptbxl_zip = "data/ptbxl.zip"
    if os.path.exists(ptbxl_zip):
        print(f"Extracting {ptbxl_zip}...")
        subprocess.run(["unzip", "-q", "-o", ptbxl_zip, "-d", "data/"])
        # Find the extracted folder
        extracted_dirs = [d for d in os.listdir("data/") if d.startswith("ptb-xl-a-large") and os.path.isdir(os.path.join("data", d))]
        if extracted_dirs:
            src_dir = os.path.join("data", extracted_dirs[0])
            dest_dir = "data/ptb_xl"
            print(f"Moving contents of {src_dir} to {dest_dir}...")
            os.makedirs(dest_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                shutil.move(os.path.join(src_dir, item), os.path.join(dest_dir, item))
            shutil.rmtree(src_dir)
            print("PTB-XL extracted successfully.")
        else:
            print("Warning: Could not find extracted PTB-XL folder.")
    else:
        print("Warning: ptbxl.zip not found!")

def run_pipeline():
    print("Running fix_queue.py...")
    subprocess.run(["python3", "fix_queue.py"])

    print("Starting queue_manager.py --resume...")
    subprocess.run(["python3", "scripts/queue_manager.py", "--resume"])

    print("Running quarto render...")
    subprocess.run(["quarto", "render", "book/"])

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    wait_for_downloads()
    extract_ptbxl()
    run_pipeline()
