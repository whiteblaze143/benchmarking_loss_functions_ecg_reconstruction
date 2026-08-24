import sys
from pathlib import Path
import os
import re

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()
def find_best_lambda(base_dir):
    best_lambda = None
    best_r2 = -float('inf')
    
    if not os.path.exists(base_dir):
        return None
    
    for folder in os.listdir(base_dir):
        if folder.startswith("grid_lambda_"):
            summary_path = os.path.join(base_dir, folder, "metrics_summary.txt")
            if os.path.exists(summary_path):
                with open(summary_path, "r") as f:
                    content = f.read()
                    l_match = re.search(r"lambda_spec: ([\d.]+)", content)
                    r_match = re.search(r"best_val_r2: ([-.\d]+)", content)
                    
                    if l_match and r_match:
                        l = float(l_match.group(1))
                        r = float(r_match.group(1))
                        if r > best_r2:
                            best_r2 = r
                            best_lambda = l
    return best_lambda

if __name__ == "__main__":
    PROJECT_ROOT = "/home/mithunmanivannan"
    
    ecgfm_best = find_best_lambda(os.path.join(PROJECT_ROOT, "checkpoints/fast_ecgfm"))
    hubert_best = find_best_lambda(os.path.join(PROJECT_ROOT, "checkpoints/theory_validation"))
    
    print(f"ECGFM_BEST={ecgfm_best}")
    print(f"HUBERT_BEST={hubert_best}")
