import json
import itertools
import random
from pathlib import Path

def get_lead_subset_bank(bank_path: Path, seed: int = 2026) -> dict:
    if bank_path.exists():
        with open(bank_path, "r") as f:
            return json.load(f)
            
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # E5: 5 random combinations per lead count
    # E6: up to 25 random combinations per lead count
    rng = random.Random(seed)
    bank = {}
    for k in range(1, 13):
        # generate all combinations of length k
        all_combs = list(itertools.combinations(leads, k))
        
        # sample up to 25
        num_samples = min(25, len(all_combs))
        
        # for consistency, sort before sampling so it's deterministic
        all_combs.sort()
        
        if num_samples == len(all_combs):
            selected = all_combs
        else:
            selected = rng.sample(all_combs, num_samples)
            
        bank[str(k)] = [list(comb) for comb in selected]
        
    with open(bank_path, "w") as f:
        json.dump(bank, f, indent=2)
        
    return bank
