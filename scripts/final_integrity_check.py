#!/usr/bin/env python3
import numpy as np
from scipy.stats import pearsonr

# Load samples
m0 = np.load('results/blinded_m0_10samples.npy')
m1 = np.load('results/blinded_m1hpo_10samples.npy')
target = np.load('results/blinded_targets_10samples.npy')

print("=== FINAL INTEGRITY AUDIT (M1-HPO vs M0) ===")
print(f"{'Sample':<10} | {'M0 Corr':<10} | {'M1 Corr':<10} | {'Advantage':<10}")
print("-" * 50)

for i in range(10):
    # Flatten across 12 leads and time
    gt = target[i].flatten()
    s0 = m0[i].flatten()
    s1 = m1[i].flatten()
    
    c0, _ = pearsonr(gt, s0)
    c1, _ = pearsonr(gt, s1)
    
    adv = "M1 (+)" if c1 > c0 else "M0 (-)"
    print(f"{i+1:02d}         | {c0:.4f}     | {c1:.4f}     | {adv}")

print("\nConclusion: M1-HPO should ideally show higher global correlation and better ST preservation.")
