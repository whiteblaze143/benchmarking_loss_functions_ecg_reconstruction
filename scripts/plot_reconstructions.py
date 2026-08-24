#!/usr/bin/env python3
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate side-by-side PNG visualizations")
    parser.add_argument("--save_dir", type=str, default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Generating reconstruction plots in {args.save_dir}...")

    # Mock plotting
    for i in range(3):
        fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
        for j in range(6):
            target = np.sin(np.linspace(0, 4*np.pi, 1000)) + np.random.randn(1000) * 0.1
            recon = target + np.random.randn(1000) * 0.05
            
            axes[j].plot(target, label="Raw Target", color="blue", alpha=0.7)
            axes[j].plot(recon, label="Reconstruction", color="red", alpha=0.7, linestyle="--")
            axes[j].set_ylabel(f"Lead V{j+1}")
            if j == 0:
                axes[j].legend(loc="upper right")
                
        axes[-1].set_xlabel("Time (samples)")
        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f"recon_case_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"  Saved {save_path}")

if __name__ == "__main__":
    main()
