import torch
import glob
import os

print("Finding checkpoints...")
files = glob.glob("checkpoints/factorial_1*.pt")
print(f"Found {len(files)} factorial checkpoints.")

freed = 0
for file in files:
    size_before = os.path.getsize(file)
    try:
        ckpt = torch.load(file, map_location='cpu')
        
        # Only convert if it hasn't been converted already
        if "model_state_dict" in ckpt:
            # Check the dtype of the first tensor
            first_tensor = next(iter(ckpt["model_state_dict"].values()))
            if first_tensor.dtype == torch.float32:
                model_state = {k: v.to(torch.float16) for k, v in ckpt["model_state_dict"].items()}
                ckpt["model_state_dict"] = model_state
                torch.save(ckpt, file)
                size_after = os.path.getsize(file)
                freed += (size_before - size_after)
    except Exception as e:
        print(f"Failed {file}: {e}")

print(f"Finished. Freed {freed / 1024 / 1024:.2f} MB")
