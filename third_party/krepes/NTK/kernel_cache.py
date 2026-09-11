import torch
import pathlib
import logging
import os

def get_kernel_dir(kernel_name, config):
    cache_dir = pathlib.Path("kernel_cache")
    kernel_dir_name = f"{kernel_name}_{config.dataset}_k{config.k}_m{config.n_landmark}"
    return cache_dir / kernel_dir_name

def save_kernel_batch(kernel_tensor, kernel_name, batch_idx, config):
    kernel_dir = get_kernel_dir(kernel_name, config)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = kernel_dir / f"batch_{batch_idx}.pt"
    torch.save(kernel_tensor, file_path)
    logging.info(f"Saved {kernel_name} batch {batch_idx} to {file_path}")

def load_kernel_batch(kernel_name, batch_idx, config):
    kernel_dir = get_kernel_dir(kernel_name, config)
    file_path = kernel_dir / f"batch_{batch_idx}.pt"
    
    if os.path.exists(file_path):
        return torch.load(file_path, map_location=config.device)
    else:
        return None

def save_kernel_full(kernel_tensor, kernel_name, config):

    kernel_dir = get_kernel_dir(kernel_name, config)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    file_path = kernel_dir / "full_kernel.pt"
    torch.save(kernel_tensor, file_path)
    logging.info(f"Saved full {kernel_name} kernel to {file_path}")

def load_kernel_full(kernel_name, config):

    kernel_dir = get_kernel_dir(kernel_name, config)
    file_path = kernel_dir / "full_kernel.pt"
    
    if os.path.exists(file_path):
        logging.info(f"Loading full {kernel_name} kernel from {file_path}")
        return torch.load(file_path, map_location=config.device)
    else:
        return None


class KernelLoader:

    def __init__(self, kernel_path):

        self.kernel_path = kernel_path
        self.kernel_matrix = None
        self._load_kernel()

    def _load_kernel(self):

        logging.info(f"Loading kernel from: {self.kernel_path}")
        self.kernel_matrix = torch.load(self.kernel_path, map_location='cpu')
        logging.info(f"Successfully loaded kernel with shape: {self.kernel_matrix.shape}")

    def get_batch(self, indices):

        if self.kernel_matrix is None:
            raise RuntimeError("Kernel has not been loaded. Call _load_kernel() first.")
        
        return self.kernel_matrix[indices]