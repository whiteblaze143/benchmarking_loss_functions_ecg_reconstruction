
import math
import torch
import numpy as np
from torch.utils.data import Sampler, Dataset

class MultiScaleConfig:
    """ Configuration for Multi-Scale Training """
    SHORT_SEQ_LEN = 2500  # 10s @ 250Hz
    MEDIUM_SEQ_LEN = 7500 # 30s @ 250Hz
    LONG_SEQ_LEN = 150000 # 10min @ 250Hz
    
    # Probability Distribution
    P_SHORT = 0.30
    P_MEDIUM = 0.50
    P_LONG = 0.20

class MultiScaleSampler(Sampler):
    """
    Sampler that yields batches of indices with a specific scale (Short, Medium, Long).
    Ensures that each batch contains samples of ONLY ONE scale to allow tensor stacking.
    Mixing happens across batches, not within batches.
    """
    def __init__(self, data_source: Dataset, batch_size: int, config: MultiScaleConfig):
        self.data_source = data_source
        self.batch_size = batch_size
        self.config = config
        self.num_samples = len(data_source)
        
        # Calculate number of batches per scale based on probabilities
        total_batches = self.num_samples // batch_size
        self.n_short = int(total_batches * config.P_SHORT)
        self.n_medium = int(total_batches * config.P_MEDIUM)
        self.n_long = total_batches - self.n_short - self.n_medium
        
    def __iter__(self):
        # Create a pool of "batch types"
        batch_types = ([0] * self.n_short) + ([1] * self.n_medium) + ([2] * self.n_long)
        np.random.shuffle(batch_types) # Mixing strategy
        
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices) # Randomize sample order
        
        idx_ptr = 0
        for b_type in batch_types:
            batch_indices = indices[idx_ptr : idx_ptr + self.batch_size]
            idx_ptr += self.batch_size
            
            # Yield tuple: (indices, scale_type)
            # scale_type: 0=Short, 1=Medium, 2=Long
            yield batch_indices.tolist(), b_type

    def __len__(self):
        return self.n_short + self.n_medium + self.n_long

def get_crop_length(scale_type):
    if scale_type == 0:
        return MultiScaleConfig.SHORT_SEQ_LEN
    elif scale_type == 1:
        return MultiScaleConfig.MEDIUM_SEQ_LEN
    else:
        return MultiScaleConfig.LONG_SEQ_LEN
