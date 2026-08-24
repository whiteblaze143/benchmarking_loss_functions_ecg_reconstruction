import torch
import numpy as np
from torch.utils.data import DataLoader, Sampler
import random

class MasonBalancedSampler(Sampler):
    """
    Ensures each diagnostic superclass is represented in every batch.
    NORM, MI, STTC, CD, HYP
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.labels = np.array(data_source.labels)
        self.classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
        # Calculate how many of each class per batch
        self.num_per_class = batch_size // len(self.classes)
        self.remaining = batch_size % len(self.classes)
        
        self.num_batches = len(data_source) // batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for c in self.classes:
                indices = self.class_indices[c]
                if len(indices) > 0:
                    selected = np.random.choice(indices, self.num_per_class, replace=True)
                    batch.extend(selected.tolist())
                else:
                    # Fallback if a class is missing (unlikely in PTB-XL)
                    indices = np.arange(len(self.data_source))
                    selected = np.random.choice(indices, self.num_per_class, replace=True)
                    batch.extend(selected.tolist())
            
            # Fill remaining with random samples
            if self.remaining > 0:
                indices = np.arange(len(self.data_source))
                selected = np.random.choice(indices, self.remaining, replace=True)
                batch.extend(selected.tolist())
            
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

class MasonProtocolDataLoader:
    """
    Wrap PyTorch DataLoader with Prioritized Experience Replay (PER).
    """
    def __init__(self, dataset, batch_size, prioritize_percent=0.25, prioritize_size=1024, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.prioritize_percent = prioritize_percent
        self.prioritize_size = prioritize_size
        
        self.sampler = MasonBalancedSampler(dataset, batch_size)
        self.loader = DataLoader(dataset, batch_sampler=self.sampler, num_workers=num_workers)
        
        self.priority_buffer_x = []
        self.priority_buffer_y = []
        self.priority_buffer_demo = []
        self.priority_losses = []
        
        self.num_priority_per_batch = int(batch_size * prioritize_percent)
        self.num_fresh_per_batch = batch_size - self.num_priority_per_batch

    def __iter__(self):
        for fresh_x, fresh_y, fresh_demo in self.loader:
            if len(self.priority_buffer_x) >= self.batch_size:
                # Augment with priority samples
                p_indices = np.random.choice(len(self.priority_buffer_x), self.num_priority_per_batch, replace=False)
                
                px = torch.stack([self.priority_buffer_x[i] for i in p_indices])
                py = torch.stack([self.priority_buffer_y[i] for i in p_indices])
                
                # Stack dictionary components
                pd = {}
                for k in self.priority_buffer_demo[0].keys():
                    pd[k] = torch.stack([self.priority_buffer_demo[i][k] for i in p_indices])
                
                # Replace last N fresh samples with priority samples
                x = torch.cat([fresh_x[:self.num_fresh_per_batch], px], dim=0)
                y = torch.cat([fresh_y[:self.num_fresh_per_batch], py], dim=0)
                
                # Concatenate dictionary components
                demo = {}
                for k in fresh_demo.keys():
                    demo[k] = torch.cat([fresh_demo[k][:self.num_fresh_per_batch], pd[k]], dim=0)
            else:
                x, y, demo = fresh_x, fresh_y, fresh_demo
            
            yield x, y, demo

    def update_priority(self, x, y, demo, losses):
        """
        Update buffer with high-loss samples from the current batch.
        losses: (B,) tensor of R2 or MSE errors per record.
        """
        # Take top-K from current batch
        k = self.num_priority_per_batch
        if k > 0:
            top_losses, top_indices = torch.topk(losses, min(k, len(losses)))
            for idx in top_indices:
                self.priority_buffer_x.append(x[idx].detach().cpu())
                self.priority_buffer_y.append(y[idx].detach().cpu())
                # Handle dictionary indexing for buffer
                if isinstance(demo, dict):
                    sample_demo = {k: v[idx].detach().cpu() for k, v in demo.items()}
                else:
                    sample_demo = demo[idx].detach().cpu()
                self.priority_buffer_demo.append(sample_demo)
                
            # Keep buffer size stable
            if len(self.priority_buffer_x) > self.prioritize_size:
                self.priority_buffer_x = self.priority_buffer_x[-self.prioritize_size:]
                self.priority_buffer_y = self.priority_buffer_y[-self.prioritize_size:]
                self.priority_buffer_demo = self.priority_buffer_demo[-self.prioritize_size:]

    def __len__(self):
        return len(self.loader)
