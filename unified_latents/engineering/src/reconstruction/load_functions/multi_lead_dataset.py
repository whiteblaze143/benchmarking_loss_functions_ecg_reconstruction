"""
Multi-Lead Dataset wrapper for PTB-XL.
Provides anatomical-group lead masking strategies for bridge training.
"""

import random
import torch
from torch.utils.data import Dataset


# Standard Lead Order
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


class MultiLeadPTBXL(Dataset):
    ANATOMICAL_GROUPS = {
        'limb': [0, 1, 2],       # I, II, III
        'augmented': [3, 4, 5],  # aVR, aVL, aVF
        'v1v3': [6, 7, 8],      # V1, V2, V3
        'v4v6': [9, 10, 11],    # V4, V5, V6
        'sota_core': [0, 1, 8], # I, II, V3 (Central Lead Set)
        'wearecg': [1, 6, 10]   # II, V1, V5 (Orthogonal Triad)
    }

    def __init__(self, ptbxl_ds, strategy='random', deterministic=False):
        self.ds = ptbxl_ds
        self.strategy = strategy  # 'random', 'structured', 'mixed' or a specific group name
        self.deterministic = deterministic

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        _, gt = self.ds[idx]  # gt is (12, L)

        if self.deterministic:
            rng = random.Random(idx)
        else:
            rng = random

        current_strat = self.strategy
        if current_strat == 'mixed':
            current_strat = 'structured' if rng.random() < 0.5 else 'random'

        if current_strat == 'structured':
            # Pick one anatomical group
            group_name = rng.choice(list(self.ANATOMICAL_GROUPS.keys()))
            indices = sorted(self.ANATOMICAL_GROUPS[group_name])
        elif current_strat in self.ANATOMICAL_GROUPS:
            # Pick the specific group requested
            indices = sorted(self.ANATOMICAL_GROUPS[current_strat])
        else:
            # Random masking (1-3 leads)
            n = rng.randint(1, 3)
            indices = sorted(rng.sample(range(12), n))

        x_in = gt[indices, :]
        return x_in, gt, torch.tensor(indices)


def collate_multi_lead(batch):
    x_list, gt_list, idx_list = zip(*batch)
    target_indices = idx_list[0].tolist()
    new_x_list = []
    for gt in gt_list:
        new_x_list.append(gt[target_indices, :])
    return torch.stack(new_x_list), torch.stack(gt_list), torch.tensor(target_indices)
