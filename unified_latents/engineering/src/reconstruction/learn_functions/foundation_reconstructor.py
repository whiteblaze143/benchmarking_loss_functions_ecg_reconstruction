#!/usr/bin/env python3
import torch
import torch.nn as nn
import os

class FoundationReconstructor(object):
    """
    Wrapper for Foundation Model Bridges to fit the Mason et al. (2024) architecture.
    Follows the 'object' with list-of-leads interface found in reconstructor.py.
    """
    def __init__(self, bridge, input_indices, device):
        self.bridge = bridge.to(device)
        self.input_indices = input_indices
        self.device = device
        self.output_lead_num = 12 # Standard 12-lead reconstruction
        
        # Placeholder for author's stats plotting (not compatible with transformers)
        self.input_network = self
        self.middle_network = self
        self.output_network = self

    def reset(self):
        """Foundation models weights are typically not reset in this protocol."""
        pass
        
    def parameters(self):
        """Returns a list of parameters for the optimizer."""
        return list(self.bridge.parameters())
        
    def named_parameters(self):
        """Returns dummy structure to satisfy compute_model_stats if called."""
        return {0: {}, 1: {}, 2: {}}
        
    def save_state_dict(self, path: str):
        """Saves the bridge state."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        torch.save(self.bridge.state_dict(), os.path.join(path, "bridge.pt"))
        
    def load_state_dict(self, path: str):
        """Loads the bridge state."""
        checkpoint_path = os.path.join(path, "bridge.pt")
        if os.path.exists(checkpoint_path):
            self.bridge.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
    def forward(self, input_leads):
        """
        Forward pass.
        Args:
            input_leads: list of N tensors, each (BatchSize, 1, SeqLen)
        Returns:
            list of 12 tensors, each (BatchSize, SeqLen)
        """
        # 1. Concatenate into (B, N, T)
        x = torch.cat(input_leads, dim=1) 
        
        # 2. Run through bridge
        # Bridge is expected to handle internal padding to 12 if needed
        # and use the provided lead_indices for its adapter.
        output = self.bridge(x, lead_indices=self.input_indices) # (B, 12, T)
        
        # 3. Split back into list of 12 tensors (B, T) as per ReconstructionManager
        return [output[:, i, :] for i in range(12)]
