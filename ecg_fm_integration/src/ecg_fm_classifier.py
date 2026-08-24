#!/usr/bin/env python3
"""
ECG-FM Classifier Module

Provides the ECGFMClassifier class for classification tasks using ECG-FM backbone.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add fairseq-signals to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'fairseq-signals'))


class ECGFMClassificationHead(nn.Module):
    """Classification head for ECG-FM embeddings."""
    
    def __init__(self, embed_dim: int = 768, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ECGFMClassifier(nn.Module):
    """Complete ECG-FM classifier with frozen backbone and trainable head."""
    
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 5,
        dropout: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load ECG-FM backbone
        from fairseq_signals.utils import checkpoint_utils
        
        self.backbone, self.cfg, self.task = checkpoint_utils.load_model_and_task(checkpoint_path)
        self.backbone.eval()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = 768  # ECG-FM uses 768-dim embeddings
        
        # Create classification head
        self.head = ECGFMClassificationHead(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    
    def extract_embeddings(self, x: torch.Tensor, pool: bool = True) -> torch.Tensor:
        """Extract embeddings from ECG-FM backbone."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        if x.shape[1] != 12:
            pad = torch.zeros(x.shape[0], 12 - x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone.extract_features(x, padding_mask=None)
            embeddings = features['x']
        
        if pool:
            embeddings = embeddings.mean(dim=1)
        
        return embeddings
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and classification head."""
        embeddings = self.extract_embeddings(x, pool=True)
        logits = self.head(embeddings)
        return logits
