# Paper 02 Tutorial: Local Phase Kernel Mean Embeddings (Phase-CNN)

## 1. Theoretical Foundation

### The Core Question
Do we really need to explicitly calculate and store a massive, dense $N \times N$ recurrence matrix (as in Paper 01)? Can a standard neural network simply look at the chronological sequence of local KMEs and figure out the temporal transitions itself?

### The Mathematical Object
Instead of a pairwise matrix, the mathematical object fed into this model is the raw temporal sequence of the kernel means: $\mathbf{K} = (K_1, K_2, \dots, K_G)$. 

## 2. Implementation Tutorial

### Step-by-Step Architecture
We employ a lightweight 1D Convolutional Neural Network (CNN). 
The 1D-CNN slides its convolutional filters sequentially over the temporal sequence of KME vectors, treating the sequence of distributions exactly like a 1D sequence of pixels.

```python
# PyTorch Pseudocode for Paper 02
import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=0)
        self.conv2 = nn.Conv1d(width, width, 3, padding=0)
        
    def forward(self, x):
        residual = x
        # Circular padding to handle the cyclic nature of the heartbeat!
        x_pad = F.pad(x, (1, 1), mode="circular")
        x = F.gelu(self.conv1(x_pad))
        x_pad = F.pad(x, (1, 1), mode="circular")
        return F.gelu(self.conv2(x_pad) + residual)

class PhaseCNN(nn.Module):
    def __init__(self, input_dim=128, classes=5):
        super().__init__()
        self.input = nn.Conv1d(input_dim, 128, 1)
        self.blocks = nn.Sequential(
            CircularResidualBlock(128),
            CircularResidualBlock(128)
        )
        self.head = nn.Linear(128, classes)
        
    def forward(self, x):
        # x shape: [Batch, 16_phases, 128_features]
        # Transpose for Conv1d: [Batch, 128, 16]
        x = x.transpose(1, 2)
        features = self.blocks(self.input(x)).mean(dim=-1)
        return self.head(features)
```

### The Biological Context
CNNs are translationally invariant. By sliding a CNN over statistical KMEs rather than raw waveforms, the network is forced to focus on **macro-state transitions** (e.g., the smooth, distributional transition from the QRS complex into the ST segment) rather than overfitting to high-frequency micro-voltage noise.

## 3. Rigorous Falsification (The Kill Test)
*   **Matched Control**: A standard 1D-CNN with identical parameter capacity applied directly to the raw 8-lead physical voltages.
*   **The Kill Test**: We apply a heavy temporal smoothing filter (a moving average) to the sequence of KMEs during inference. If the model is genuinely learning local, sharp transitions between distinct phase distributions, over-smoothing the phases will collapse these boundaries and significantly destroy the model's accuracy.
