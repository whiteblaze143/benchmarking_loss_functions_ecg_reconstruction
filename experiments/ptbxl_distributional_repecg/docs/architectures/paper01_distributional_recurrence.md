# Paper 01 Tutorial: Distributional Recurrence Operator

## 1. Theoretical Foundation

### The Biological Problem
How do different phases of the cardiac cycle interact with each other statistically? For example, how does the atrial depolarization (P-wave) influence the ventricular repolarization (T-wave)?

### The Mathematical Solution: Maximum Mean Discrepancy (MMD)
Traditional recurrence plots in physics measure whether the raw scalar voltage at time $t_i$ is close to the voltage at $t_j$. We upgrade this to the distributional level using **Maximum Mean Discrepancy (MMD)**.

MMD is a rigorous statistical test that measures the distance between two probability distributions $P$ and $Q$ by calculating the distance between their Kernel Mean Embeddings in the RKHS:
$$ MMD^2(P, Q) = ||\mu_P - \mu_Q||^2_{\mathcal{H}} $$

If we have Nyström-approximated vectors $K_i$ and $K_j$ for phase cells $i$ and $j$, the MMD distance is simply the squared Euclidean distance:
$$ R_{i, j} = ||K_i - K_j||^2_2 $$

## 2. Implementation Tutorial

### Step-by-Step Architecture
1.  **Input**: The model receives a sequence of 16 KME vectors for a heartbeat. Tensor shape: `[Batch, 16, 128]`.
2.  **Operator Construction**: We dynamically compute the $16 \times 16$ pairwise MMD recurrence matrix on the GPU using `torch.cdist`. Tensor shape becomes `[Batch, 16, 16]`.
3.  **Encoding**: The $16 \times 16$ matrix is treated as a 1-channel image (`[Batch, 1, 16, 16]`) and passed through a lightweight CNN (e.g., `RecurrenceCNN`) to extract spatial recurrence patterns.
4.  **Classification**: A final linear head predicts the 5 diagnostic superclasses (Normal, MI, STTC, CD, HYP).

```python
# PyTorch Pseudocode for Paper 01
import torch
import torch.nn as nn

class RecurrenceCNN(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(32, classes)
        
    def forward(self, x):
        # x is the sequence of KMEs: [Batch, 16, 128]
        # Compute MMD recurrence matrix: [Batch, 16, 16]
        recurrence_matrix = torch.cdist(x, x, p=2).pow(2)
        
        # Add channel dim: [Batch, 1, 16, 16]
        features = self.encoder(recurrence_matrix.unsqueeze(1)).flatten(1)
        return self.head(features)
```

## 3. Rigorous Falsification (The Kill Test)
To publish this, we must prove the model isn't cheating.
*   **Matched Control**: We replace the KME vectors with simple mean-voltage vectors and run the exact same `RecurrenceCNN`. The true MMD matrix must heavily outperform this naive control.
*   **The Kill Test**: We randomly shuffle the chronological order of the 16 phase cells before computing the recurrence matrix. Because biological recurrence relies on strict temporal causality, this shuffling must completely destroy the model's accuracy. If it doesn't, the model is falsified.
