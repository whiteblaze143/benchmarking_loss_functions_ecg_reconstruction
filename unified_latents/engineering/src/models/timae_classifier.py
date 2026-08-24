
import torch
import torch.nn as nn
from src.models.ti_mae import TiMAE

class TiMAEClassifier(nn.Module):
    """
    Downstream Classifier using Pretrained Ti-MAE Encoder.
    """
    def __init__(self, num_classes=5, pretrained_path=None, freeze_encoder=True, **kwargs):
        super().__init__()
        
        # Initialize TiMAE with same config as pretraining (usually default)
        # 10s = 2500 samples (at 250Hz) or 5000 at 500Hz?
        # Pretraining was multi-scale. 
        # TiMAE default seq_len=5000 (from file).
        # We need to ensure we pass correct args.
        self.encoder = TiMAE(**kwargs)
        
        if pretrained_path:
            print(f"Loading Ti-MAE weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            # Handle if state_dict has 'module.' prefix or is full checkpoint
            state_dict = checkpoint
            if 'model' in checkpoint: # heuristic if saved as dict
                state_dict = checkpoint['model']
            
            # Remove 'decoder' weights if we only want encoder? 
            # TiMAE loads everything. It's fine.
            msg = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"Load status: {msg}")
            
        # Freeze?
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Ti-MAE Encoder Frozen.")
        else:
            print("Ti-MAE Encoder Unfrozen (Fine-Tuning).")
            
        # Classification Head
        self.head = nn.Linear(self.encoder.embed_dim, num_classes)
        
        # Initialize head
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        
    def forward(self, x):
        # x: [B, 1, L]
        # TiMAE forward_encoder returns x (with CLS), mask, ids
        latent, _, _ = self.encoder.forward_encoder(x)
        
        # Extract CLS token (Index 0)
        cls_token = latent[:, 0, :] # [B, embed_dim]
        
        logits = self.head(cls_token)
        return logits
