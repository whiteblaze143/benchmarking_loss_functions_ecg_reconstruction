import torch
from torch import nn
from typing import Iterator

def finetune_model_for_configuration(
    model: nn.Module,
    train_loader: Iterator,
    val_loader: Iterator,
    config_name: str,
    epochs: int = 5,
    lr: float = 1e-4
) -> nn.Module:
    """
    E13 & E14: Configuration-specific fine-tuning.
    Args:
        model: Pretrained zero-shot model
        train_loader: Dataloader that yields batches (x, y) formatted for the configuration
        val_loader: Dataloader for validation
        config_name: 6limb, 1lead_II, ICM, etc.
        epochs: Number of adaptation epochs
        lr: Learning rate
    """
    import copy
    import torch.optim as optim
    import torch.nn.functional as F
    
    # Create a copy to prevent modifying the original zero-shot weights
    adapted_model = copy.deepcopy(model)
    adapted_model.train()
    
    optimizer = optim.AdamW(adapted_model.parameters(), lr=lr)
    
    # Simple fine-tuning loop
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = adapted_model(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits, batch_y.float())
                
            loss.backward()
            optimizer.step()
            
    adapted_model.eval()
    return adapted_model
