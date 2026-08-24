
import torch
import numpy as np
import copy
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from src.models.classifier import xresnet1d101

def train_optimized_xresnet_classifier(train_loader, val_loader, test_loader, device, num_classes=5, input_channels=12, epochs=50, seed=42, use_compile=True):
    """
    Train XResNet1d-101 on 12-lead ECG with high-performance optimizations.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True # Kernel selection optimization
    
    # Build model
    model = xresnet1d101(num_classes=num_classes, input_channels=input_channels).to(device)
    
    # [HACK] Torch Compile for graph-level fusion
    if use_compile and hasattr(torch, 'compile'):
        try:
            print("  > Applying torch.compile()...")
            model = torch.compile(model)
        except Exception as e:
            print(f"  ! torch.compile failed: {e}")

    # NVIDIA Hacks: Use fused=True for faster updates on A100
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2, fused=True)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = torch.nn.BCEWithLogitsLoss() 
    
    # [HACK] Automatic Mixed Precision (FP16)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    early_stop_count = 0
    patience = 5
    best_state = None
    
    print(f"  > Training Optimized Downstream Classifier (Seed {seed})...")
    
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            # [HACK] Non-blocking transfers
            x_batch = batch['input'].to(device, non_blocking=True)
            y_batch = batch['label'].to(device, non_blocking=True).float()
            
            # [HACK] Autocast for Mixed Precision
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            if (i+1) % 50 == 0:
                print(f"    Epoch {epoch+1} | Step {i+1} | Loss: {loss.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch['input'].to(device, non_blocking=True)
                y_batch = batch['label'].to(device, non_blocking=True).float()
                
                with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)
                    
                val_loss += loss.item() * x_batch.size(0)
                count += x_batch.size(0)
        val_loss /= count
        
        # Early Stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            # Handle compiled model serialization
            actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            best_state = copy.deepcopy(actual_model.state_dict())
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

        # [HACK] Regular checkpointing for reliability
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/diagnostic_oracle_epoch{epoch+1}.pt"
            actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(actual_model.state_dict(), checkpoint_path)
            print(f"    > Periodic checkpoint saved to {checkpoint_path}")
    
    # Save final best model
    if best_state:
        final_path = "checkpoints/diagnostic_oracle_best.pt"
        torch.save(best_state, final_path)
        print(f"  > Final best model saved to {final_path}")
    
    # Test evaluation
    if best_state:
        actual_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        actual_model.load_state_dict(best_state)
    
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch['input'].to(device, non_blocking=True)
            y_batch = batch['label'] # Already on CPU for concat later
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                logits = model(x_batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y_batch.numpy())
            
    test_probs = np.concatenate(all_probs, axis=0)
    test_labels = np.concatenate(all_targets, axis=0)
    
    # AUROC Calculation...
    num_classes = test_labels.shape[1]
    auroc_per_class = []
    for i in range(num_classes):
        try:
            score = roc_auc_score(test_labels[:, i], test_probs[:, i])
            auroc_per_class.append(score)
        except ValueError:
            auroc_per_class.append(0.5)
            
    return auroc_per_class, np.mean(auroc_per_class), test_probs, test_labels
