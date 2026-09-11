import torch
import torch.nn.functional as F
import numpy as np
import time
import copy
import wandb
from tqdm.auto import tqdm
from utils import epoch_eval, eval, sup_evaluate, linear_probe
from .model import CNN_sup, FC_Supervised, TinyTransformer, ResMLP, Resnet, BYOL, Projector, SSLModel

def one_hot_encode(y, num_classes):
        return F.one_hot(y, num_classes=num_classes).float()

def train(data, criterion, cnf):

    (train_loader, valid_loader, test_loader) = data
 
    train_size = len(train_loader.dataset)
    n_classes = cnf.N_cls
    DEVICE = cnf.device

    if cnf.loss_f == 'byol':
        if cnf.dataset in ['cifar10', 'imagenet']:
            backbone = Resnet().to(DEVICE)
            feature_dim = 512
        elif cnf.dataset in ['adult', 'covetype', 'higgs']:
            backbone = ResMLP(in_features=cnf.in_features, width=cnf.mlp_width, num_blocks=cnf.num_blocks).to(DEVICE)
            feature_dim = cnf.mlp_width   
        nn_model = BYOL(backbone, feature_dim).to(cnf.device)
    else:
        if cnf.dataset in ['cifar10', 'imagenet']:
            nn_model = Resnet().to(DEVICE)
        elif cnf.dataset in ['adult', 'covetype', 'higgs']:
            nn_model = ResMLP(in_features=cnf.in_features, width=cnf.mlp_width, num_blocks=cnf.num_blocks).to(DEVICE)
    
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=cnf.learning_rate, weight_decay=cnf.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    num_warmup_steps = train_size / cnf.batch_size


    best_acc = float('-inf')
    best_model_state = None
    patience = cnf.patience
    losses = []
    nn_times = []
    
    for epoch in tqdm(range(cnf.epochs), desc="Epochs"):
        epoch_loss = 0
        num_batches = 0
        nn_model.train()
        # for i in tqdm(range(0, n, cnf.batch_size), desc=f"Epoch {epoch+1}/{cnf.epochs}", leave=False):
        for i, (batch_X, batch_X_aug, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            current_step = epoch * train_size + i
            if current_step < num_warmup_steps:
                lr_scale = float(current_step + 1) / float(num_warmup_steps)
                for g in optimizer.param_groups:
                    g['lr'] = cnf.learning_rate * lr_scale

            batch_X, batch_X_aug, batch_y = batch_X.to(DEVICE), batch_X_aug.to(DEVICE), batch_y.to(DEVICE)

            start_time_nn = time.time()
            optimizer.zero_grad()
       
            if cnf.loss_f == 'byol':
                online_pred1, online_pred2, target_z1, target_z2 = nn_model(batch_X, batch_X_aug)
                loss1 = criterion(online_pred1, target_z2)
                loss2 = criterion(online_pred2, target_z1)
                loss = loss1 + loss2
                
            else:
                z_A = nn_model(batch_X)
                z_B = nn_model(batch_X_aug)

                if cnf.loss_f == 'vicreg':
                    loss = criterion(z_A, z_B, train_size, cnf.tau, cnf.lambda_inv, cnf.mu_var, cnf.nu_cov)
                elif cnf.loss_f == 'bt':
                    loss = criterion(z_A, z_B, cnf.lambda_reg)
                elif cnf.loss_f == 'simclr':

                    loss = criterion(z_A, z_B)
                elif cnf.loss_f == 'spectral' or cnf.loss_f == 'simple':
                    z_C = nn_model(neg_X[i:i+cnf.batch_size])
                    loss = criterion(z_A, z_B, z_C, cnf.lambda_reg, nn_model)
                    
            losses.append(loss.item())
            epoch_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
            wandb.log({"nn_grad_norm": \
                       torch.sqrt(sum(p.grad.norm()**2 \
                                      for p in nn_model.parameters() \
                                        if p.grad is not None)).item()})
            optimizer.step()
            if cnf.loss_f == 'byol':
                nn_model.update_target_network(cnf.ema_decay) 
            end_time_nn = time.time()
            nn_times.append(end_time_nn - start_time_nn)

        avg_epoch_loss = epoch_loss / num_batches
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
    
        wandb.log({
            "learning_rate": current_lr,
            "Epoch": epoch + 1,
            "nn Loss": avg_epoch_loss,
            "nn_grad_norm": \
                       torch.sqrt(sum(p.grad.norm()**2 \
                                      for p in nn_model.parameters() \
                                        if p.grad is not None)).item()
        })
        nn_model.eval()
        with torch.no_grad():
            if cnf.loss_f == 'byol':
                train_acc = epoch_eval(backbone, valid_loader, cnf)
            else:
                train_acc = epoch_eval(nn_model, valid_loader, cnf)

        wandb.log({
            "Epoch": epoch + 1,
            "Train Accuracy": train_acc
        })
        if train_acc > best_acc:
            best_acc = train_acc
            best_model_state = copy.deepcopy(nn_model.state_dict())
            patience = cnf.patience
        else:
            patience -= 1
            if patience == 0:
                wandb.log({"Early Stopping Epoch": epoch + 1})
                break   

    avg_time_nn = np.mean(nn_times)
    wandb.log({
    "Average Time per Epoch (nn)": avg_time_nn
})
    if best_model_state is not None:
        nn_model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state found. Using the last model state.")
    nn_model.eval()
    with torch.no_grad():
        if cnf.loss_f == 'byol':
            test_acc = eval(backbone, valid_loader, test_loader, cnf)
        else:
            test_acc = eval(nn_model, valid_loader, test_loader, cnf)

    return train_acc, test_acc, nn_model