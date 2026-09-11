import torch
from sklearn.metrics import balanced_accuracy_score
import time
import wandb
import copy
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader
from NTK import compute_ntk, ConvNet, ResMLP, KernelLoader
from utils import epoch_eval,\
                eval, rbf_kernel_torch, epoch_eval_cached, \
                eval_cached, epoch_eval_imgnet, eval_imgnet
from model.precondition import KmmPCGPreconditioner, BarlowTwinsGNHPreconditioner, SimclrGNHPreconditioner


def train_kerepes_model(model, config, data):
    
    if config.kcache:
        KA_path = config.KA_path
        KB_path = config.KB_path
        Kmm_path = config.Kmm_path
        Kv_path = config.Kv_path
        Kt_path = config.Kt_path
        kA_loader = KernelLoader(KA_path)
        kB_loader = KernelLoader(KB_path)
        kv_loader = KernelLoader(Kv_path)
        kt_loader = KernelLoader(Kt_path)
        ntk_model = None
        (y, y_valid, y_test, C_var, y_C) = data
        train_size = len(y)
        n_classes = len(torch.unique(y))
        indices_dataset = TensorDataset(torch.arange(train_size))
        train_iterator = DataLoader(indices_dataset, batch_size=config.batch_size, shuffle=False)
    else:
        if config.data_mod == 'tab':
            (train_loader, valid_loader, test_loader, C_var, y_C) = data
            posX_loader = None 
        else:
            (train_loader, posX_loader, valid_loader, test_loader, C_var, y_C) = data

        train_size = len(train_loader.dataset)
        train_iterator = train_loader
        n_classes = len(torch.unique(train_loader.dataset.targets))



    device = config.device
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    num_warmup_steps = len(y) / config.batch_size

    params_to_precondition = [p for name, p in model.named_parameters() if name == 'A']
    preconditioner = None
    if config.pcnr_type == 'PCG':
        preconditioner = KmmPCGPreconditioner(
            model, 
            params_to_precondition=params_to_precondition, 
            reg_lambda=config.reg_lambda, 
            cg_max_iter=config.cg_max_iter
        )

    elif config.pcnr_type == 'BT_GNH':
        if config.loss_f!= 'bt':
            raise ValueError("This Gauss-Newton preconditioner is only valid for the 'bt' loss function.")
        preconditioner = BarlowTwinsGNHPreconditioner(
            model,
            params_to_precondition=params_to_precondition,
            loss_fn=model.criterion,
            damping=config.reg_lambda,
            cg_max_iter=config.cg_max_iter
        )
    elif config.pcnr_type == 'SIMCLR_GNH':
        print("Using SimclrGNHPreconditioner.")
        if config.loss_f!= 'simclr':
            raise ValueError("This Gauss-Newton preconditioner is only valid for the 'simclr' loss function.")
        preconditioner = SimclrGNHPreconditioner(
            model,
            params_to_precondition=params_to_precondition,
            temperature=config.tau,
            damping=config.reg_lambda,
            cg_max_iter=config.cg_max_iter
        )

    best_acc = float('-inf')
    best_model_state = None
    patience = config.patience
    losses, nystrom_times, epoch_time= [], [], []
    
    if not config.kcache:
        if config.kernel_fn == 'rbf':
            ntk_model = None
        else:
            if config.dataset in ['mnist', 'fashion_mnist']:
                ntk_model = ConvNet(in_channels=1, num_classes=10, input_size_hw=(28, 28)).to(device)
            elif config.data_mod == 'tab':
                ntk_model = ResMLP(in_features=C_var.shape[-1], width=config.width, num_blocks=config.num_blocks, num_classes=n_classes).to(device)

    # --- INITIALIZATION ONCE HERE ---
    if config.kcache:
        K_mm = torch.load(Kmm_path).to(device)
    else:
        K_mm = compute_ntk(ntk_model, C_var, C_var).detach() if config.kernel_fn == 'torch_ntk' else rbf_kernel_torch(C_var, C_var, config.gamma)

    if config.init_meth == 'eigen_dir':
        model.initialize_A_from_eigen(K_mm)
        if config.loss_f == 'byol':
            model.initialize_A_from_eigen(K_mm, tgt=True)
    
    for epoch in tqdm(range(config.epochs), desc="Epochs"):
        epoch_loss = 0
        model.train()

        total_epoch_time = 0.0

        if config.kcache:
            loop_iterator = train_iterator
        elif posX_loader is not None:
            loop_iterator = zip(train_iterator, posX_loader)
        else:
            loop_iterator = train_iterator
        num_batches = len(train_iterator)

  
        for i, batch_data_tuple in enumerate(tqdm(loop_iterator, desc=f"Epoch {epoch+1}", leave=False, total=num_batches)):
           
            encoded_X, K_nm_D, K_bb = None, None, None
            if config.kcache:
                batch_indices = batch_data_tuple[0]
                
                K_A = kA_loader.get_batch(batch_indices).to(device)
                K_B = kB_loader.get_batch(batch_indices).to(device) if kB_loader else None
                batch_y = y[batch_indices].to(device)

            else:
                if posX_loader is not None:
                    (batch_X, batch_y), (batch_X_aug, _) = batch_data_tuple
                else:
                    batch_X, batch_X_aug, batch_y = batch_data_tuple

                batch_X, batch_X_aug, batch_y = batch_X.to(device), batch_X_aug.to(device), batch_y.to(device)

                if config.kernel_fn == 'torch_ntk':
                    K_A = compute_ntk(ntk_model, batch_X, C_var).to(config.device)
                    K_B = compute_ntk(ntk_model, batch_X_aug, C_var).to(device) if config.loss_f not in ['krr', 'kpca', 'XE'] else None
                    if config.loss_f == 'kpca':
                        K_bb = compute_ntk(ntk_model, batch_X, batch_X).to(config.device)
                else:
                    K_A = rbf_kernel_torch(batch_X, C_var, config.gamma)
                    K_B = rbf_kernel_torch(batch_X_aug, C_var, config.gamma)
                    if config.loss_f == 'kpca':
                        K_bb = rbf_kernel_torch(batch_X, batch_X, config.gamma)

            current_step = epoch * train_size + (i * config.batch_size)
            if current_step < num_warmup_steps:
                lr_scale = float(current_step + 1) / float(num_warmup_steps)
                for g in optimizer.param_groups:
                    g['lr'] = config.learning_rate * lr_scale

            if config.loss_f == 'ae':
                s_time = time.time()
                encoded_X = model.encode(K_A)
                encoded_C = model.encode(K_mm)
                K_nm_D = rbf_kernel_torch(encoded_X, encoded_C, config.gamma)
            elif config.loss_f == 'kpca':
                # K_bb = rbf_kernel_torch(X[i:i+config.batch_size], X[i:i+config.batch_size], config.gamma)
                K_bb = compute_ntk(ntk_model, batch_X, batch_X).to(config.device)

            s_time = time.time()
            optimizer.zero_grad()
            loss = model(
                train_size,
                K_A, K_B, K_mm,
                encoded_X=encoded_X if config.loss_f == 'ae' else None,
                original_X=batch_X if config.loss_f == 'ae' else None,
                K_nm_D=K_nm_D if config.loss_f == 'ae' else None,
                K_bb=K_bb if config.loss_f == 'kpca' else None,
                y_true=batch_y,
                config=config
            )

            loss.backward()
            if config.pcnr_type:
                preconditioner.step(K_mm=K_mm, K_A=K_A, K_B=K_B, config=config)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            e_time = time.time()

            batch_time = e_time - s_time
            total_epoch_time += batch_time

            loss_value = loss.item()
            epoch_loss += loss_value
            losses.append(loss_value)
            nystrom_times.append(e_time-s_time)

        scheduler.step()
        avg_loss = epoch_loss / (y.size(0) // config.batch_size)

        epoch_time.append(total_epoch_time)

        model.eval()
        with torch.no_grad():

            if config.kcache:
                if config.dataset == 'imagenet':
                    train_acc = epoch_eval_imgnet(model, kv_loader, y_valid, config)
                else:
                    train_acc = epoch_eval_cached(model, kv_loader, y_valid, config)
            else:
                train_acc = epoch_eval(model, C_var, config, valid_loader, ntk_model)
        # if epoch == 1: 
        #     threshold = 0.75
        #     if train_acc < threshold:
        #         print(f"Early rejection: Accuracy {train_acc} is below threshold {threshold}")
        #         wandb.run.tags = wandb.run.tags + ("rejected_run",)
        #         wandb.finish(exit_code=1) 
        #         return (None,) * 8

        wandb.log({
            "Epoch": epoch + 1,
            "Loss": avg_loss,
            "Train Accuracy": train_acc,
            "Learning Rate": scheduler.get_last_lr()[0]
        })

        if train_acc > best_acc:
            best_acc = train_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience = config.patience
        else:
            patience -= 1
            if patience == 0:
                wandb.log({"Early Stopping Epoch": epoch + 1})
                break   
    
    avg_time_nystrom = np.mean(nystrom_times)
    wandb.log({
    "Average Time per Epoch (Nystrom)": avg_time_nystrom
    })

    avg_time = np.mean(epoch_time) 
    wandb.log({
        "AvgTime/Epoch": avg_time
    })



    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state found. Using the last model state.")
    model.eval()
    with torch.no_grad():

        if config.kcache:
            if config.dataset == 'imagenet':
                test_acc, pc_acc, tst_reps, tst_labels, C_reps = eval_imgnet(model, K_mm, kv_loader, y_valid, kt_loader, y_test, y_C, config)
            else:
                test_acc, pc_acc, tst_reps, tst_labels, C_reps = eval_cached(model, K_mm, kv_loader, y_valid, kt_loader, y_test, y_C, config)
        else:
            test_acc, pc_acc, tst_reps, tst_labels, C_reps = eval(test_loader, model, C_var, config, valid_loader, ntk_model)
            kv_loader = None
            kt_loader = None
                
    wandb.log({"Test Accuracy (before FT)": test_acc})

    return model, pc_acc, tst_reps, tst_labels, C_reps, kv_loader, kt_loader, ntk_model