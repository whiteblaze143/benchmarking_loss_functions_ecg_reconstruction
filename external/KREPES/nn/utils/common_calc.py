import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
import time
import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from liblinear.liblinearutil import train, predict
from collections import Counter


def epoch_eval1(model, valid_loader, cnf):

    train_reps = []
    train_labels = []

    for i, (batch_X, _, batch_y) in enumerate(tqdm(valid_loader, desc=f"Validation Accuracy", leave=False)):
        batch_X = batch_X.to(cnf.device)
        batch_y = batch_y.to(cnf.device)

        batch_reps = model(batch_X)
        train_reps.append(batch_reps)
        train_labels.extend(batch_y.cpu().numpy())

    print("train_reps shape", train_reps[0].shape)
    train_reps = torch.cat(train_reps, dim=0)
    train_reps = F.normalize(train_reps, dim=1)
    train_reps = train_reps.detach().cpu().numpy()

    y_train = np.array(train_labels).astype(int)

    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    weight_options = " ".join([f"-w{cls} {weight}" for cls, weight in class_weights.items()])
    print("Training classifier...")
    model = train(y_train.tolist(), train_reps.tolist(), f"-s 2 -c 1 {weight_options}")
    pred_labels, accuracy, _ = predict(y_train.tolist(), train_reps.tolist(), model)

    accuracy_train = balanced_accuracy_score(y_train, pred_labels)

    return accuracy_train

def eval1(model, valid_loader, test_loader, config):

    train_reps = []
    train_labels = []

    test_reps = []
    test_labels = []

    inf_time = []

    for i, (batch_X, _, batch_y) in enumerate(tqdm(valid_loader, desc=f"Validation Accuracy", leave=False)):
        batch_X = batch_X.to(config.device)
        batch_y = batch_y.to(config.device)
        batch_reps = model(batch_X)

        train_reps.append(batch_reps)      
        train_labels.extend(batch_y.cpu().numpy())
    
    for i, (batch_X, _, batch_y) in enumerate(tqdm(test_loader, desc=f"Test Accuracy", leave=False)):
        batch_X = batch_X.to(config.device)
        batch_y = batch_y.to(config.device)

        s_time = time.time()
        batch_reps = model(batch_X)
        e_time = time.time()
        inf_time.append(e_time-s_time)
        test_reps.append(batch_reps)      
        test_labels.extend(batch_y.cpu().numpy())
    
    avg_inf_time = np.mean(inf_time)
    wandb.log({
    "AvgInfTime": avg_inf_time
    })

    train_reps = torch.cat(train_reps, dim=0)
    train_reps = F.normalize(train_reps, dim=1)
    train_reps = train_reps.detach().cpu().numpy()

    test_reps = torch.cat(test_reps, dim=0)
    test_reps = F.normalize(test_reps, dim=1)
    test_reps = test_reps.detach().cpu().numpy()
    
    y_train = np.array(train_labels).astype(int)
    y_test = np.array(test_labels).astype(int)

    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    weight_options = " ".join([f"-w{cls} {weight}" for cls, weight in class_weights.items()])

    print("Training classifier...")
    model = train(y_train.tolist(), train_reps.tolist(), f"-s 2 -c 1 {weight_options}")
    pred_labels, accuracy, _ = predict(y_test.tolist(), test_reps.tolist(), model)

    accuracy_test = balanced_accuracy_score(y_test, pred_labels)
    

    return accuracy_test

def generate_negative_samples(X):

    n = X.size(0)
    
    random_indices = torch.randint(0, n, (n,), device=X.device)
    
    while torch.any(random_indices == torch.arange(n, device=X.device)):
        random_indices = torch.randint(0, n, (n,), device=X.device)
    
    neg_X_random = X[random_indices]  
    
    return neg_X_random

# =============== Linear Probe for ImageNet ===================

def run_linear_probe_online(backbone, train_loader, test_loader, config, eval_on_train=False):
    device = config.device
    backbone.eval() 

    with torch.no_grad():
        dummy_batch = next(iter(train_loader))
        if len(dummy_batch) == 3: 
            x_dummy = dummy_batch[0]
        else:
            x_dummy = dummy_batch[0]
        
        x_dummy = x_dummy.to(device)
        feats = backbone(x_dummy)
        feats = feats.view(feats.size(0), -1)
        input_dim = feats.shape[1]

    num_classes = config.N_cls
    linear_probe = torch.nn.Linear(input_dim, num_classes).to(device)
    
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = 20

    print(f"Training Linear Probe (Online) for {epochs} epochs...")
    for epoch in range(epochs):
        linear_probe.train()
        
        for batch in train_loader:
            if len(batch) == 3: 
                batch_X, _, batch_y = batch 
            else:
                batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            with torch.no_grad():
                features = backbone(batch_X)
                features = features.view(features.size(0), -1)
                features = F.normalize(features, dim=1) 

    
            with torch.enable_grad():
                optimizer.zero_grad()
                logits = linear_probe(features)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
    
    eval_loader = train_loader if eval_on_train else test_loader
    prefix = "Train" if eval_on_train else "Test"
    
    return inference_linear_probe(backbone, linear_probe, eval_loader, device, prefix)


def inference_linear_probe(backbone, linear_probe, loader, device, prefix="Test"):
    backbone.eval()
    linear_probe.eval()
    
    all_preds = []
    all_labels = []
    inf_times = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{prefix} Inference", leave=False):
            if len(batch) == 3:
                batch_X, _, batch_y = batch
            else:
                batch_X, batch_y = batch

            batch_X = batch_X.to(device)
            
            # Timing Inference (Backbone only, to match your previous metric)
            s_time = time.time()
            features = backbone(batch_X)
            e_time = time.time()
            inf_times.append(e_time - s_time)

            features = features.view(features.size(0), -1)
            features = F.normalize(features, dim=1)
            
            logits = linear_probe(features)
            preds = torch.argmax(logits, dim=1).cpu()
            
            all_preds.append(preds)
            all_labels.append(batch_y)

    if prefix == "Test":
        avg_inf_time = np.mean(inf_times)
        wandb.log({"AvgInfTime": avg_inf_time})

    all_preds = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).cpu().numpy()
    
    acc = balanced_accuracy_score(y_true, all_preds)
    return acc


def epoch_eval(model, valid_loader, cnf):
    acc = run_linear_probe_online(
        backbone=model, 
        train_loader=valid_loader, 
        test_loader=valid_loader, 
        config=cnf, 
        eval_on_train=True
    )
    return acc


def eval(model, valid_loader, test_loader, config):
    acc = run_linear_probe_online(
        backbone=model, 
        train_loader=valid_loader, 
        test_loader=test_loader, 
        config=config, 
        eval_on_train=False
    )
    return acc