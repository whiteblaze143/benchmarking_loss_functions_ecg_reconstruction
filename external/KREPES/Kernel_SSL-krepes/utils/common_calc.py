import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.interpretability import log_per_class_accuracy_wandb, \
    log_sample_specific_influences_wandb, \
    log_majority_influence_cases_wandb, \
    export_influence_to_csv
import wandb
from torch.optim.lr_scheduler import _LRScheduler

from liblinear.liblinearutil import train, predict, load_model, save_model
from collections import Counter
import time

from NTK import CNTK, compute_JAX_ntk
from NTK import compute_ntk, compute_ntk_vps
eps = 1e-12

from data_process.zca import ZCAWhitening

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=2.0, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute the current learning rate using the poly formula."""
        return [
            base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power
            for base_lr in self.base_lrs
        ]

def rbf_kernel_torch(X1, X2, gamma):

    X1_norm_sq = torch.sum(X1 ** 2, dim=-1)
    X2_norm_sq = torch.sum(X2 ** 2, dim=-1)

    dot_product = torch.matmul(X1, X2.T)
    dist_sq = X1_norm_sq.unsqueeze(1) + X2_norm_sq.unsqueeze(0) - 2 * dot_product
    dist_sq = torch.clamp(dist_sq, min=0.0)
    K = torch.exp(-gamma * dist_sq)
    return K

def euclidean(samples, centers, squared=True):

    samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers**2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)        
        distances.sqrt_()

    return distances

def laplacian(samples, centers, bandwidth, device):

    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=False)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()

    return kernel_mat.to(device)

def get_representation(model, C, batch_X, config, ntk_):
    if config.kernel_fn == 'torch_ntk':
        K_Cx = compute_ntk(ntk_, batch_X, C).detach()
    else:
        K_Cx = rbf_kernel_torch(batch_X, C, config.gamma)
    
    f_x = model.encoder(K_Cx)
    return f_x
    
def eval(test_loader, model, C, config, valid_loader, ntk):
    
    train_reps = []
    train_labels = []

    test_reps = []
    test_labels = []

    for i, (batch_X, batch_y) in enumerate(tqdm(valid_loader, desc=f"Validation Accuracy", leave=False)):
        batch_X = batch_X.to(config.device)
        batch_y = batch_y.to(config.device)

        batch_reps = get_representation(model, C, batch_X, config, ntk)
        if config.norm_mode == 'np':
            train_reps.append(batch_reps.detach().cpu().numpy())
        else:
            train_reps.append(batch_reps)      
        train_labels.extend(batch_y.cpu().numpy())

    for i, (batch_X, batch_y) in enumerate(tqdm(test_loader, desc=f"Test Accuracy", leave=False)):
        batch_X = batch_X.to(config.device)
        batch_y = batch_y.to(config.device)

        batch_reps = get_representation(model, C, batch_X, config, ntk)
        if config.norm_mode == 'np':
            test_reps.append(batch_reps.detach().cpu().numpy())
        else:
            test_reps.append(batch_reps)      
        test_labels.extend(batch_y.cpu().numpy())
    # --- NEW: Log t-SNE Plot ---
    # tst_reps = torch.cat(test_reps, dim=0)
    tst_labels = test_labels

    C_reps = []
    for i in tqdm(range(0, len(C), config.batch_size)):
        batch_C = C[i:i + config.batch_size]
        batch_reps = get_representation(model, C, batch_C, config, ntk)
        C_reps.append(batch_reps)      
    # --- END: Log t-SNE Plot ---
    if config.norm_mode == 'np':
        scaler = StandardScaler()
        train_reps = np.vstack(train_reps)
        train_reps = scaler.fit_transform(train_reps)

        test_reps = np.vstack(test_reps)
        test_reps = scaler.fit_transform(test_reps)
    else:
        train_reps = torch.cat(train_reps, dim=0)
        train_reps = F.normalize(train_reps, dim=1)
        train_reps = train_reps.detach().cpu().numpy()

        test_reps = torch.cat(test_reps, dim=0)
        test_reps = F.normalize(test_reps, dim=1)
        test_reps = test_reps.detach().cpu().numpy()

        C_reps = torch.cat(C_reps, dim=0)
        C_reps = F.normalize(C_reps, dim=1)
        C_reps = C_reps.detach().cpu().numpy()



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

    # Per-class accuracy
    pred_labels = np.array(pred_labels)
    unique_classes = np.unique(y_test)
    per_class_acc = {}
    for cls in unique_classes:
        idx = np.where(y_test == cls)[0]
        cls_correct = np.sum(pred_labels[idx] == cls)
        per_class_acc[cls] = cls_correct / len(idx)

    log_per_class_accuracy_wandb(per_class_acc, class_names = [str(i) for i in range(10)], stage_prefix="Pre-Training")
    

    return accuracy_test, per_class_acc, test_reps, tst_labels, C_reps

def eval_cached(model, K_mm, kv_loader, y_valid, kt_loader, y_test, y_C, config):

    device = next(model.parameters()).device
    model.eval()
    train_reps, test_reps, C_reps, y_valid_ = [], [], [], []
    inf_time = []

    # epoch_indices = torch.randperm(len(y_valid))

    with torch.no_grad():
        print("Processing validation data...")
        for i in tqdm(range(0, len(y_valid), config.batch_size)):
            batch_indices = range(i, min(i + config.batch_size, len(y_valid)))
            # batch_indices = epoch_indices[i : i + config.batch_size]
            batch_K = kv_loader.get_batch(batch_indices).to(device)
            batch_reps = model.encoder(batch_K)
            train_reps.append(batch_reps)
            # y_valid_.append(y_valid[batch_indices])
        print("Processing test data...")
        for i in tqdm(range(0, len(y_test), config.batch_size)):
            batch_indices = range(i, min(i + config.batch_size, len(y_test)))
            batch_K = kt_loader.get_batch(batch_indices).to(device)
            s_time = time.time()
            batch_reps = model.encoder(batch_K)
            e_time = time.time()
            test_reps.append(batch_reps)
            inf_time.append(e_time-s_time)
        print("Processing landmark data...")

        
        C_reps = model.encoder(K_mm)
    
    avg_inf_time = np.mean(inf_time)
    wandb.log({
    "AvgInfTime": avg_inf_time
    })
    
    train_reps = torch.cat(train_reps, dim=0)
    test_reps = torch.cat(test_reps, dim=0)
    # y_valid = torch.cat(y_valid_, dim=0)
    
    train_reps = F.normalize(train_reps, dim=1)
    test_reps = F.normalize(test_reps, dim=1)
    C_reps = F.normalize(C_reps, dim=1)

    train_reps = train_reps.detach().cpu().numpy()
    test_reps = test_reps.detach().cpu().numpy()
    C_reps = C_reps.detach().cpu().numpy()

    y_train = y_valid.cpu().numpy().astype(int)
    y_test = y_test.cpu().numpy().astype(int)

    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    weight_options = " ".join([f"-w{cls} {weight}" for cls, weight in class_weights.items()])

    print("Training final classifier...")
    classifier_model = train(y_train.tolist(), train_reps.tolist(), f"-s 2 -c 1 {weight_options}")
    pred_labels, _, _ = predict(y_test.tolist(), test_reps.tolist(), classifier_model)
    accuracy_test = balanced_accuracy_score(y_test, pred_labels)

    # Per-class accuracy
    pred_labels = np.array(pred_labels)
    unique_classes = np.unique(y_test)
    per_class_acc = {}
    for cls in unique_classes:
        idx = np.where(y_test == cls)[0]
        cls_correct = np.sum(pred_labels[idx] == cls)
        per_class_acc[cls] = cls_correct / len(idx)

    log_per_class_accuracy_wandb(per_class_acc, class_names = [str(i) for i in range(len(unique_classes))], stage_prefix="Pre-Training")

    # log_sample_specific_influences_wandb(
    #     model=model,
    #     y_test=y_test,
    #     pred_labels=pred_labels,
    #     y_C=y_C,          
    #     config=config,
    #     kt_loader=kt_loader, 
    #     stage_prefix="Final Evaluation (Cached)",
    #     X_test=X_test, C=C, ntk_model=ntk_model,
    # )

    # log_majority_influence_cases_wandb(
    #     model=model, 
    #     X_test=X_test,    
    #     y_test=y_test,
    #     kt_loader=kt_loader,     
    #     C=C,           
    #     y_C=y_C, 
    #     config=config,
    #     ntk_model=ntk_model,
    #     num_cases_to_show=10,
    #     stage_prefix="Final Evaluation"
    # )

    return accuracy_test, per_class_acc, test_reps, y_test, C_reps

def selectNystromCenters(X, lev_scores, M, n):
    if lev_scores is None or len(lev_scores) == 0:  # Uniform Nystrom
        D = np.eye(M)
        C = X[np.random.permutation(n)[:M], :]
    else:  # Approx. Lev. Scores Nystrom
        prob = lev_scores / np.sum(lev_scores)
        # prob = prob+ 1e-8
        count, ind = discrete_prob_sample(M, prob)
        D = np.diag(1. / np.sqrt(n * prob[ind] * count))
        C = X[ind, :]
    return C, D

def discrete_prob_sample(M, prob):
    bins = np.histogram(np.random.rand(M), bins=np.concatenate(([0], np.cumsum(prob))))[0]
    ind = np.where(bins > 0)[0]
    count = bins[ind]
    return count, ind

def epoch_eval(model, C, config, valid_loader, ntk):

    train_reps = []
    train_labels = []

    for i, (batch_X, batch_y) in enumerate(tqdm(valid_loader, desc=f"Validation Accuracy", leave=False)):
        batch_X = batch_X.to(config.device)
        batch_y = batch_y.to(config.device)

        batch_reps = get_representation(model, C, batch_X, config, ntk)
        if config.norm_mode == 'np':
            train_reps.append(batch_reps.detach().cpu().numpy())
        else:
            train_reps.append(batch_reps)
        train_labels.extend(batch_y.cpu().numpy())

    if config.norm_mode == 'np':

        train_reps = np.vstack(train_reps)
        scaler = StandardScaler()
        train_reps = scaler.fit_transform(train_reps)
    else:
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

def epoch_eval_cached(model, kv_loader, y_valid, config):

    device = next(model.parameters()).device
    model.eval()
    train_reps, y_valid_ = [], []

    # epoch_indices = torch.randperm(len(y_valid))
    
    with torch.no_grad():
        print("Processing validation data for classifier training...")
        for i in tqdm(range(0, len(y_valid), config.batch_size)):
            batch_indices = range(i, min(i + config.batch_size, len(y_valid)))
            # batch_indices = epoch_indices[i : i + config.batch_size]
            batch_K = kv_loader.get_batch(batch_indices).to(device)
            batch_reps = model.encoder(batch_K)
            train_reps.append(batch_reps)
            # y_valid_.append(y_valid[batch_indices])

    train_reps = torch.cat(train_reps, dim=0)
    # y_valid = torch.cat(y_valid_, dim=0)
 
    train_reps = F.normalize(train_reps, dim=1)
    train_reps = train_reps.detach().cpu().numpy()

    y_train = y_valid.cpu().numpy().astype(int)
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    weight_options = " ".join([f"-w{cls} {weight}" for cls, weight in class_weights.items()])
    
    print("Training classifier on validation representations...")
    classifier_model = train(y_train.tolist(), train_reps.tolist(), f"-s 2 -c 1 {weight_options}")
    pred_labels, _, _ = predict(y_train.tolist(), train_reps.tolist(), classifier_model)

    accuracy_train = balanced_accuracy_score(y_train, pred_labels)
    return accuracy_train
# =============== Linear Probe for ImageNet ===================

def train_and_evaluate_online(model, train_loader_obj, train_labels, test_loader_obj, test_labels, config, stage="Validation"):

    device = next(model.parameters()).device
    model.eval() 

    num_classes = len(torch.unique(train_labels))
    input_dim = model.k
    linear_probe = torch.nn.Linear(input_dim, num_classes).to(device)

    y_train_np = train_labels.cpu().numpy()
    class_counts = Counter(y_train_np)
    total_samples = len(y_train_np)
    weights = torch.ones(num_classes).to(device)
    for cls in range(num_classes):
        if cls in class_counts:
            weights[cls] = total_samples / (len(class_counts) * class_counts[cls])
            
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3, weight_decay=1e-6)

    epochs = 20
    batch_size = config.batch_size
    n_train = len(train_labels)

    print(f"[{stage}] Training Linear Probe (Online) for {epochs} epochs...")
    
    for epoch in range(epochs):
        linear_probe.train()
        indices = torch.randperm(n_train)
    
        for i in range(0, n_train, batch_size):
            batch_idxs = indices[i : min(i + batch_size, n_train)]
            batch_K = train_loader_obj.get_batch(batch_idxs).to(device)
            batch_y = train_labels[batch_idxs].to(device)

            with torch.no_grad():
                batch_features = model.encoder(batch_K)
                batch_features = F.normalize(batch_features, dim=1)

            with torch.enable_grad():
                optimizer.zero_grad()
                logits = linear_probe(batch_features)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    return run_inference_batchwise(model, linear_probe, test_loader_obj, test_labels, config)


def run_inference_batchwise(model, linear_probe, loader_obj, labels, config):

    device = next(model.parameters()).device
    model.eval()
    linear_probe.eval()
    
    all_preds = []
    n_samples = len(labels)
   
    with torch.no_grad():
        for i in range(0, n_samples, config.batch_size):
            batch_idxs = range(i, min(i + config.batch_size, n_samples))
            batch_K = loader_obj.get_batch(batch_idxs).to(device)
            batch_features = model.encoder(batch_K)
            batch_features = F.normalize(batch_features, dim=1)
            logits = linear_probe(batch_features)
            preds = torch.argmax(logits, dim=1).cpu()
            
            all_preds.append(preds)

    all_preds = torch.cat(all_preds).numpy()
    y_true = labels.cpu().numpy()
    
    acc = balanced_accuracy_score(y_true, all_preds)
    
    unique_classes = np.unique(y_true)
    per_class_acc = {}
    for cls in unique_classes:
        idx = np.where(y_true == cls)[0]
        cls_correct = np.sum(all_preds[idx] == cls)
        per_class_acc[cls] = cls_correct / len(idx)
        
    return acc, per_class_acc, all_preds

def epoch_eval_imgnet(model, kv_loader, y_valid, config):

    acc, _, _ = train_and_evaluate_online(
        model=model, 
        train_loader_obj=kv_loader, 
        train_labels=y_valid, 
        test_loader_obj=kv_loader,
        test_labels=y_valid, 
        config=config,
        stage="Epoch Eval"
    )
    return acc


def eval_imgnet(model, K_mm, kv_loader, y_valid, kt_loader, y_test, y_C, config):

    test_acc, per_class_acc, pred_labels = train_and_evaluate_online(
        model=model,
        train_loader_obj=kv_loader,
        train_labels=y_valid,
        test_loader_obj=kt_loader,
        test_labels=y_test,
        config=config,
        stage="Final Eval"
    )

    export_influence_to_csv(
        model=model,
        y_test=y_test.cpu().numpy(),
        pred_labels=pred_labels,
        y_C=y_C,
        config=config,
        kt_loader=kt_loader,
        output_filename="imagenet_influence.csv",
        top_k=5
    )

    with torch.no_grad():
        C_reps = model.encoder(K_mm)
        C_reps = F.normalize(C_reps, dim=1).cpu().numpy()
    
    return test_acc, per_class_acc, None, y_test.cpu().numpy(), C_reps