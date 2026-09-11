import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from scipy.stats import kendalltau
from sklearn.manifold import TSNE
import seaborn as sns
from NTK import compute_ntk
from collections import Counter
from tqdm.auto import tqdm
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

def _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model):
    if config.kcache:
        similarities = kt_loader.get_batch(range(test_idx, test_idx + 1)).to(config.device).squeeze()
    else:
        x_sample = X_test[test_idx].unsqueeze(0).to(config.device)
        similarities = compute_ntk(ntk_model, x_sample, C).squeeze()
    
    return similarities

def _denormalize_image(img_tensor, mean, std):
    device = img_tensor.device
    mean = mean.to(device)
    std = std.to(device)
    denorm_img = img_tensor * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
    denorm_img = torch.clamp(denorm_img, 0, 1) 
    
    return denorm_img.cpu().numpy().transpose(1, 2, 0)

def log_majority_influence_cases_wandb(model, X_test, y_test, kt_loader, C, y_C, config, ntk_model, 
                                       num_cases_to_show=10, stage_prefix=""):

    print("Analyzing for majority influence cases...")
    device = next(model.parameters()).device
    

    A = model.A.detach()
    global_influences = torch.norm(A, dim=1)

    majority_cases = []

    for test_idx in range(len(X_test)):
        x_sample = X_test[test_idx].unsqueeze(0).to(device)

        similarities = _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model)
        sample_influences = similarities * global_influences
        
        _, top_5_indices = torch.topk(sample_influences, 5)
        y_C = y_C.to(top_5_indices.device)
        top_5_labels = y_C[top_5_indices].cpu().numpy()

        label_counts = Counter(top_5_labels)
        most_common_label, count = label_counts.most_common(1)[0]
        
        if count >= 3:
            case_info = {
                "test_idx": test_idx,
                "true_label": y_test[test_idx].item(),
                "majority_class": most_common_label,
                "top_5_indices": top_5_indices
            }
            majority_cases.append(case_info)
        
        if len(majority_cases) >= num_cases_to_show:
            break
            
    if not majority_cases:
        print("No test cases found with a majority influence from a single class.")
        return

    columns = ["Test Sample", "Test Label", "Majority Influence Class"]
    for i in range(1, 6):
        columns.extend([f"Top {i} Landmark", f"Top {i} Label"])

    table = wandb.Table(columns=columns)

    for case in majority_cases:
        test_idx = case["test_idx"]
        true_label = case["true_label"]
        majority_class = case["majority_class"]
        top_5_indices = case["top_5_indices"]
        # test_img_np = X_test[test_idx].squeeze().cpu().numpy().transpose(1, 2, 0) 

        test_img_np = _denormalize_image(X_test[test_idx], torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.247, 0.243, 0.261]))

        row = [
            wandb.Image(test_img_np),
            f"Label: {true_label}",
            f"Class: {majority_class}"
        ]
        
        for landmark_idx_tensor in top_5_indices:
            landmark_idx = landmark_idx_tensor.item()
            landmark_tensor = C[landmark_idx]
            landmark_label = y_C[landmark_idx].item()
            # landmark_img_np = C[landmark_idx].squeeze().cpu().numpy().transpose(1, 2, 0)
            landmark_img_np = _denormalize_image(
                landmark_tensor, torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
            )
            row.extend([
                wandb.Image(landmark_img_np),
                f"Label: {landmark_label}"
            ])
        
        table.add_data(*row)

    log_key = f"{stage_prefix} Majority Influence Landmark Cases".strip()
    wandb.log({log_key: table})

def log_sample_specific_influences_wandb(model, y_test, pred_labels, y_C, config, stage_prefix="", num_samples_to_show=10,
                                          X_test=None, C=None, ntk_model=None,
                                          kt_loader=None):
    num_samples_to_show = len(X_test)
    device = next(model.parameters()).device
    if config.kcache and kt_loader is None:
        raise ValueError("A 'kt_loader' must be provided when config.kcache is True.")
    if not config.kcache and (X_test is None or C is None):
        raise ValueError("'X_test' and 'C' must be provided when config.kcache is False.")

    y_test_np = np.array(y_test)
    pred_labels_np = np.array(pred_labels)
    correct_indices = np.where(y_test_np == pred_labels_np)[0]

    if len(correct_indices) == 0:
        print("No correctly predicted samples to analyze.")
        return

    samples_to_analyze_indices = correct_indices[:num_samples_to_show]
    A = model.A.detach()
    global_influences = torch.norm(A, dim=1)


    can_show_images = not config.kcache and (X_test is not None and C is not None and len(X_test.shape) > 2)


    columns = [ "Test Sample Index", "True/Pred Label"]
    if can_show_images:
        columns[0] = "Test Sample"
    
    for i in range(1, 6):
        landmark_col_name = f"Top {i} Landmark" if can_show_images else f"Top {i} Landmark Index"
        columns.extend([landmark_col_name, f"Top {i} Label"])

    influence_table = wandb.Table(columns=columns)

    for test_idx in samples_to_analyze_indices:
        true_label = y_test_np[test_idx]
        similarities = _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model)
        
        sample_influences = similarities * global_influences
        _, top_5_indices = torch.topk(sample_influences, 5)
        
        row_data = []
        if can_show_images:
            sample_img_data = X_test[test_idx].squeeze().cpu().numpy()
            row_data.append(wandb.Image(sample_img_data))
        else:
            row_data.append(test_idx)
        row_data.append(f"Label: {true_label}")

        for landmark_idx_tensor in top_5_indices:
            landmark_idx = landmark_idx_tensor.item()
            landmark_label = y_C[landmark_idx].item()
            
            if can_show_images:
                landmark_img_data = C[landmark_idx].squeeze().cpu().numpy()
                row_data.append(wandb.Image(landmark_img_data))
            else:
                row_data.append(landmark_idx)
            
            row_data.append(f"Label: {landmark_label}")
        
        influence_table.add_data(*row_data)

    log_key = f"{stage_prefix} Top 5 Influential Landmarks per Sample".strip()
    wandb.log({log_key: influence_table})

def log_landmark_class_distribution_wandb(y_C2, A, top_k=10, class_names=None, stage_prefix=""):
    
    influence_scores = torch.norm(A.detach(), dim=1)
    sorted_indices = torch.argsort(influence_scores, descending=True)

    top_k_labels = y_C2[sorted_indices[:top_k]].cpu().numpy()
    all_labels = y_C2.cpu().numpy()

    all_counts = Counter(all_labels)
    top_k_counts = Counter(top_k_labels)

    classes = sorted(set(all_labels))
    all_freqs = [all_counts.get(cls, 0) for cls in classes]
    top_k_freqs = [top_k_counts.get(cls, 0) for cls in classes]

    if class_names is None:
        class_names = [str(cls) for cls in classes]
    else:
        class_names = [class_names[cls] for cls in classes]

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, all_freqs, width, label='All Landmarks')
    ax.bar(x + width/2, top_k_freqs, width, label=f'Top {top_k} Landmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_xlabel("Class")
    ax.set_ylabel("Frequency")

    title = f"{stage_prefix} Class Distribution: All Landmarks vs. Top-K".strip()
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    log_key = f"{stage_prefix} Landmark Class Distribution".strip()
    wandb.log({log_key: wandb.Image(fig)})
    plt.close(fig)

def log_per_class_accuracy_wandb(per_class_acc, class_names=None, title="Per-Class Accuracy", stage_prefix=""):
    classes = sorted(per_class_acc.keys())
    accuracies = [per_class_acc[cls] for cls in classes]

    if class_names is None:
        class_labels = [str(cls) for cls in classes]
    else:
        class_labels = [class_names[cls] for cls in classes]

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, accuracies, color='skyblue')

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")

    full_title = f"{stage_prefix} {title}".strip()
    ax.set_title(full_title)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    wandb.log({full_title: wandb.Image(fig)})
    plt.close(fig)

def compute_kendall_tau(y_landmarks, A, per_class_accuracy):

    norms = torch.norm(A.detach(), dim=1)
    y_np = y_landmarks.cpu().numpy()

    class_norms = {}
    for cls in np.unique(y_np):
        cls_norms = norms[y_landmarks == cls]
        class_norms[cls] = cls_norms.mean().item()
    

    sorted_by_influence = sorted(class_norms.items(), key=lambda x: x[1], reverse=True)
    class_influence_ranks = {cls: rank for rank, (cls, _) in enumerate(sorted_by_influence)}

    classes = sorted(per_class_accuracy.keys())
    x_ranks = [class_influence_ranks[cls] for cls in classes]
    y_scores = [per_class_accuracy[cls] for cls in classes]

    tau, _ = kendalltau(x_ranks, y_scores)
    return tau

def log_tsne_representation_wandb(test_reps, test_labels,
                                  landmark_reps, landmark_labels, A,
                                  top_k_indices, class_names=None,
                                  title="t-SNE of Test Representations and Top Landmarks",
                                  stage_prefix=""):

    test_labels = np.vstack(test_labels)
    landmark_labels = landmark_labels.cpu().numpy()
    top_k_indices = np.array(top_k_indices)

    top_k_landmark_reps = landmark_reps[top_k_indices]
    top_k_landmark_labels = landmark_labels[top_k_indices]
    num_classes = len(np.unique(test_labels))

    combined_reps = np.vstack([test_reps, top_k_landmark_reps])

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(combined_reps)

    num_test_points = len(test_reps)
    tsne_test = tsne_results[:num_test_points]
    tsne_landmarks = tsne_results[num_test_points:]

    fig, ax = plt.subplots(figsize=(12, 8))
    palette = sns.color_palette("hsv", num_classes)

    for i in range(num_classes):
        indices = np.where(test_labels == i)
        ax.scatter(tsne_test[indices, 0], tsne_test[indices, 1],
                   color=palette[i],
                   label=class_names[i] if class_names else str(i),
                   alpha=0.4, s=20)

    for i in range(num_classes):
        indices = np.where(top_k_landmark_labels == i)
        ax.scatter(tsne_landmarks[indices, 0], tsne_landmarks[indices, 1],
                   color=palette[i],
                   marker='*',  
                   edgecolor='black',
                   s=250) 

    full_title = f"{stage_prefix} {title}".strip()
    ax.set_title(full_title)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Classes")
    

    plt.tight_layout()
    wandb.log({full_title: wandb.Image(fig)})
    plt.close(fig)

def representer_point_interpretabiliy(model, y_C, per_class_acc, test_reps, test_labels, C_reps, stage_prefix=""):

    model_param = model.A.detach().clone()
    influence_scores = torch.norm(model_param, dim=1) 
    sorted_indices = torch.argsort(influence_scores, descending=True)

    seen_classes = set()
    total_classes = set(y_C.cpu().numpy())
    for k, idx in enumerate(sorted_indices):
        cls = y_C[idx].item()
        seen_classes.add(cls)
        if seen_classes == total_classes:
            top_k = k + 1 
            break

    y_C = y_C.to(sorted_indices.device)
    num_classes = len(np.unique(test_labels))
    top_landmark_labels = y_C[sorted_indices[:num_classes]]
    top_influences = influence_scores[sorted_indices[:top_k]]

    wandb.log({
        "Top LM label": top_landmark_labels
    })
    num_classes = len(np.unique(test_labels))
    log_landmark_class_distribution_wandb(
    y_C, model_param, top_k=top_k, stage_prefix=stage_prefix
)


    seen_topk_classes = set()
    top_k_indices = []
    for i, idx in enumerate(sorted_indices[:top_k]):
        cls = y_C[idx].item()
        if cls not in seen_topk_classes:
            seen_topk_classes.add(cls)
            top_k_indices.append(idx.item())
        else:
            continue

    log_tsne_representation_wandb(
        test_reps=test_reps,
        test_labels=test_labels,
        landmark_reps=C_reps,
        landmark_labels=y_C,
        A=model_param,
        top_k_indices=top_k_indices,
        stage_prefix=stage_prefix
    )

def log_spectrum_plot_wandb(model, stage_prefix=""):

    A = model.A.detach().cpu()
    AtA = A.T @ A

    eigenvalues = torch.linalg.eigh(AtA).eigenvalues
    sorted_eigenvalues = torch.flip(eigenvalues, dims=[0])

    fig, ax = plt.subplots(figsize=(10, 6))
    ranks = np.arange(1, len(sorted_eigenvalues) + 1)
    ax.plot(ranks, sorted_eigenvalues.numpy(), marker='o', linestyle='-', color='b')

    ax.set_yscale('log') 
    ax.set_title(f"{stage_prefix} Eigenvalue Spectrum".strip(), fontsize=16)
    ax.set_xlabel("Rank (Eigenvalue Index)", fontsize=12)
    ax.set_ylabel("Eigenvalue (Log Scale)", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    log_key = f"{stage_prefix} Spectrum".strip()
    wandb.log({log_key: wandb.Image(fig)})
    plt.close(fig)

    table_data = [[rank, eig] for rank, eig in zip(ranks, sorted_eigenvalues)]
    spectrum_table = wandb.Table(data=table_data, columns=["Rank", "Eigenvalue"])
    log_key_table = f"{stage_prefix} Spectrum Data".strip()
    wandb.log({log_key_table: spectrum_table})

# Quantitative Interpretability Metrics

def log_label_consistency_wandb(model, y_test, pred_labels, y_C, config, 
                                X_test=None, C=None, ntk_model=None, kt_loader=None,
                                k_values=[1, 5, 10, 20, 50], stage_prefix=""):

    print(f"Calculating Label Consistency (Precision@K) for K={k_values}...")
    
    device = next(model.parameters()).device
    
    if isinstance(y_test, list): y_test = np.array(y_test)
    if isinstance(pred_labels, list): pred_labels = np.array(pred_labels)
    
    if len(y_test) != len(pred_labels):
        print(f"Warning: Length mismatch y_test ({len(y_test)}) vs pred_labels ({len(pred_labels)}). Truncating to minimum.")
        min_len = min(len(y_test), len(pred_labels))
        y_test = y_test[:min_len]
        pred_labels = pred_labels[:min_len]

    correct_indices = np.where(y_test == pred_labels)[0]
    num_correct = len(correct_indices)

    if num_correct == 0:
        print("No correctly predicted samples found. Skipping Label Consistency.")
        return

    A = model.A.detach()
    global_influences = torch.norm(A, dim=1) 
    
    precision_sums = {k: 0.0 for k in k_values}
    
    for test_idx in tqdm(correct_indices, desc="Consistency Eval", leave=False):
        
        similarities = _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model)
        sample_influences = similarities * global_influences
        
        max_k = max(k_values)
        _, top_indices = torch.topk(sample_influences, max_k)
        top_indices = top_indices.to(y_C.device)
        top_landmark_labels = y_C[top_indices].cpu().numpy()
        true_label = y_test[test_idx]
        
        for k in k_values:
            k_labels = top_landmark_labels[:k]
    
            matches = np.sum(k_labels == true_label)
            precision = matches / k
            precision_sums[k] += precision

 
    avg_precisions = {k: precision_sums[k] / num_correct for k in k_values}
    
   
    print(f"\n--- Label Consistency (Correct Predictions Only: N={num_correct}) ---")
    for k, p in avg_precisions.items():
        print(f"Precision@{k}: {p:.4f}")

    data = [[f"Top-{k}", score] for k, score in avg_precisions.items()]
    table = wandb.Table(data=data, columns=["Top-K", "Precision"])
    
    bar_plot = wandb.plot.bar(table, "Top-K", "Precision", 
                              title=f"{stage_prefix} Landmark Label Consistency (Precision@K)")
    
    wandb.log({f"{stage_prefix} Label Consistency Metric": bar_plot})
    
    return avg_precisions

def log_hit_rate_wandb(model, y_test, pred_labels, y_C, config, 
                       X_test=None, C=None, ntk_model=None, kt_loader=None,
                       k_values=[1, 2, 3, 5, 10], stage_prefix=""):

    print(f"Calculating Hit Rate (At Least One Match) for K={k_values}...")
    
    device = next(model.parameters()).device
    
    if isinstance(y_test, list): y_test = np.array(y_test)
    if isinstance(pred_labels, list): pred_labels = np.array(pred_labels)
    
    if len(y_test) != len(pred_labels):
        min_len = min(len(y_test), len(pred_labels))
        y_test = y_test[:min_len]
        pred_labels = pred_labels[:min_len]

    correct_indices = np.where(y_test == pred_labels)[0]
    num_correct = len(correct_indices)

    if num_correct == 0:
        return {}

    A = model.A.detach()
    global_influences = torch.norm(A, dim=1)
    
    hit_sums = {k: 0.0 for k in k_values}
    
    for test_idx in tqdm(correct_indices, desc="Hit Rate Eval", leave=False):
        
        similarities = _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model)
        sample_influences = similarities * global_influences
        
        max_k = max(k_values)
        _, top_indices = torch.topk(sample_influences, max_k)
        
        top_indices = top_indices.to(y_C.device)
        top_landmark_labels = y_C[top_indices].cpu().numpy()
        true_label = y_test[test_idx]
        
        for k in k_values:
            k_labels = top_landmark_labels[:k]
            if np.any(k_labels == true_label):
                hit_sums[k] += 1.0

    avg_hits = {k: hit_sums[k] / num_correct for k in k_values}
    
    print(f"\n--- Hit Rate (Correct Predictions Only: N={num_correct}) ---")
    for k, p in avg_hits.items():
        print(f"HitRate@{k}: {p:.4f}")

    data = [[f"Top-{k}", score] for k, score in avg_hits.items()]
    table = wandb.Table(data=data, columns=["Top-K", "Hit Rate"])
    
    bar_plot = wandb.plot.bar(table, "Top-K", "Hit Rate", 
                              title=f"{stage_prefix} Landmark Hit Rate (At Least One Match)")
    
    wandb.log({f"{stage_prefix} Hit Rate Metric": bar_plot})
    
    return avg_hits

def log_majority_consistency_wandb(model, y_test, pred_labels, y_C, config, 
                                   X_test=None, C=None, ntk_model=None, kt_loader=None,
                                   k_values=[1, 5, 10, 20, 50], stage_prefix=""):

    print(f"Calculating Majority Consistency (Count >= k/2) for K={k_values}...")
    
    device = next(model.parameters()).device
    
    if isinstance(y_test, list): y_test = np.array(y_test)
    if isinstance(pred_labels, list): pred_labels = np.array(pred_labels)
    
    if len(y_test) != len(pred_labels):
        min_len = min(len(y_test), len(pred_labels))
        y_test = y_test[:min_len]
        pred_labels = pred_labels[:min_len]

    correct_indices = np.where(y_test == pred_labels)[0]
    num_correct = len(correct_indices)

    if num_correct == 0:
        return {}

    A = model.A.detach()
    global_influences = torch.norm(A, dim=1)
    
    majority_sums = {k: 0.0 for k in k_values}
    
    for test_idx in tqdm(correct_indices, desc="Majority Eval", leave=False):
        
        similarities = _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model)
        sample_influences = similarities * global_influences
        
        max_k = max(k_values)
        _, top_indices = torch.topk(sample_influences, max_k)
        
        top_indices = top_indices.to(y_C.device)
        top_landmark_labels = y_C[top_indices].cpu().numpy()
        true_label = y_test[test_idx]
        
        for k in k_values:
            k_labels = top_landmark_labels[:k]
            match_count = np.sum(k_labels == true_label)
            if match_count >= (k / 2.0):
                majority_sums[k] += 1.0

    avg_majority = {k: majority_sums[k] / num_correct for k in k_values}
    
    print(f"\n--- Majority Consistency (Correct Predictions Only: N={num_correct}) ---")
    for k, p in avg_majority.items():
        print(f"Majority@{k}: {p:.4f}")

    data = [[f"Top-{k}", score] for k, score in avg_majority.items()]
    table = wandb.Table(data=data, columns=["Top-K", "Majority Consistency"])
    
    bar_plot = wandb.plot.bar(table, "Top-K", "Majority Consistency", 
                              title=f"{stage_prefix} Landmark Majority Consistency (>= k/2)")
    
    wandb.log({f"{stage_prefix} Majority Consistency Metric": bar_plot})
    
    return avg_majority

def log_feature_agreement_plot_wandb(model, X_test, C, feature_names, config, 
                                     kt_loader=None, ntk_model=None, 
                                     num_samples_to_avg=50, stage_prefix=""):

    print(f"Generating Feature Agreement Plot (Averaged over {num_samples_to_avg} samples)...")
    device = next(model.parameters()).device
    c_min, _ = torch.min(C, dim=0)
    c_max, _ = torch.max(C, dim=0)
    feature_ranges = torch.clamp(c_max - c_min, min=1e-6).to(device)
    
    num_features = C.shape[1]
    global_agreement_influential = torch.zeros(num_features, device=device)
    global_agreement_random = torch.zeros(num_features, device=device)
    
    test_indices = np.random.choice(len(X_test), min(len(X_test), num_samples_to_avg), replace=False)
    
    A = model.A.detach()
    global_influences = torch.norm(A, dim=1)

    for test_idx in tqdm(test_indices, desc="Feature Agreement"):
        
        similarities = _get_kernel_similarities(X_test, test_idx, C, kt_loader, config, ntk_model)
        sample_influences = similarities * global_influences
        _, top_k_indices = torch.topk(sample_influences, 5)
        
        random_indices = torch.randint(0, len(C), (5,), device=device)

        x_sample = X_test[test_idx].to(device)
        
        top_k_indices = top_k_indices.to(C.device)
        random_indices = random_indices.to(C.device)

        top_landmarks = C[top_k_indices].to(device)
        random_landmarks = C[random_indices].to(device)
        

        dist_influential = torch.abs(top_landmarks - x_sample.unsqueeze(0)) # [5, Feat]
        norm_dist_inf = dist_influential / feature_ranges.unsqueeze(0)

        agreement_inf = 1.0 - torch.clamp(norm_dist_inf, 0, 1)
        avg_agreement_inf = torch.mean(agreement_inf, dim=0) # Avg over the 5 landmarks
        
        dist_random = torch.abs(random_landmarks - x_sample.unsqueeze(0))
        norm_dist_rand = dist_random / feature_ranges.unsqueeze(0)
        agreement_rand = 1.0 - torch.clamp(norm_dist_rand, 0, 1)
        avg_agreement_rand = torch.mean(agreement_rand, dim=0)

        global_agreement_influential += avg_agreement_inf
        global_agreement_random += avg_agreement_rand


    global_agreement_influential /= len(test_indices)
    global_agreement_random /= len(test_indices)

    inf_vals_raw = global_agreement_influential.cpu().numpy()
    rand_vals_raw = global_agreement_random.cpu().numpy()


    cat_parents = ['workclass', 'education', 'marital-status', 'occupation', 
                   'relationship', 'race', 'gender', 'native-country']
    
    agg_map = {} 
    
    for i, raw_name in enumerate(feature_names):

        clean_name = raw_name.split('__')[-1] if '__' in raw_name else raw_name
        parent = None

        for p in cat_parents:
            if clean_name.startswith(p):
                parent = p
                break

        if parent is None:
            parent = clean_name
            
        if parent not in agg_map:
            agg_map[parent] = {'inf': [], 'rand': []}
            
        agg_map[parent]['inf'].append(inf_vals_raw[i])
        agg_map[parent]['rand'].append(rand_vals_raw[i])
        
    final_features = []
    final_inf = []
    final_rand = []
    
    for parent, vals in agg_map.items():
        final_features.append(parent)
        final_inf.append(np.mean(vals['inf']))
        final_rand.append(np.mean(vals['rand']))
        
    final_inf = np.array(final_inf)
    final_rand = np.array(final_rand)
    
    gaps = final_inf - final_rand
    sorted_idx = np.argsort(gaps)[::-1] 
    
    sorted_features = [final_features[i] for i in sorted_idx]
    sorted_inf = final_inf[sorted_idx]
    sorted_rand = final_rand[sorted_idx]
    
    x = np.arange(len(sorted_features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, sorted_inf, width, label='Influential Landmarks', color='royalblue')
    rects2 = ax.bar(x + width/2, sorted_rand, width, label='Random Baseline', color='lightgrey')
    
    ax.set_ylabel('Feature Agreement (Avg 0-1)')
    ax.set_title(f'{stage_prefix} Global Feature Importance (Aggregated)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_features, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    wandb.log({f"{stage_prefix} Feature Agreement Plot": wandb.Image(fig)})
    plt.close(fig)


def log_bias_audit_plot_wandb(Z_test, X_test, feature_names, stage_prefix=""):
    
    print(f"Generating Bias Audit Plot...")

    gender_idx = -1
    label_name = ""
    targets = ['gender_Male', 'gender_1', 'gender_0', 'gender']
    
    for target in targets:
        for i, raw_name in enumerate(feature_names):

            if target in raw_name:
                gender_idx = i
                label_name = raw_name
                break
        if gender_idx != -1:
            break
            
    if gender_idx == -1:
        print("CRITICAL ERROR: Could not find 'gender' column. Skipping Bias Audit.")
        return

    print(f"Found Sensitive Attribute '{label_name}' at index {gender_idx}")
    

    if isinstance(X_test, torch.Tensor):
        y_sensitive = X_test[:, gender_idx].cpu().numpy()
    else:
        y_sensitive = X_test[:, gender_idx]

  
    total_samples = len(Z_test)
    n_concept = min(int(total_samples * 0.5), 5000) 
    
    idx_train = np.arange(n_concept)
    idx_audit = np.arange(n_concept, total_samples)
    
    Z_train, Z_audit = Z_test[idx_train], Z_test[idx_audit]
    y_train, y_audit = y_sensitive[idx_train], y_sensitive[idx_audit]

    print(f"Training CAV on {n_concept} samples...")
    cav_clf = LinearSVC(class_weight='balanced', max_iter=5000, fit_intercept=True)
    cav_clf.fit(Z_train, y_train)
    
    v_raw = cav_clf.coef_[0]
    v_cav = v_raw / np.linalg.norm(v_raw)
    
    scores = Z_audit @ v_cav
    if len(np.unique(y_audit)) > 1:
        auc = roc_auc_score(y_audit, scores)
    else:
        auc = 0.5
        
    print(f"Latent Separability AUC: {auc:.4f}")

    if '1' in label_name or 'Male' in label_name:
        lbl_1 = "Male (1)"
        lbl_0 = "Female (0)"
    else:
        lbl_1 = "Female (1)"
        lbl_0 = "Male (0)"
    
    plot_df = pd.DataFrame({
        'Score': scores,
        'Gender': [lbl_1 if y == 1 else lbl_0 for y in y_audit]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=plot_df, x='Score', hue='Gender', kde=True, 
                 bins=40, palette=['orange', 'royalblue'], 
                 element="step", alpha=0.6, ax=ax)
    
    ax.set_title(f"Latent Space Projection (Bias Audit)\nAUC: {auc:.3f} (Separability)", fontsize=14)
    ax.set_xlabel(f"Projection on Gender Direction", fontsize=12)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    wandb.log({f"{stage_prefix} Bias Audit Histogram": wandb.Image(fig)})
    plt.close(fig)

def export_influence_to_csv(model, y_test, pred_labels, y_C, config, kt_loader, 
                            output_filename="influence_results.csv", 
                            top_k=5):

    print(f"Exporting Top-{top_k} Influential Landmarks to {output_filename}...")
    
    device = next(model.parameters()).device
    model.eval()
    y_test_np = np.array(y_test) if isinstance(y_test, list) else y_test
    pred_labels_np = np.array(pred_labels) if isinstance(pred_labels, list) else pred_labels

    correct_indices = np.where(y_test_np == pred_labels_np)[0]
    
    if len(correct_indices) == 0:
        print("No correctly predicted samples found.")
        return

    A = model.A.detach()
    global_influences = torch.norm(A, dim=1) 

    results = []
    for test_idx in tqdm(correct_indices, desc="Computing Influence"):
        try:
            similarities = kt_loader.get_batch(range(test_idx, test_idx + 1)).to(device).squeeze()
        except Exception as e:
            print(f"Error loading kernel for index {test_idx}: {e}")
            continue

        sample_influences = similarities * global_influences
        _, top_indices = torch.topk(sample_influences, top_k)
        top_indices = top_indices.cpu().numpy()
        current_landmark_labels = y_C[top_indices].cpu().numpy()
        
        row = {
            "test_idx": test_idx,
            "true_label": y_test_np[test_idx],
            "pred_label": pred_labels_np[test_idx],
        }
        
        for rank, (idx, label) in enumerate(zip(top_indices, current_landmark_labels)):
            row[f"landmark_idx_{rank+1}"] = idx
            row[f"landmark_label_{rank+1}"] = label
            
        results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    print(f"Successfully saved {len(df)} rows to {output_filename}")
    
    try:
        artifact = wandb.Artifact('influence_csv', type='dataset')
        artifact.add_file(output_filename)
        wandb.log_artifact(artifact)
        wandb.log({"Influence CSV Preview": wandb.Table(dataframe=df.head(50))})
    except Exception as e:
        print(f"WandB logging skipped: {e}")

    return df