import torch
from torch.distributions import Categorical
from pykeops.torch import LazyTensor
from tqdm.auto import tqdm

def rbf_kernel_keops(X, Y, sigma=1.0):
    if LazyTensor is None:
        raise ImportError("KeOps is required for this function but not installed.")
    X_i = LazyTensor(X.unsqueeze(1).contiguous())
    Y_j = LazyTensor(Y.unsqueeze(0).contiguous())
    D_ij = ((X_i - Y_j) ** 2).sum(-1)
    
    return (-D_ij / (2 * sigma ** 2)).exp()

def approximate_leverage_scores_keops(X_subset, k=50, reg=1e-8, gamma=1.0):

    m, d = X_subset.shape
    device = X_subset.device
    dtype = X_subset.dtype
    K = rbf_kernel_keops(X_subset, X_subset, sigma=gamma)
    
    p = 10 
    Omega = torch.randn(m, k + p, device=device, dtype=dtype)
    Y = K @ Omega
    Q, _ = torch.linalg.qr(Y)
    K_Q = K @ Q
    B = Q.T @ K_Q
    _, U_B = torch.linalg.eigh(B)
    U_B_top_k = U_B[:, -(k):]
    U = Q @ U_B_top_k
    leverage_scores = (U ** 2).sum(dim=1)
    del K, Omega, Y, Q, K_Q, B, U_B, U
    torch.cuda.empty_cache()
    
    return leverage_scores

def levs_landmarks_keops(X, y, m, num_landmarks, gamma, k=50):
    n = X.shape[0]
    if X.ndim >= 3:
        flt_X = X.view(X.shape[0], -1)
    else:
        flt_X = X

    rand_idx = torch.randperm(n, device=X.device)[:m]
    X_subset = flt_X[rand_idx]
    X_subset_notflt = X[rand_idx]

    lev_scores = approximate_leverage_scores_keops(X_subset, k=k, gamma=gamma)
    lev_scores_prob = lev_scores / lev_scores.sum()
    
    sampled_subset_indices = Categorical(probs=lev_scores_prob).sample((num_landmarks,))
    landmarks = X_subset_notflt[sampled_subset_indices]
    original_indices = rand_idx[sampled_subset_indices]
    
    return landmarks, original_indices


# ****************************************************************************************************
#          KeOps
# ****************************************************************************************************

def kmeans_pp_landmarks(X, labels, num_landmarks, device='cuda'):

    if X.ndim >= 3:
        flt_X = X.view(X.shape[0], -1)
    else:
        flt_X = X

    flt_X = flt_X.to(device)

    indices = [torch.randint(0, flt_X.shape[0], (1,)).item()]
    last_landmark = flt_X[indices[0]].unsqueeze(0)

    min_dists_sq = torch.cdist(flt_X, last_landmark).squeeze()**2

    for _ in tqdm(range(1, num_landmarks), desc="Selecting K-Means++ Landmarks (Fast)"):
        probs = min_dists_sq / min_dists_sq.sum()
        next_idx = torch.multinomial(probs, 1).item()
        indices.append(next_idx)

        last_landmark = flt_X[next_idx].unsqueeze(0)
        dists_to_new_sq = torch.cdist(flt_X, last_landmark).squeeze()**2
        
        min_dists_sq = torch.minimum(min_dists_sq, dists_to_new_sq)

    return X[indices].to(device), indices


def get_random_landmarks(features_tensor, labels_tensor, num_landmarks, device='cuda'):
    total_size = features_tensor.shape[0]
    indices = torch.randperm(total_size)[:num_landmarks]
    landmarks_features = features_tensor[indices].to(device)
    landmarks_labels = labels_tensor[indices].to(device)
    
    return landmarks_features, landmarks_labels