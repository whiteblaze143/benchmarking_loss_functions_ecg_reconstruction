import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from data_process.data_augmentation import mnist_aug
from utils.landmark import kmeans_pp_landmarks, levs_landmarks_keops

class AugmentedDataset(Dataset):
 
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.std = 0.3

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample, label = self.original_dataset[idx]
        aug_view = mnist_aug(sample, self.std)
        return aug_view, label

def get_VD(cnf):
    if cnf.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif cnf.dataset == 'cifar10':

        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) # CIFAR-10 mean and std
    ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    elif cnf.dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2861], std=[0.3530])  # Fashion-MNIST mean and std
        ])
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_val_split_ratio = (40, 10)

    train_indices, valid_indices = train_test_split(
        np.arange(len(dataset)),
        train_size=train_val_split_ratio[0] / sum(train_val_split_ratio),
        stratify= np.array(dataset.targets),
        random_state=42
    )
    
    train_subset = Subset(dataset, train_indices)
    valid_subset = Subset(dataset, valid_indices)


    if cnf.dataset == 'mnist' or cnf.dataset == 'fashion_mnist':
        aug_train_dataset = AugmentedDataset(train_subset)

        num_samples = len(aug_train_dataset)
        sample_tensor, _ = aug_train_dataset[0]
        sample_shape = sample_tensor.shape
        sample_dtype = sample_tensor.dtype
        augmented_data_tensor = torch.empty((num_samples, *sample_shape), dtype=sample_dtype)

        for i in tqdm(range(num_samples)):
            augmented_data_tensor[i], _ = aug_train_dataset[i]
    
    subset_targets = np.array(dataset.targets)[train_indices]
    if cnf.landmark_mod == 'kmeans':

        C, idx_C = kmeans_pp_landmarks(
            train_subset.dataset.data[train_subset.indices].float() / 255.0,
            subset_targets,
            cnf.n_landmark
        )
        posC, idx_C2 = kmeans_pp_landmarks(
            augmented_data_tensor,
            subset_targets,
            cnf.n_landmark
        )

    elif cnf.landmark_mod == 'levs':
        C, idx_C = levs_landmarks_keops(
            train_subset.dataset.data[train_subset.indices].float() / 255.0, 
            subset_targets, 
            cnf.Ny_centers, 
            cnf.n_landmark, 
            cnf.gamma)

        posC, idx_C2 = levs_landmarks_keops(
            augmented_data_tensor,
            subset_targets,
            cnf.Ny_centers, 
            cnf.n_landmark, 
            cnf.gamma)
    
    # else:
    #     
    C2 = torch.cat((C.unsqueeze(1), posC), dim=0)

    subset_targets = np.array(dataset.targets)[train_indices]
    labels_orig = subset_targets[idx_C]
    labels_aug = subset_targets[idx_C2]
    y_C = torch.from_numpy(np.concatenate([labels_orig, labels_aug]))

    train_loader = DataLoader(train_subset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers, pin_memory=True)
    posX_loader = DataLoader(aug_train_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=cnf.batch_size, shuffle=True, num_workers=cnf.loader_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers, pin_memory=True)


    C = ridge_leverage_landmarks(X_train, cnf.Ny_centers, cnf.n_landmark, cnf.gamma)
    pos_C = ridge_leverage_landmarks(X_train_aug, cnf.Ny_centers, cnf.n_landmark, cnf.gamma)

    C, idx_C = kmeans_pp_landmarks(X_train, y_train, cnf.n_landmark)
    C, idx_C = levs_landmarks_falkon_style(
        X=X_train, 
        y=y_train,
        m=cnf.Ny_centers, 
        num_landmarks=cnf.n_landmark, 
        gamma=cnf.gamma,
        lam=1e-5,           
        sketch_dim=400       
    )
    C, idx_C = levs_landmarks_keops(X_train, y_train, cnf.Ny_centers, cnf.n_landmark, cnf.gamma)
    

        pos_C, idx_pos_C = kmeans_pp_landmarks(pos_X, y_train, cnf.n_landmark)

        pos_C, idx_pos_C = levs_landmarks_keops(pos_X, y_train, cnf.Ny_centers, cnf.n_landmark, cnf.gamma)
        C2 = torch.cat((C, pos_C), dim=0)
        idx2 = torch.cat([idx_C, idx_pos_C]) # when using levs_landmarks_keops
        idx2 = torch.cat((idx_C, idx_pos_C), dim=0)
        idx2 = idx_C + idx_pos_C  # when using kmeans_pp_landmarks
        y_C = y_train[idx2]
    
    return train_loader, posX_loader, valid_loader, test_loader, C2.to(cnf.device) , y_C