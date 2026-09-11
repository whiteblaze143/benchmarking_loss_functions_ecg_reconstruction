import torch
import numpy as np
import os
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from data_process.data_augmentation import mnist_aug

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        augmented_image = self.transform(image)
        return image, augmented_image, label

def get_VD(cnf):
    if cnf.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform) 
        test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    elif cnf.dataset == 'cifar10':

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        simclr_aug = transforms.Compose([
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    elif cnf.dataset == 'imagenet':
        data_root = './data/CLS-LOC'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
        test_data = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)

        simclr_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

        dataset.targets = [s[1] for s in dataset.samples]
        test_data.targets = [s[1] for s in test_data.samples]

    elif cnf.dataset == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2861], std=[0.3530]) 
        ])
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_val_split_ratio = (45, 5)
    
    
    if isinstance(dataset, Subset):
        targets = np.array([dataset.dataset.targets[i] for i in dataset.indices])
    else:
        targets = dataset.targets.numpy() if isinstance(dataset.targets, torch.Tensor) else np.array(dataset.targets)
    
    total_size = len(dataset)
    
    train_indices, val_indices  = train_test_split(
        np.arange(total_size),
        train_size=(train_val_split_ratio[0] / sum(train_val_split_ratio)),
        stratify=targets,
        random_state=42
    )
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    aug_train_dataset = AugmentedDataset(train_subset, simclr_transforms)
    posX_loader = DataLoader(aug_train_dataset, batch_size=cnf.batch_size, shuffle=False)
    valid_loader = DataLoader(val_subset, batch_size=cnf.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cnf.batch_size, shuffle=False)

    return posX_loader, valid_loader, test_loader