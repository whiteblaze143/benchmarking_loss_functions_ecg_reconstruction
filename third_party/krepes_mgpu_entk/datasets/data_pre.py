import torch
import numpy as np
import logging
import pathlib
from torch.utils.data import Subset, DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from datasets.data_augmentation import simclr_aug, mnist_aug
import os
from utility import kmeans_pp_landmarks, randlandmarks
from tqdm.auto import tqdm
from collections import defaultdict

class SubImageFolder(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = datasets.folder.default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


def create_imagenet_subset(dataset, num_classes=100, samples_per_class=1000, random_seed=42):
    print(f"Creating an ImageNet subset with {num_classes} classes and {samples_per_class} samples per class.")
    np.random.seed(random_seed)

    class_to_indices = defaultdict(list)
    for idx, target in enumerate(tqdm(dataset.targets, desc="Grouping samples by class")):
        class_to_indices[target].append(idx)

    all_class_ids = sorted(class_to_indices.keys())
    selected_class_ids = np.random.choice(all_class_ids, num_classes, replace=False)

    final_indices = []
    for class_id in tqdm(selected_class_ids, desc="Sampling from selected classes"):
        indices = class_to_indices[class_id]
        k = min(samples_per_class, len(indices))
        sampled_indices = np.random.choice(indices, k, replace=False)
        final_indices.extend(sampled_indices)


    samples = [dataset.samples[i] for i in final_indices]

    print(f"Created a final subset with {len(samples)} samples.")
    return SubImageFolder(samples, transform=dataset.transform)

def data_loader_to_tensor(data_loader, data_name, device, kernel_fn):
    """Loads all data from a DataLoader into a single pair of tensors on the specified device."""
    dataset_size = len(data_loader.dataset)

    first_image, _ = data_loader.dataset[0]
    if kernel_fn or data_name in ['cifar10', 'imagenet']:
        image_shape = first_image.shape
        X = torch.empty((dataset_size, *image_shape), device=device)
    else:
        image_size = first_image.numel()
        X = torch.empty((dataset_size, image_size), device=device)
    y = torch.empty((dataset_size,), dtype=torch.long, device=device)
    
    current_index = 0
    for images, labels in tqdm(data_loader, desc="Loading data subset into memory"):
        if not (kernel_fn or data_name in ['cifar10', 'imagenet']):
            images = images.view(images.size(0), -1)
        
        batch_end = current_index + images.size(0)
        X[current_index:batch_end] = images.to(device)
        y[current_index:batch_end] = labels.to(device)
        current_index = batch_end
    
    return X, y

class PairedAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_image, label = self.original_dataset[idx]
        augmented_image = self.transform(original_image)
        return original_image, augmented_image, label

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        augmented_image = self.transform(image)
        return augmented_image, label


def get_landmark_subset_loader(subset_dataset, cnf, num_samples_per_class=10):

    print("\nStep 1: Grouping subset indices by class for landmark selection...")
    
    class_to_indices = defaultdict(list)
    for subset_idx, original_idx in enumerate(tqdm(subset_dataset.indices, desc="Grouping Indices")):
        label = subset_dataset.dataset.targets[original_idx]
        class_to_indices[label].append(original_idx)

    print(f"Found {len(class_to_indices)} unique classes in the subset.")

    print(f"\nStep 2: Sampling {num_samples_per_class} indices from each class...")
    balanced_indices = []
    for class_id in tqdm(sorted(class_to_indices.keys()), desc="Sampling per Class"):
        indices = class_to_indices[class_id]
        k = min(num_samples_per_class, len(indices))
        sampled_indices = np.random.choice(indices, size=k, replace=False)
        balanced_indices.extend(sampled_indices)

    print(f"Total number of balanced indices selected: {len(balanced_indices)}")

    print("\nStep 3: Creating the balanced subset for landmarks...")
    landmark_selection_subset = Subset(subset_dataset.dataset, balanced_indices)

    landmark_loader = DataLoader(
        landmark_selection_subset,
        batch_size=cnf.batch_size,
        shuffle=False,
        num_workers=cnf.loader_workers
    )
    print("Successfully created a balanced DataLoader for landmarks.")
    
    return landmark_loader


def get_VD(cnf, savedir):

    if cnf.dataset == 'imagenet':
        data_root = './data/CLS-LOC'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        full_train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)

        full_train_dataset.targets = [s[1] for s in full_train_dataset.samples]
        test_dataset.targets = [s[1] for s in test_dataset.samples]

        dataset = create_imagenet_subset(full_train_dataset, num_classes=100, samples_per_class=1000)
        del full_train_dataset
        import gc; gc.collect()

        targets = np.array(dataset.targets)
        test_indices = [i for i, trgt in enumerate(test_dataset.targets) if trgt in targets]
        test_dataset = Subset(test_dataset, test_indices)

    elif cnf.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {cnf.dataset}")


    train_val_split_ratio = (40, 10)

    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        train_size=(train_val_split_ratio[0] / sum(train_val_split_ratio)),
        stratify=targets,
        random_state=42
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    simclr_transforms = transforms.Compose([

        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    aug_train_dataset = AugmentedDataset(train_subset, simclr_transforms)
    train_loader = DataLoader(train_subset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    posX_loader = DataLoader(aug_train_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    
    if cnf.dataset == 'imagenet':
        C, idx_C = randlandmarks(
        dataset=train_subset, 
        num_landmarks=cnf.n_landmark, 
        device=cnf.device
    )

        C_aug, idx_C_aug = randlandmarks(
        dataset=aug_train_dataset, 
        num_landmarks=cnf.n_landmark, 
        device=cnf.device
    )


        subset_targets = targets[train_indices]
        labels_orig = subset_targets[idx_C.cpu().numpy()]
        labels_aug = subset_targets[idx_C_aug.cpu().numpy()]
        y_C = torch.from_numpy(np.concatenate([labels_orig, labels_aug]))
        C2 = torch.cat((C, C_aug), dim=0)

    
    valid_loader = DataLoader(val_subset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    test_loader = DataLoader(test_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    
    landmark_dataset = TensorDataset(C2)
    final_landmark_loader = DataLoader(landmark_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)

    y_valid = torch.from_numpy(targets[val_indices])
    test_labels = [label for _, label in tqdm(test_dataset)]
    y_test = torch.tensor(test_labels)

    metadata = {
        "landmarks": C2, 
        "y_landmarks": y_C,
        "idx_C": idx_C, 
        "idx_pos_C": idx_C_aug,
        "y_valid": y_valid,
        "y_test": y_test
    }

    logging.info("Saving metadata file...")
    metadata_path = pathlib.Path(savedir) / f"{cnf.dataset}_{cnf.model}_metadata.pt"
    torch.save(metadata, metadata_path)
    logging.info(f"Saved metadata to {metadata_path}")

    del metadata
    del C2
    del y_valid
    del y_test
    del y_C
    torch.cuda.empty_cache()

    loaders = {
        "train": train_loader, "valid": valid_loader, "test": test_loader,
        "aug": posX_loader, "landmark": final_landmark_loader
    }

    return loaders