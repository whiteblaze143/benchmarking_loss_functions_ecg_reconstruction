import torch
import numpy as np
from torch.utils.data import Subset, DataLoader, random_split, Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torchvision.datasets.folder import default_loader 
from data_process.data_augmentation import simclr_aug, mnist_aug
import os
from utils.landmark import get_random_landmarks, randlandmarks


def get_resnet_feature_extractor(model_name='resnet34', device='cuda'):

    if model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feature_dim = 512
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feature_dim = 2048 
    elif model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feature_dim = 512
    else:
        raise ValueError("Model name must be 'resnet34' or 'resnet50'")

    model.fc = nn.Identity()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    print(f"Output dimension: {feature_dim}")
    return model, feature_dim

class FeatureDataset(Dataset):

    def __init__(self, original_dataset, feature_extractor, device='cuda'):
        self.original_dataset = original_dataset
        self.feature_extractor = feature_extractor.to(device)
        self.device = device
        
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        image = image.unsqueeze(0).to(self.device)
  
        with torch.no_grad():
            feature = self.feature_extractor(image)
        return feature.squeeze(0).cpu(), label

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_subset, transform):

        self.original_subset = original_subset
        self.transform = transform

    def __len__(self):
        return len(self.original_subset)

    def __getitem__(self, idx):

        global_idx = self.original_subset.indices[idx]
        path, label = self.original_subset.dataset.samples[global_idx]
        image = default_loader(path)
        augmented_image = self.transform(image)
        
        return augmented_image, label

def get_VD(cnf):

    feature_path = './imgnet_features18'

    train_data = torch.load(os.path.join(feature_path, 'train_features.pt'))
    aug_data = torch.load(os.path.join(feature_path, 'aug_features.pt'))
    valid_data = torch.load(os.path.join(feature_path, 'valid_features.pt'))
    test_data = torch.load(os.path.join(feature_path, 'test_features.pt'))

    X_train, y_train = train_data['features'], train_data['labels']
    X_train_aug, y_train_aug = aug_data['features'], aug_data['labels']
    X_valid, y_valid = valid_data['features'], valid_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']

    C, y_C_orig = get_random_landmarks(X_train, y_train, cnf.n_landmark , cnf.device)
    C_aug, y_C_aug = get_random_landmarks(X_train_aug, y_train_aug, cnf.n_landmark , cnf.device)

    C2 = torch.cat((C, C_aug), dim=0)
    y_C = torch.cat((y_C_orig, y_C_aug), dim=0)

    train_dataset = TensorDataset(X_train, y_train)
    aug_train_dataset = TensorDataset(X_train_aug, y_train_aug)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cnf.batch_size, shuffle=True)
    posX_loader = DataLoader(aug_train_dataset, batch_size=cnf.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cnf.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cnf.batch_size, shuffle=False)
    
    return train_loader, posX_loader, valid_loader, test_loader, C2, y_C

def get_VD_rl_feat(cnf):

    if cnf.dataset == 'imagenet':
        data_root = '/data/CLS-LOC'
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'train'), transform=transform)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_root, 'val'), transform=transform)

        train_dataset.targets = [s[1] for s in train_dataset.samples]
        test_dataset.targets = [s[1] for s in test_dataset.samples]

        targets = np.array(train_dataset.targets)


    else:
        raise ValueError(f"Unknown dataset: {cnf.dataset}")

    train_val_split_ratio = (40, 10)

    train_indices, val_indices = train_test_split(
        np.arange(len(train_dataset)),
        train_size=(train_val_split_ratio[0] / sum(train_val_split_ratio)),
        stratify=targets,
        random_state=42
    )
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    simclr_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    aug_train_dataset = AugmentedDataset(train_subset, simclr_transforms)


    feature_extractor, feature_dim = get_resnet_feature_extractor('resnet34', cnf.device)
    feat_train_dataset = FeatureDataset(train_subset, feature_extractor, cnf.device)
    feat_aug_dataset = FeatureDataset(aug_train_dataset, feature_extractor, cnf.device)
    feat_valid_dataset = FeatureDataset(val_subset, feature_extractor, cnf.device)
    feat_test_dataset = FeatureDataset(test_dataset, feature_extractor, cnf.device)


    C, idx_C = randlandmarks(
    dataset=feat_train_dataset, 
    num_landmarks=cnf.n_landmark, 
    device=cnf.device
)

    C_aug, idx_C_aug = randlandmarks(
    dataset=feat_aug_dataset, 
    num_landmarks=cnf.n_landmark, 
    device=cnf.device
)

    subset_targets = targets[train_indices]
    labels_orig = subset_targets[idx_C.cpu().numpy()]
    labels_aug = subset_targets[idx_C_aug.cpu().numpy()]
    y_C = torch.from_numpy(np.concatenate([labels_orig, labels_aug]))
    C2 = torch.cat((C, C_aug), dim=0)

    
    train_loader = DataLoader(feat_train_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    posX_loader = DataLoader(feat_aug_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    valid_loader = DataLoader(feat_valid_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    test_loader = DataLoader(feat_test_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    
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

    del metadata
    del C2
    del y_valid
    del y_test
    del y_C
    torch.cuda.empty_cache()

    loaders = {
        "train": train_loader, "valid": valid_loader, "test": test_loader,
        "aug": posX_loader
    }

    return train_loader, posX_loader, valid_loader, test_loader, C2, y_C