from torchvision import transforms
import torch
import torch.distributions as dist


def simclr_aug(X):

    simclr_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=X.shape[2:], scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5)
        # transforms.RandomApply([
        #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        # ], p=0.8), 
        # transforms.RandomGrayscale(p=0.2), 
        # transforms.RandomApply([
        #     transforms.GaussianBlur(kernel_size=23)
        # ], p=0.5),
        # transforms.RandomSolarize(threshold=128, p=0.1),
    ])

    augmented_X = torch.stack([
        transforms.ToTensor()(simclr_transforms(transforms.ToPILImage()(img)))
        for img in X
    ])
    return augmented_X.to(X.device)

def mnist_aug(X, cnf):
    pos_X = X.clone()
    if cnf.loss_f == 'spectral':
        noise_std = 0.1
    else:
        noise_std = 0.3
    noise = torch.normal(0, noise_std, pos_X.shape, device=X.device)
    pos_X += noise

    dropout_prob = 0.1  

    dropout_mask = torch.bernoulli(torch.full(pos_X.shape, 1 - dropout_prob, device=pos_X.device))
    pos_X *= dropout_mask

    return pos_X

class TabularContrastiveAugmenter:
    def __init__(self, pm=0.3, device='cuda'):
        self.pm = pm
        self.device = device
        self.empirical_distributions = None
        self.feature_value_counts = None
        self.feature_values = None
        
    def fit(self, dataframe):
        """Compute empirical distributions from a pandas DataFrame"""
        features = torch.tensor(dataframe.iloc[:, :-1].values, dtype=torch.float32)
        num_samples, num_features = features.shape
        
        self.feature_values = []
        self.feature_value_counts = []
        self.empirical_distributions = []
        
        for j in range(num_features):
            unique_values, counts = torch.unique(features[:, j], return_counts=True)
            self.feature_values.append(unique_values)
            self.feature_value_counts.append(counts.float())
            probs = counts.float() / counts.sum()
            self.empirical_distributions.append(dist.Categorical(probs=probs))
    
    def __call__(self, x):
   
        if self.empirical_distributions is None:
            raise ValueError("Call fit() first")
            
        if x.dim() == 1: 
            return self._augment_single(x)
        elif x.dim() == 2:  
            return torch.stack([self._augment_single(sample) for sample in x])
        else:
            raise ValueError("Input must be 1D (single sample) or 2D (batch)")
    
    def _augment_single(self, x):
        num_features = x.shape[0]
        mask = torch.bernoulli(torch.full((num_features,), self.pm)).to(self.device)
        
        replacement_values = torch.zeros_like(x)

        for j in range(num_features):
            if len(self.feature_values[j]) > 1:
                sampled_idx = self.empirical_distributions[j].sample()
                replacement_values[j] = self.feature_values[j][sampled_idx]
            else:
                replacement_values[j] = self.feature_values[j][0]
        
        return mask * replacement_values + (1 - mask) * x
    
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_data, augmenter):
        self.data = torch.tensor(original_data.iloc[:, :-1].values, dtype=torch.float32)
        self.targets = torch.tensor(original_data.iloc[:, -1].values, dtype=torch.float32)
        self.augmenter = augmenter
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        augmented_x = self.augmenter(x)
        return x, augmented_x, y