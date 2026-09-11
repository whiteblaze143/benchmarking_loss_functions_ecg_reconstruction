import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm import tqdm
from utility import kmeans_pp_landmarks, randlandmarks
import logging
import pathlib
import time
from scipy.io import arff

# ======================================================================== ADULT PREPROCESS ======================================================
def adult_preprocess(train_path, test_path):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    X_train = train_df.drop(columns='income')
    y_train = train_df['income']

    X_test = test_df.drop(columns='income')
    y_test = test_df['income']

    print("before preproc:", X_train.shape)
    print("bef", y_train.shape)

    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    mixed_columns = ['capital-loss', 'capital-gain']
    general_columns = ['age']
    integer_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  #  dense output
    ])
    
    mixed_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, general_columns + integer_columns),
            ('cat', categorical_transformer, categorical_columns),
            ('mixed', mixed_transformer, mixed_columns)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    
    train_data = pd.DataFrame(X_train_processed, columns=feature_names)
    train_data['target'] = y_train

    test_data = pd.DataFrame(X_test_processed, columns=feature_names)
    test_data['target'] = y_test.values
    
    return train_data, test_data
# ======================================================================== COVERTYPE PREPROCESS ======================================================
def covtype_preprocess(train_path, test_path):
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    col_names = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ] + [f'Wilderness_Area_{i}' for i in range(1, 5)] \
      + [f'Soil_Type_{i}' for i in range(1, 41)] \
      + ['Cover_Type']
    train_df.columns = col_names
    test_df.columns = col_names

    X_train = train_df.drop(columns='Cover_Type')
    y_train = train_df['Cover_Type']
    y_train = y_train - 1 

    X_test = test_df.drop(columns='Cover_Type')
    y_test = test_df['Cover_Type']
    y_test = y_test - 1 

    preprocessor = get_covtype_preprocessor(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    train_data = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    train_data['target'] = y_train

    test_data = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    test_data['target'] = y_test

    return train_data, test_data

def get_covtype_preprocessor(X_train):
    
    numerical_cols = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points'
    ]
    
    binary_cols = [col for col in X_train.columns if col not in numerical_cols]
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', 'passthrough', binary_cols)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X_train)    
    return preprocessor

# ======================================================================== HIGGS PREPROCESS ======================================================
def higgs_preprocess(file_path):

    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    df.dropna(inplace=True)

    if df['target'].iloc[0] and isinstance(df['target'].iloc[0], bytes):
        print("Detected 'bytes' in target column. Decoding...")
        df['target'] = df['target'].str.decode('utf-8').astype(float)

    X = df.drop('target', axis=1)
    y = df['target'].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train_val)

    X_train_scaled = scaler.transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    train_data = pd.DataFrame(X_train_scaled, columns=X_train_val.columns)
    train_data['target'] = y_train_val.values

    test_data = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_data['target'] = y_test.values

    return train_data, test_data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ $ AUGMENTATION $ ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def add_gaussian_noise_tensor(features, noise_level=0.05):
    noise = torch.normal(0, noise_level, size=features.size(), dtype=features.dtype, device=features.device)
    return features + noise

def random_feature_dropout_tensor(features, drop_prob=0.1):
    if drop_prob <= 0:
        return features
    mask = torch.rand_like(features) > drop_prob
    return features * mask.float()

def augment_tensor(sample, noise_level=0.05, drop_prob=0.1):

    augmented_sample = add_gaussian_noise_tensor(sample, noise_level)
    augmented_sample = random_feature_dropout_tensor(augmented_sample, drop_prob)
    return augmented_sample

class TabularDataset(Dataset):

    def __init__(self, data):
        self.features = torch.tensor(data.iloc[:, :-1].values, dtype=torch.float32)
        self.targets = torch.tensor(data.iloc[:, -1].values, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class AugmentedTabularDataset(Dataset):
 
    def __init__(self, original_dataset, noise_level=0.25, drop_prob=0.2):
        self.original_dataset = original_dataset
        self.noise_level = noise_level
        self.drop_prob = drop_prob

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        sample, label = self.original_dataset[idx]
        aug_view = augment_tensor(sample, noise_level=self.noise_level, drop_prob=self.drop_prob)
        return aug_view, label
# ________________________________________________________________ *** MAIN DATA LOADING *** ____________________________________________________________________________

def get_TD(cnf, savedir):
    device = cnf.device

    if cnf.dataset == "adult":
        train_data, test_data = adult_preprocess(cnf.train_path, cnf.test_path)
    elif cnf.dataset == "covtype":
        train_data, test_data = covtype_preprocess(cnf.train_path, cnf.test_path)
    elif cnf.dataset == "higgs":
        train_data, test_data = higgs_preprocess(cnf.train_path)
    else:
        raise ValueError(f"Tabular dataset {cnf.dataset} not implemented.")

    full_train_dataset = TabularDataset(train_data)
    test_dataset = TabularDataset(test_data)

    train_val_split_ratio = (80, 20)
    
    train_indices, val_indices = train_test_split(
        np.arange(len(full_train_dataset)),
        train_size=(train_val_split_ratio[0] / sum(train_val_split_ratio)),
        stratify=full_train_dataset.targets.numpy(),
        random_state=42
    )

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_train_dataset, val_indices)

    C, idx_C = kmeans_pp_landmarks(
        full_train_dataset.features[train_indices], 
        full_train_dataset.targets[train_indices], 
        cnf.n_landmark, device=cnf.device)

    y_C = full_train_dataset.targets[train_indices][idx_C]


    X_aug_for_landmarks = augment_tensor(full_train_dataset.features[train_indices])

    pos_C, idx_pos_C = kmeans_pp_landmarks(
        X_aug_for_landmarks,
        full_train_dataset.targets[train_indices],
        cnf.n_landmark,
        device=device
    )
    y_pos_C = full_train_dataset.targets[train_indices][idx_pos_C]
    del X_aug_for_landmarks
    torch.cuda.empty_cache()

    C2 = torch.cat((C, pos_C), dim=0)
    y_C2 = torch.cat((y_C, y_pos_C), dim=0)

    aug_train_dataset = AugmentedTabularDataset(train_subset)
    posX_loader = DataLoader(aug_train_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)

    train_loader = DataLoader(train_subset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    valid_loader = DataLoader(val_subset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    test_loader = DataLoader(test_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)
    landmark_dataset = TensorDataset(C2)
    final_landmark_loader = DataLoader(landmark_dataset, batch_size=cnf.batch_size, shuffle=False, num_workers=cnf.loader_workers)

    y_valid = full_train_dataset.targets[val_indices]
    y_test = test_dataset.targets

    metadata = {
        "y_train": full_train_dataset.targets[train_indices], 
        "y_valid": y_valid, 
        "y_test": y_test,
        "landmarks": C2, 
        "y_landmarks": y_C2, 
        "idx_C": idx_C, 
        "idx_pos_C": idx_pos_C,
    }
    timestamp = int(time.time())
    logging.info("Saving metadata file...")
    metadata_path = pathlib.Path(savedir) / f"{cnf.dataset}_{cnf.model}_metadata_{timestamp}.pt"
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