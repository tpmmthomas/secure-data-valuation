import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
import random
import numpy as np


def preprocess_features(X, preprocessor=None):
    # Define feature groups
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation',
        'relationship', 'race', 'sex', 'native-country'
    ]
    numeric_features = [
        'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
    ]
    # If no preprocessor is provided, create and fit one.
    if preprocessor is None:
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            # For scikit-learn < 1.2 use sparse=False; for >=1.2, you may use sparse_output=False.
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        X_preprocessed = preprocessor.fit_transform(X)
        return X_preprocessed, preprocessor
    else:
        X_preprocessed = preprocessor.transform(X)
        return X_preprocessed
    
def preprocess_targets(y):
    y_processed = np.where(y.to_numpy() == '>50K.', 1, 0)
    return y_processed.reshape(-1, )

def get_dataset(dataset):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        outdataset = torchvision.datasets.CIFAR10(root='./data', train=True,transform=transform,download=True)
    elif dataset == "mnist":
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])
        outdataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    elif dataset == "adult":
        adult = fetch_ucirepo(id=2) 
        X_preprocessed, _ = preprocess_features(adult.data.features)
        y_preprocessed = preprocess_targets(adult.data.targets)
        outdataset = [(x,y) for x,y in zip(X_preprocessed, y_preprocessed)]
    else:    
        raise ValueError("Unknown dataset")
    return outdataset


def split_dataset(data, train_num, test_num):
    train_data, test_data = train_test_split(data, train_size=train_num, test_size=test_num, random_state=42)
    return train_data, test_data

def add_noise(data, var=None):
    if var is None:
        var = random.random() * 2 + 0.1
    def _add_noise(data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        noise = torch.randn_like(data) * (var ** 0.5)
        return data + noise
    return [(_add_noise(x), y) for x, y in data]
    