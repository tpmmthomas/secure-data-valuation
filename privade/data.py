"""
Data loading utilities for PrivaDE experiments.
Supports MNIST, CIFAR-10, CIFAR-100, and ImageNet datasets.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, Dict, Any
import warnings

# Default data directory - always relative to the privade package location
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_CURRENT_DIR, 'data')

# Dataset configurations
DATASET_CONFIGS = {
    'mnist': {
        'num_classes': 10,
        'input_channels': 1,
        'input_size': 28,
        'mean': [0.1307],
        'std': [0.3081],
    },
    'cifar10': {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
    },
    'cifar100': {
        'num_classes': 100,
        'input_channels': 3,
        'input_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761],
    },
    'imagenet': {
        'num_classes': 1000,
        'input_channels': 3,
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
}


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Get configuration for a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def get_data_directory() -> str:
    """Get the absolute path to the data directory."""
    return os.path.abspath(DATA_DIR)


def get_transforms(dataset_name: str, train: bool = True, augment: bool = True) -> transforms.Compose:
    """Get data transforms for a dataset."""
    config = get_dataset_config(dataset_name)
    
    if dataset_name == 'mnist':
        if train and augment:
            transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
    
    elif dataset_name in ['cifar10', 'cifar100']:
        if train and augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
    
    elif dataset_name == 'imagenet':
        if train and augment:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(config['mean'], config['std'])
            ])
    
    return transform


def load_mnist(data_dir: str = DATA_DIR, train: bool = True, download: bool = True, 
               augment: bool = True) -> torch.utils.data.Dataset:
    """Load MNIST dataset."""
    transform = get_transforms('mnist', train=train, augment=augment)
    
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    return dataset


def load_cifar10(data_dir: str = DATA_DIR, train: bool = True, download: bool = True,
                 augment: bool = True) -> torch.utils.data.Dataset:
    """Load CIFAR-10 dataset."""
    transform = get_transforms('cifar10', train=train, augment=augment)
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    return dataset


def load_cifar100(data_dir: str = DATA_DIR, train: bool = True, download: bool = True,
                  augment: bool = True) -> torch.utils.data.Dataset:
    """Load CIFAR-100 dataset."""
    transform = get_transforms('cifar100', train=train, augment=augment)
    
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    return dataset


def load_imagenet(data_dir: str = DATA_DIR, train: bool = True, download: bool = False,
                  augment: bool = True) -> torch.utils.data.Dataset:
    """
    Load ImageNet dataset.
    
    Note: ImageNet must be manually downloaded due to registration requirements.
    Expected directory structure:
    data_dir/
        imagenet/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                    ...
                class2/
                ...
            val/
                class1/
                    img1.jpg
                    ...
                class2/
                ...
    """
    if download:
        warnings.warn("ImageNet cannot be automatically downloaded. "
                     "Please download manually from https://image-net.org/")
    
    transform = get_transforms('imagenet', train=train, augment=augment)
    
    split = 'train' if train else 'val'
    imagenet_dir = os.path.join(data_dir, 'imagenet', split)
    
    if not os.path.exists(imagenet_dir):
        raise FileNotFoundError(
            f"ImageNet directory not found: {imagenet_dir}\n"
            "Please download ImageNet manually and organize it in the expected structure."
        )
    
    dataset = torchvision.datasets.ImageFolder(
        root=imagenet_dir,
        transform=transform
    )
    
    return dataset


def get_dataset(dataset_name: str, data_dir: str = DATA_DIR, train: bool = True, 
                download: bool = True, augment: bool = True) -> torch.utils.data.Dataset:
    """
    Get dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'cifar100', 'imagenet')
        data_dir: Directory to store/load data
        train: Whether to load training set (True) or test/validation set (False)
        download: Whether to download the dataset if not present
        augment: Whether to apply data augmentation (only for training)
    
    Returns:
        PyTorch dataset object
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return load_mnist(data_dir, train, download, augment)
    elif dataset_name == 'cifar10':
        return load_cifar10(data_dir, train, download, augment)
    elif dataset_name == 'cifar100':
        return load_cifar100(data_dir, train, download, augment)
    elif dataset_name == 'imagenet':
        return load_imagenet(data_dir, train, download, augment)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(DATASET_CONFIGS.keys())}")


def get_dataloader(dataset_name: str, data_dir: str = DATA_DIR, train: bool = True,
                   batch_size: int = 32, shuffle: Optional[bool] = None, 
                   num_workers: int = 4, download: bool = True, 
                   augment: bool = True) -> DataLoader:
    """
    Get DataLoader for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory to store/load data
        train: Whether to load training set
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle data (defaults to train value)
        num_workers: Number of worker processes for data loading
        download: Whether to download the dataset if not present
        augment: Whether to apply data augmentation (only for training)
    
    Returns:
        PyTorch DataLoader object
    """
    if shuffle is None:
        shuffle = train
    
    dataset = get_dataset(dataset_name, data_dir, train, download, augment)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


def print_dataset_info(dataset_name: str, data_dir: str = DATA_DIR):
    """Print information about a dataset."""
    config = get_dataset_config(dataset_name)
    
    print(f"\n=== {dataset_name.upper()} Dataset Info ===")
    print(f"Number of classes: {config['num_classes']}")
    print(f"Input channels: {config['input_channels']}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")
    print(f"Normalization mean: {config['mean']}")
    print(f"Normalization std: {config['std']}")
    
    try:
        # Try to load a small sample to check if data is available
        train_dataset = get_dataset(dataset_name, data_dir, train=True, download=False)
        test_dataset = get_dataset(dataset_name, data_dir, train=False, download=False)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print(f"Data directory: {get_data_directory()}")
        print("Status: ✓ Available")
        
    except Exception as e:
        print(f"Status: ✗ Not available ({str(e)})")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing PrivaDE Data Loading...")
    print(f"Data directory: {get_data_directory()}")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Test each dataset
    for dataset_name in ['mnist', 'cifar10', 'cifar100', 'imagenet']:
        try:
            print_dataset_info(dataset_name)
            
            if dataset_name != 'imagenet':  # Skip download test for ImageNet
                print(f"\nTesting {dataset_name} loading...")
                
                # Test dataset loading
                train_dataset = get_dataset(dataset_name, train=True, download=True)
                test_dataset = get_dataset(dataset_name, train=False, download=True)
                
                # Test dataloader
                train_loader = get_dataloader(dataset_name, train=True, batch_size=4)
                test_loader = get_dataloader(dataset_name, train=False, batch_size=4)
                
                # Get a sample batch
                train_batch = next(iter(train_loader))
                test_batch = next(iter(test_loader))
                
                print(f"Train batch shape: {train_batch[0].shape}, labels: {train_batch[1].shape}")
                print(f"Test batch shape: {test_batch[0].shape}, labels: {test_batch[1].shape}")
                print("✓ Loading successful")
                
        except Exception as e:
            print(f"✗ Error with {dataset_name}: {e}")
        
        print("-" * 50)
