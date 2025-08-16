"""Dataset implementations for computer vision tasks."""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
from typing import Tuple, Dict, Any, Optional
from .base import BaseDatasetLoader


class CIFAR10Dataset(BaseDatasetLoader):
    """CIFAR-10 dataset loader."""
    
    def __init__(self, 
                 augmentation: bool = True,
                 normalize: bool = True,
                 validation_split: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.normalize = normalize
        self.validation_split = validation_split
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        # CIFAR-10 normalization stats
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        
        # Training transforms
        train_transforms = []
        if self.augmentation:
            train_transforms.extend([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(p=0.5),
            ])
        train_transforms.append(T.ToTensor())
        if self.normalize:
            train_transforms.append(T.Normalize(mean, std))
        train_transform = T.Compose(train_transforms)
        
        # Test transforms
        test_transforms = [T.ToTensor()]
        if self.normalize:
            test_transforms.append(T.Normalize(mean, std))
        test_transform = T.Compose(test_transforms)
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=self.download, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=self.download, transform=test_transform
        )
        
        # Validation split
        val_loader = None
        if self.validation_split > 0:
            val_size = int(len(train_dataset) * self.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader, val_loader
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "CIFAR-10",
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "train_samples": 50000,
            "test_samples": 10000,
            "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        }


class CIFAR100Dataset(BaseDatasetLoader):
    """CIFAR-100 dataset loader."""
    
    def __init__(self, 
                 augmentation: bool = True,
                 normalize: bool = True,
                 validation_split: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.normalize = normalize
        self.validation_split = validation_split
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        # CIFAR-100 normalization stats
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        
        # Training transforms
        train_transforms = []
        if self.augmentation:
            train_transforms.extend([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(p=0.5),
            ])
        train_transforms.append(T.ToTensor())
        if self.normalize:
            train_transforms.append(T.Normalize(mean, std))
        train_transform = T.Compose(train_transforms)
        
        # Test transforms
        test_transforms = [T.ToTensor()]
        if self.normalize:
            test_transforms.append(T.Normalize(mean, std))
        test_transform = T.Compose(test_transforms)
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=True, download=self.download, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=False, download=self.download, transform=test_transform
        )
        
        # Validation split
        val_loader = None
        if self.validation_split > 0:
            val_size = int(len(train_dataset) * self.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader, val_loader
    
    @property
    def num_classes(self) -> int:
        return 100
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "CIFAR-100",
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "train_samples": 50000,
            "test_samples": 10000,
        }


class MNISTDataset(BaseDatasetLoader):
    """MNIST dataset loader."""
    
    def __init__(self, 
                 augmentation: bool = False,
                 normalize: bool = True,
                 validation_split: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.normalize = normalize
        self.validation_split = validation_split
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        # Training transforms
        train_transforms = []
        if self.augmentation:
            train_transforms.extend([
                T.RandomRotation(10),
                T.RandomAffine(0, translate=(0.1, 0.1)),
            ])
        train_transforms.append(T.ToTensor())
        if self.normalize:
            train_transforms.append(T.Normalize((0.1307,), (0.3081,)))
        train_transform = T.Compose(train_transforms)
        
        # Test transforms
        test_transforms = [T.ToTensor()]
        if self.normalize:
            test_transforms.append(T.Normalize((0.1307,), (0.3081,)))
        test_transform = T.Compose(test_transforms)
        
        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=self.download, transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=self.download, transform=test_transform
        )
        
        # Validation split
        val_loader = None
        if self.validation_split > 0:
            val_size = int(len(train_dataset) * self.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader, val_loader
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "MNIST",
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "train_samples": 60000,
            "test_samples": 10000,
        }


class FashionMNISTDataset(BaseDatasetLoader):
    """Fashion-MNIST dataset loader."""
    
    def __init__(self, 
                 augmentation: bool = True,
                 normalize: bool = True,
                 validation_split: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.normalize = normalize
        self.validation_split = validation_split
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        # Training transforms
        train_transforms = []
        if self.augmentation:
            train_transforms.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(10),
            ])
        train_transforms.append(T.ToTensor())
        if self.normalize:
            train_transforms.append(T.Normalize((0.2860,), (0.3530,)))
        train_transform = T.Compose(train_transforms)
        
        # Test transforms
        test_transforms = [T.ToTensor()]
        if self.normalize:
            test_transforms.append(T.Normalize((0.2860,), (0.3530,)))
        test_transform = T.Compose(test_transforms)
        
        # Load datasets
        train_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=self.download, transform=train_transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=self.download, transform=test_transform
        )
        
        # Validation split
        val_loader = None
        if self.validation_split > 0:
            val_size = int(len(train_dataset) * self.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader, val_loader
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (1, 28, 28)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "Fashion-MNIST",
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "train_samples": 60000,
            "test_samples": 10000,
            "classes": ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        }


class SVHNDataset(BaseDatasetLoader):
    """SVHN dataset loader."""
    
    def __init__(self, 
                 augmentation: bool = True,
                 normalize: bool = True,
                 validation_split: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.augmentation = augmentation
        self.normalize = normalize
        self.validation_split = validation_split
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        # SVHN normalization stats
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        
        # Training transforms
        train_transforms = []
        if self.augmentation:
            train_transforms.extend([
                T.RandomCrop(32, padding=4),
            ])
        train_transforms.append(T.ToTensor())
        if self.normalize:
            train_transforms.append(T.Normalize(mean, std))
        train_transform = T.Compose(train_transforms)
        
        # Test transforms
        test_transforms = [T.ToTensor()]
        if self.normalize:
            test_transforms.append(T.Normalize(mean, std))
        test_transform = T.Compose(test_transforms)
        
        # Load datasets
        train_dataset = torchvision.datasets.SVHN(
            root=self.data_dir, split='train', download=self.download, transform=train_transform
        )
        test_dataset = torchvision.datasets.SVHN(
            root=self.data_dir, split='test', download=self.download, transform=test_transform
        )
        
        # Validation split
        val_loader = None
        if self.validation_split > 0:
            val_size = int(len(train_dataset) * self.validation_split)
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory
            )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader, val_loader
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        return (3, 32, 32)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "SVHN",
            "num_classes": self.num_classes,
            "input_shape": self.input_shape,
            "train_samples": 73257,
            "test_samples": 26032,
        }


# Dataset registry
DATASETS = {
    'cifar10': CIFAR10Dataset,
    'cifar100': CIFAR100Dataset,
    'mnist': MNISTDataset,
    'fashion_mnist': FashionMNISTDataset,
    'svhn': SVHNDataset,
}


def create_dataset(dataset_name: str, **kwargs) -> BaseDatasetLoader:
    """Factory function to create datasets."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASETS.keys())}")
    
    return DATASETS[dataset_name](**kwargs)
