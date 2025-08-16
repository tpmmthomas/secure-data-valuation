"""Base dataset classes."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader, Dataset


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders."""
    
    def __init__(self, 
                 data_dir: str = "./data",
                 batch_size: int = 128,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 download: bool = True,
                 **kwargs):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.download = download
        self.kwargs = kwargs
    
    @abstractmethod
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Return train and test dataloaders."""
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return dataset information."""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Return number of classes in the dataset."""
        pass
    
    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, ...]:
        """Return input shape (C, H, W)."""
        pass
