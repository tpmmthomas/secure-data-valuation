"""
Configuration classes for C2PI boundary layer finding.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import torch
import json
from pathlib import Path


@dataclass
class C2PIConfig:
    """Configuration for C2PI boundary finding algorithm."""
    
    # Privacy Parameters
    privacy_threshold: float = 0.9  # SSIM threshold for privacy
    ssim_threshold: float = 0.3
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    
    # Device configuration
    device: Union[str, torch.device] = 'auto'  # 'auto', 'cpu', 'cuda', or torch.device
    
    # Attack configuration
    attack_epochs: int = 20
    attack_patience: int = 10  # Early stopping patience
    
    # Evaluation parameters
    max_samples: Optional[int] = None  # Limit dataset size for faster evaluation
    verbose: bool = True
    img_size: int = 28
    
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Handle device configuration
        if isinstance(self.device, str):
            if self.device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.device)
        
        # Validate parameters
        assert 0.0 <= self.privacy_threshold <= 1.0, "Privacy threshold must be in [0, 1]"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'C2PIConfig':
        """Create config from dictionary."""
        # Handle device deserialization
        if 'device' in config_dict and isinstance(config_dict['device'], str):
            if config_dict['device'] not in ['auto', 'cpu', 'cuda']:
                config_dict['device'] = torch.device(config_dict['device'])
        
        return cls(**config_dict)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'C2PIConfig':
        """Load configuration from JSON file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DINAConfig:
    """Configuration for DINA attack."""
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256])
    dropout_rate: float = 0.2
    
    # Training parameters
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss coefficients (α₀=1, α₁=3, αⱼ=2×αⱼ₋₁)
    alpha_coefficients: Optional[List[float]] = None
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    def __post_init__(self):
        """Compute alpha coefficients if not provided."""
        if self.alpha_coefficients is None:
            # α₀=1, α₁=3, αⱼ=2×αⱼ₋₁ for j≥2
            max_layers = 10  # Reasonable default
            self.alpha_coefficients = [1.0, 3.0]
            for j in range(2, max_layers):
                self.alpha_coefficients.append(2 * self.alpha_coefficients[j-1])


@dataclass
class ExperimentConfig:
    """Configuration for complete C2PI experiments."""
    
    # Model and dataset
    model_name: str = 'vgg11'
    dataset_name: str = 'cifar10'
    num_classes: int = 10
    
    # Data loading
    data_root: str = './data'
    batch_size: int = 64
    num_workers: int = 4
    
    # Experiment parameters
    c2pi_config: C2PIConfig = field(default_factory=C2PIConfig)
    dina_config: DINAConfig = field(default_factory=DINAConfig)
    
    # Output
    output_dir: str = './results'
    experiment_name: Optional[str] = None
    
    # Layer selection
    layer_candidates: Optional[List[int]] = None  # Auto-detect if None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_{self.dataset_name}"
        
        # Set dataset-specific parameters
        if self.dataset_name.lower() == 'cifar10':
            self.num_classes = 10
        elif self.dataset_name.lower() == 'cifar100':
            self.num_classes = 100
        elif self.dataset_name.lower() == 'imagenet':
            self.num_classes = 1000
    
    def save(self, path: Union[str, Path]):
        """Save experiment configuration."""
        path = Path(path)
        config_dict = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'num_classes': self.num_classes,
            'data_root': self.data_root,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'layer_candidates': self.layer_candidates,
            'c2pi_config': self.c2pi_config.to_dict(),
            'dina_config': self.dina_config.__dict__
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load experiment configuration."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct nested configs
        c2pi_config = C2PIConfig.from_dict(config_dict.pop('c2pi_config'))
        dina_config = DINAConfig(**config_dict.pop('dina_config'))
        
        return cls(
            c2pi_config=c2pi_config,
            dina_config=dina_config,
            **config_dict
        )


# Convenience function for creating default configs
def create_default_config(model_name: str = 'vgg11', 
                         dataset_name: str = 'cifar10',
                         privacy_threshold: float = 0.3) -> ExperimentConfig:
    """Create a default configuration for common use cases."""
    
    c2pi_config = C2PIConfig(privacy_threshold=privacy_threshold)
    
    return ExperimentConfig(
        model_name=model_name,
        dataset_name=dataset_name,
        c2pi_config=c2pi_config
    )
