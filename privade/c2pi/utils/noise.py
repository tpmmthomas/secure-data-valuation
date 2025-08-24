"""
Noise injection utilities for privacy-preserving neural networks.
"""

import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np


def add_uniform_noise(tensor: torch.Tensor, 
                     noise_level: float = 0.1, 
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Add uniform noise to tensor.
    
    Args:
        tensor: Input tensor
        noise_level: Magnitude of noise (fraction of tensor range)
        device: Device to create noise on
    
    Returns:
        Tensor with added uniform noise
    """
    if device is None:
        device = tensor.device
    
    # Generate uniform noise in [-noise_level, noise_level]
    noise = torch.rand_like(tensor, device=device) * 2 * noise_level - noise_level
    
    return tensor + noise


def add_gaussian_noise(tensor: torch.Tensor, 
                      noise_level: float = 0.1,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Add Gaussian noise to tensor.
    
    Args:
        tensor: Input tensor
        noise_level: Standard deviation of noise
        device: Device to create noise on
    
    Returns:
        Tensor with added Gaussian noise
    """
    if device is None:
        device = tensor.device
    
    # Generate Gaussian noise
    noise = torch.randn_like(tensor, device=device) * noise_level
    
    return tensor + noise


def add_laplace_noise(tensor: torch.Tensor,
                     noise_level: float = 0.1,
                     device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Add Laplace (differential privacy) noise to tensor.
    
    Args:
        tensor: Input tensor
        noise_level: Scale parameter for Laplace distribution
        device: Device to create noise on
    
    Returns:
        Tensor with added Laplace noise
    """
    if device is None:
        device = tensor.device
    
    # Generate Laplace noise using exponential distributions
    exp1 = torch.exponential(torch.ones_like(tensor, device=device))
    exp2 = torch.exponential(torch.ones_like(tensor, device=device))
    noise = (exp1 - exp2) * noise_level
    
    return tensor + noise


def calibrate_noise_level(model: nn.Module,
                         data_loader,
                         target_accuracy: float = 0.95,
                         max_noise: float = 1.0,
                         num_levels: int = 10,
                         device: Optional[torch.device] = None) -> float:
    """
    Calibrate noise level to achieve target accuracy.
    
    Args:
        model: Neural network model
        data_loader: Test data loader
        target_accuracy: Target accuracy to maintain
        max_noise: Maximum noise level to test
        num_levels: Number of noise levels to test
        device: Device to run on
    
    Returns:
        Optimal noise level
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Test different noise levels
    noise_levels = np.linspace(0, max_noise, num_levels)
    best_noise = 0.0
    
    for noise_level in noise_levels:
        # Test accuracy with this noise level
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Add noise to input
                noisy_data = add_gaussian_noise(data, noise_level, device)
                
                # Forward pass
                outputs = model(noisy_data)
                predictions = outputs.argmax(dim=1)
                
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        
        if accuracy >= target_accuracy:
            best_noise = noise_level
        else:
            break  # Accuracy dropped too much
    
    return best_noise


class NoiseInjector:
    """
    Class for systematic noise injection at different layers.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 noise_type: str = 'gaussian',
                 noise_level: float = 0.1):
        """
        Initialize noise injector.
        
        Args:
            model: Neural network model
            noise_type: Type of noise ('gaussian', 'uniform', 'laplace')
            noise_level: Magnitude of noise
        """
        self.model = model
        self.noise_type = noise_type
        self.noise_level = noise_level
        
        # Noise functions
        self.noise_functions = {
            'gaussian': add_gaussian_noise,
            'uniform': add_uniform_noise,
            'laplace': add_laplace_noise
        }
        
        if noise_type not in self.noise_functions:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        self.noise_fn = self.noise_functions[noise_type]
        
        # Hooks for layer-wise noise injection
        self.hooks = []
        self.target_layer = None
    
    def inject_at_layer(self, layer_idx: int):
        """
        Set up noise injection at specific layer.
        
        Args:
            layer_idx: Index of layer to inject noise
        """
        self.clear_hooks()
        self.target_layer = layer_idx
        
        # Get all layers
        layers = list(self.model.modules())
        
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(layers)} layers)")
        
        target = layers[layer_idx]
        
        def noise_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                return self.noise_fn(output, self.noise_level)
            elif isinstance(output, (tuple, list)):
                # Handle multiple outputs
                return tuple(self.noise_fn(o, self.noise_level) if isinstance(o, torch.Tensor) else o 
                           for o in output)
            else:
                return output
        
        hook = target.register_forward_hook(noise_hook)
        self.hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all noise injection hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.target_layer = None
    
    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self.clear_hooks()


def measure_noise_impact(model: nn.Module,
                        data_loader,
                        layer_idx: int,
                        noise_levels: list,
                        noise_type: str = 'gaussian',
                        device: Optional[torch.device] = None) -> dict:
    """
    Measure impact of noise at specific layer across different noise levels.
    
    Args:
        model: Neural network model
        data_loader: Test data loader
        layer_idx: Layer to inject noise
        noise_levels: List of noise levels to test
        noise_type: Type of noise to inject
        device: Device to run on
    
    Returns:
        Dictionary with noise level -> accuracy mapping
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = {}
    
    # Baseline accuracy (no noise)
    baseline_accuracy = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
    
    baseline_accuracy = correct / total
    results[0.0] = baseline_accuracy
    
    # Test each noise level
    injector = NoiseInjector(model, noise_type)
    
    for noise_level in noise_levels:
        injector.noise_level = noise_level
        injector.inject_at_layer(layer_idx)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        results[noise_level] = accuracy
    
    injector.clear_hooks()
    
    return results
