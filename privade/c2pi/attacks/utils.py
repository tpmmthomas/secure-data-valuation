"""
Attack utilities and result structures.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch


@dataclass
class AttackResult:
    """Structure to hold attack evaluation results."""
    ssim_scores: np.ndarray
    mse_scores: np.ndarray
    avg_ssim: float
    avg_mse: float
    success_rate: float
    
    def __post_init__(self):
        """Calculate additional statistics."""
        self.std_ssim = np.std(self.ssim_scores)
        self.std_mse = np.std(self.mse_scores)
        self.median_ssim = np.median(self.ssim_scores)
        self.median_mse = np.median(self.mse_scores)


def evaluate_attack_success(ssim_scores: np.ndarray, threshold: float = 0.3) -> dict:
    """
    Evaluate attack success based on SSIM threshold.
    
    Args:
        ssim_scores: Array of SSIM scores
        threshold: SSIM threshold for success
    
    Returns:
        Dictionary with success metrics
    """
    success_mask = ssim_scores >= threshold
    success_rate = np.mean(success_mask)
    
    return {
        "success_rate": success_rate,
        "failure_rate": 1 - success_rate,
        "num_successful": np.sum(success_mask),
        "num_failed": np.sum(~success_mask),
        "total_samples": len(ssim_scores),
        "avg_ssim_successful": np.mean(ssim_scores[success_mask]) if np.any(success_mask) else 0.0,
        "avg_ssim_failed": np.mean(ssim_scores[~success_mask]) if np.any(~success_mask) else 0.0
    }


def normalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Normalize tensor with given mean and std (ImageNet normalization)."""
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def denormalize_tensor(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Denormalize tensor with given mean and std."""
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    return tensor * std + mean


def clip_and_convert_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """Clip tensor to [0, 1] and convert to uint8 range."""
    return torch.clamp(tensor, 0, 1) * 255


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy image array (H, W, C)."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.cpu().permute(1, 2, 0).numpy()


class ActivationHook:
    """Helper class to capture intermediate activations."""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hook(self, module: torch.nn.Module, name: str):
        """Register forward hook on module."""
        def hook_fn(module, input, output):
            self.activations[name] = output.detach()
        
        hook = module.register_forward_hook(hook_fn)
        self.hooks.append(hook)
        return hook
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activation(self, name: str) -> Optional[torch.Tensor]:
        """Get activation by name."""
        return self.activations.get(name)


def get_layer_output_shape(model: torch.nn.Module, input_shape: tuple, layer_idx: int) -> tuple:
    """Get output shape of a specific layer."""
    dummy_input = torch.randn(1, *input_shape)
    
    # Forward through model up to target layer
    x = dummy_input
    if hasattr(model, 'features'):
        for i, layer in enumerate(model.features):
            x = layer(x)
            if i == layer_idx:
                break
    
    return tuple(x.shape[1:])  # Remove batch dimension


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
